"""
Backtesting engine — validate strategies against historical data before risking real money.

Usage:
  python backtester.py                     # backtest last 3 months
  python backtester.py --months 6          # backtest last 6 months
  python backtester.py --months 12 --plot  # backtest 1 year + save equity chart

Simulates the full pipeline:
  1. Fetch historical OHLCV data
  2. Compute indicators (same as live market_data.py)
  3. Run ML model predictions (if trained)
  4. Apply risk manager sizing (Kelly + ATR)
  5. Simulate trades with stop-loss / take-profit execution
  6. Calculate performance metrics (Sharpe, drawdown, win rate, etc.)

Does NOT call Claude API — uses ML predictions + simple rule-based decisions
to test the quantitative edge independently of LLM reasoning.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import risk_manager
from ml_signal import (
    engineer_features, triple_barrier_label, detect_regime, encode_regime,
    fetch_ohlcv_paginated,
    LOOKAHEAD_HOURS, PROFIT_TARGET_PCT, STOP_LOSS_PCT,
)

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 200.0
TRADING_FEE_PCT = 0.1   # Binance spot taker fee: 0.1% per side
SLIPPAGE_PCT = 0.05     # Estimated market-impact slippage: 0.05% per side
# Combined round-trip cost = 2 × (fee + slippage) = 2 × 0.15% = 0.30%


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime | None = None
    action: str = ""
    entry_price: float = 0
    exit_price: float = 0
    amount_usd: float = 0
    btc_qty: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    pnl_usd: float = 0         # net P&L after fees + slippage
    pnl_pct: float = 0
    gross_pnl_usd: float = 0   # gross P&L before any costs
    fees_paid_usd: float = 0   # total fees + slippage for this trade
    exit_reason: str = ""      # "take_profit", "stop_loss", "time_exit"
    signal_prob: float = 0     # model probability that triggered the entry


@dataclass
class BacktestResult:
    total_return_pct: float = 0         # net return (after fees + slippage)
    total_return_gross_pct: float = 0   # gross return (before any costs)
    total_fees_usd: float = 0           # total fees + slippage paid
    buy_and_hold_pct: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    max_drawdown_pct: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    total_trades: int = 0
    avg_trade_pnl_pct: float = 0
    avg_win_pct: float = 0
    avg_loss_pct: float = 0
    best_trade_pct: float = 0
    worst_trade_pct: float = 0
    wins: int = 0
    losses: int = 0
    holds: int = 0
    avg_hold_hours: float = 0
    final_equity: float = 0
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)


def fetch_backtest_data(exchange, months: int = 3):
    """
    Fetch 1h/4h/1d history for backtesting.

    Always pulls from production Binance (public market data, no keys):
    the sandbox exchange holds only ~40 days of candles, which silently
    shrank every "6 month" backtest to 6 weeks. The 4h/1d frames are what
    the trained model expects — without them 25/56 features were zero-filled.
    """
    import ccxt
    data_exchange = ccxt.binance({
        "enableRateLimit": True, "options": {"defaultType": "spot"},
    })

    import time
    hours = months * 30 * 24
    df_1h = fetch_ohlcv_paginated(data_exchange, "BTC/USDT", "1h", total_candles=hours)
    time.sleep(1)
    # +60 bars of warm-up so 50-period indicators are valid from the start
    df_4h = fetch_ohlcv_paginated(data_exchange, "BTC/USDT", "4h", total_candles=hours // 4 + 60)
    time.sleep(1)
    df_1d = fetch_ohlcv_paginated(data_exchange, "BTC/USDT", "1d", total_candles=months * 30 + 60)
    return df_1h, df_4h, df_1d


def _simulate_trade_exit(candles: pd.DataFrame, entry_idx: int,
                         entry_price: float, action: str,
                         stop_loss: float, take_profit: float,
                         max_bars: int = LOOKAHEAD_HOURS) -> tuple[int, float, str]:
    """
    Simulate price hitting stop-loss, take-profit, or time expiry.
    Returns (exit_idx, exit_price, reason).
    """
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(candles))):
        high = candles.iloc[i]["high"]
        low = candles.iloc[i]["low"]
        close = candles.iloc[i]["close"]

        if action == "buy":
            if low <= stop_loss:
                return i, stop_loss, "stop_loss"
            if high >= take_profit:
                return i, take_profit, "take_profit"
        elif action == "sell":
            if high >= stop_loss:
                return i, stop_loss, "stop_loss"
            if low <= take_profit:
                return i, take_profit, "take_profit"

    # Time exit — use closing price at end of window
    exit_idx = min(entry_idx + max_bars, len(candles) - 1)
    return exit_idx, candles.iloc[exit_idx]["close"], "time_exit"


# Sizing inputs for simulations: risk_manager defaults read the LIVE
# database (recent trades, consecutive losses, RL state) which must not
# leak into a backtest.
BACKTEST_TRADE_STATS = {
    "win_rate": 0.5, "avg_win_pct": 1.5, "avg_loss_pct": 1.5,
    "total_trades": 0, "wins": 0, "losses": 0,
}


def run_backtest(exchange, months: int = 3,
                 model_path=None, meta_path=None,
                 placebo_lag: int = 0,
                 offset_months: int = 0) -> BacktestResult:
    """
    Run full backtest simulation.
    Uses ML model predictions if available, otherwise falls back to
    simple indicator-based signals. Pass model_path/meta_path to simulate
    with a holdout-trained model instead of the live one.

    placebo_lag > 0 shifts every prediction that many bars into the past
    (the signal for bar i was computed at bar i−lag). A genuine edge must
    die under a 24-bar lag; results that survive prove harness leakage.
    """
    logger.info("Backtest: Fetching %d months of data...", months + offset_months)
    df_1h, df_4h, df_1d = fetch_backtest_data(exchange, months + offset_months)
    if len(df_1h) < 500:
        logger.error("Backtest: Not enough data (%d candles)", len(df_1h))
        return BacktestResult()

    logger.info("Backtest: 1h=%d, 4h=%d, 1d=%d candles loaded. Engineering features...",
                len(df_1h), len(df_4h), len(df_1d))
    df = engineer_features(df_1h, df_4h, df_1d)

    # Load the model before regime detection so the backtest labels regimes
    # the same way training did (HMM vs causal rules — see detect_regime).
    ensemble, meta = None, None
    try:
        from ml_signal import _load_model
        ensemble, meta = _load_model(model_path, meta_path)
    except Exception as exc:
        logger.info("Backtest: model load failed: %s", exc)

    regime_method = (meta or {}).get("regime_method", "auto")
    df["regime"] = detect_regime(df, method=regime_method)
    # Same regime one-hots the model was trained with (regime_is_* features)
    regime_df = encode_regime(df["regime"])
    for col in regime_df.columns:
        df[col] = regime_df[col]

    # Use trained ML model predictions if available
    ml_available = False
    buy_gate = sell_gate = 0.55  # fallback for indicator signals (prob set to 0.65)
    try:
        if ensemble and meta:
            selected = ensemble["selected_features"]
            missing = [f for f in selected if f not in df.columns]
            if missing:
                # Zero-filling model inputs silently invalidates the backtest —
                # make it impossible to miss.
                logger.warning(
                    "Backtest: %d/%d model features MISSING from the feature "
                    "pipeline and zero-filled — results are unreliable: %s",
                    len(missing), len(selected), missing,
                )
            for f in missing:
                df[f] = 0
            # Batch predict
            X = df[selected].fillna(0)
            import numpy as np
            xgb_proba = ensemble["xgb"].predict_proba(X)
            lgb_proba = ensemble["lgb"].predict_proba(X)
            stack = np.hstack([xgb_proba, lgb_proba])
            final_proba = ensemble["meta"].predict_proba(stack)
            df["ml_buy_prob"] = final_proba[:, 2] if final_proba.shape[1] > 2 else 0
            df["ml_sell_prob"] = final_proba[:, 0] if final_proba.shape[1] > 0 else 0
            ml_available = True
            # Gates derived from OOS expected value at train time — a fixed
            # 0.55 sits ~4x above what a 6%-prevalence class can produce.
            # None means the side had no positive-EV gate → disabled (2.0
            # can never fire; probabilities are bounded by 1).
            buy_gate = ensemble.get("buy_threshold")
            sell_gate = ensemble.get("sell_threshold")
            if buy_gate is None:
                logger.warning("Backtest: BUY side DISABLED — no positive-EV gate at train time")
                buy_gate = 2.0
            if sell_gate is None:
                logger.warning("Backtest: SELL side DISABLED — no positive-EV gate at train time")
                sell_gate = 2.0
            logger.info(
                "Backtest: ML model loaded — ensemble predictions, "
                "gates buy>%.3f sell>%.3f", buy_gate, sell_gate,
            )
    except Exception as exc:
        logger.info("Backtest: No ML model — using indicator signals: %s", exc)

    if placebo_lag > 0 and ml_available:
        df["ml_buy_prob"] = df["ml_buy_prob"].shift(placebo_lag).fillna(0)
        df["ml_sell_prob"] = df["ml_sell_prob"].shift(placebo_lag).fillna(0)
        logger.warning(
            "Backtest: PLACEBO — predictions lagged %d bars. "
            "A real edge must die here; strong results = harness leakage.",
            placebo_lag,
        )

    # If no ML, use simple signal rules
    if not ml_available:
        rsi = df.get("h1_rsi_14", pd.Series(50, index=df.index))
        macd_hist = df.get("h1_macd_hist", pd.Series(0, index=df.index))
        bb_pos = df.get("h1_bb_pos", pd.Series(0.5, index=df.index))

        df["ml_buy_prob"] = 0.0
        df["ml_sell_prob"] = 0.0

        buy_mask = (rsi < 35) & (macd_hist > 0) & (bb_pos < 0.2)
        sell_mask = (rsi > 70) & (macd_hist < 0) & (bb_pos > 0.8)
        df.loc[buy_mask, "ml_buy_prob"] = 0.65
        df.loc[sell_mask, "ml_sell_prob"] = 0.65

    # Windowed backtest: drop the most recent offset_months so runs with
    # different offsets form DISJOINT evidence windows (PRINCIPLES gate G2).
    if offset_months > 0:
        end_ts = int(df["ts"].max()) - offset_months * 30 * 24 * 3600 * 1000
        n_before = len(df)
        df = df[df["ts"] <= end_ts].reset_index(drop=True)
        logger.info(
            "Backtest: OFFSET — window shifted %d months into the past "
            "(%d→%d bars, window ends %s)",
            offset_months, n_before, len(df),
            pd.Timestamp(end_ts, unit="ms").date(),
        )

    # ── Simulation loop ──
    usdt = INITIAL_CAPITAL
    btc = 0.0
    trades: list[Trade] = []
    equity_curve = []
    cooldown = 0

    logger.info("Backtest: Running simulation...")
    atr_col = df.get("h1_atr_pct", pd.Series(1.5, index=df.index))

    for i in range(100, len(df) - LOOKAHEAD_HOURS - 1):
        price = df.iloc[i]["close"]
        total_equity = usdt + btc * price
        equity_curve.append({"ts": df.iloc[i]["ts"], "equity": total_equity})

        if cooldown > 0:
            cooldown -= 1
            continue

        buy_prob = df.iloc[i]["ml_buy_prob"]
        sell_prob = df.iloc[i]["ml_sell_prob"]
        atr_pct = atr_col.iloc[i] if i < len(atr_col) else 1.5
        atr_val = price * atr_pct / 100

        # Decision logic
        action = "hold"
        confidence = 0.5
        btc_alloc = (btc * price) / total_equity if total_equity > 0 else 0

        # Confidence for sizing: normalise signal strength relative to its
        # gate (risk_manager.confidence_multiplier expects ~0.5–0.95, but the
        # calibrated gates for a 6%-prevalence class sit far below that).
        # prob == gate → 0.5x sizing floor; prob == 2×gate → 0.95 full size.
        if buy_prob > buy_gate and btc_alloc < config.MAX_BTC_ALLOC_PCT:
            action = "buy"
            confidence = min(0.95, 0.5 + 0.45 * (buy_prob - buy_gate) / max(buy_gate, 1e-9))
        elif sell_prob > sell_gate and btc > 0:
            action = "sell"
            confidence = min(0.95, 0.5 + 0.45 * (sell_prob - sell_gate) / max(sell_gate, 1e-9))

        if action == "hold":
            continue

        # Realistic fill: the signal is computed on bar i's CLOSE, so the
        # earliest executable price is the NEXT bar's open. Filling at bar
        # i's own close is a look-ahead-flavored optimism the live bot
        # cannot reproduce.
        entry_price = df.iloc[i + 1]["open"]

        # Risk-managed sizing (stops/targets anchored to the actual fill)
        snapshot = {"price": entry_price, "atr": atr_val, "atr_pct": atr_pct}
        portfolio = {"usdt": usdt, "btc": btc}
        risk = risk_manager.assess_trade(
            action, confidence, snapshot, portfolio,
            stats=BACKTEST_TRADE_STATS, use_rl=False, consecutive_losses=0,
        )
        amount = risk.recommended_usd

        # Cost rate per side: taker fee + slippage
        cost_pct_per_side = (TRADING_FEE_PCT + SLIPPAGE_PCT) / 100

        if action == "buy" and usdt >= amount + 0.5:
            entry_cost = amount * cost_pct_per_side
            btc_qty = (amount - entry_cost) / entry_price
            usdt -= amount
            btc += btc_qty

            # entry at bar i+1's open; barrier scan starts at bar i+1 too
            exit_idx, exit_price, reason = _simulate_trade_exit(
                df, i, entry_price, "buy", risk.stop_loss_price, risk.take_profit_price,
            )

            exit_cost = btc_qty * exit_price * cost_pct_per_side
            gross_pnl = btc_qty * (exit_price - entry_price)       # before all costs
            net_pnl = gross_pnl - entry_cost - exit_cost
            pnl_pct = (exit_price / entry_price - 1) * 100

            usdt += btc_qty * exit_price - exit_cost
            btc -= btc_qty

            trades.append(Trade(
                entry_time=datetime.fromtimestamp(df.iloc[i + 1]["ts"] / 1000, tz=timezone.utc),
                exit_time=datetime.fromtimestamp(df.iloc[exit_idx]["ts"] / 1000, tz=timezone.utc),
                action="buy", entry_price=entry_price, exit_price=exit_price,
                amount_usd=amount, btc_qty=btc_qty,
                stop_loss=risk.stop_loss_price, take_profit=risk.take_profit_price,
                pnl_usd=net_pnl, gross_pnl_usd=gross_pnl,
                fees_paid_usd=entry_cost + exit_cost,
                pnl_pct=pnl_pct, exit_reason=reason, signal_prob=buy_prob,
            ))
            cooldown = LOOKAHEAD_HOURS

        elif action == "sell" and btc > 0:
            sell_btc = min(btc, amount / entry_price)
            entry_cost = sell_btc * entry_price * cost_pct_per_side
            usdt += sell_btc * entry_price - entry_cost
            btc -= sell_btc

            exit_idx, exit_price, reason = _simulate_trade_exit(
                df, i, entry_price, "sell", risk.stop_loss_price, risk.take_profit_price,
            )

            exit_cost = sell_btc * exit_price * cost_pct_per_side
            gross_pnl = sell_btc * (entry_price - exit_price)
            net_pnl = gross_pnl - entry_cost - exit_cost
            pnl_pct = (entry_price / exit_price - 1) * 100 if exit_price > 0 else 0

            trades.append(Trade(
                entry_time=datetime.fromtimestamp(df.iloc[i + 1]["ts"] / 1000, tz=timezone.utc),
                exit_time=datetime.fromtimestamp(df.iloc[exit_idx]["ts"] / 1000, tz=timezone.utc),
                action="sell", entry_price=entry_price, exit_price=exit_price,
                amount_usd=sell_btc * entry_price, btc_qty=sell_btc,
                stop_loss=risk.stop_loss_price, take_profit=risk.take_profit_price,
                pnl_usd=net_pnl, gross_pnl_usd=gross_pnl,
                fees_paid_usd=entry_cost + exit_cost,
                pnl_pct=pnl_pct, exit_reason=reason, signal_prob=sell_prob,
            ))
            cooldown = LOOKAHEAD_HOURS

    # Final equity
    final_price = df.iloc[-1]["close"]
    final_equity = usdt + btc * final_price

    # ── Calculate metrics ──
    result = BacktestResult()
    result.final_equity = round(final_equity, 2)
    result.equity_curve = equity_curve
    result.trades = trades
    result.total_trades = len(trades)

    # Buy and hold benchmark
    start_price = df.iloc[100]["close"]
    end_price = df.iloc[-1]["close"]
    result.buy_and_hold_pct = round((end_price / start_price - 1) * 100, 2)
    result.total_return_pct = round((final_equity / INITIAL_CAPITAL - 1) * 100, 2)

    # Gross return and total fees paid
    total_fees = sum(t.fees_paid_usd for t in trades)
    gross_equity = final_equity + total_fees  # what equity would be without fees
    result.total_fees_usd = round(total_fees, 2)
    result.total_return_gross_pct = round((gross_equity / INITIAL_CAPITAL - 1) * 100, 2)

    if trades:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        result.wins = len(wins)
        result.losses = len(losses)
        result.win_rate = round(len(wins) / len(trades), 3) if trades else 0

        pnls = [t.pnl_pct for t in trades]
        result.avg_trade_pnl_pct = round(np.mean(pnls), 3) if pnls else 0
        result.avg_win_pct = round(np.mean([t.pnl_pct for t in wins]), 3) if wins else 0
        result.avg_loss_pct = round(np.mean([t.pnl_pct for t in losses]), 3) if losses else 0
        result.best_trade_pct = round(max(pnls), 3) if pnls else 0
        result.worst_trade_pct = round(min(pnls), 3) if pnls else 0

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        result.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999

        hold_hours = [(t.exit_time - t.entry_time).total_seconds() / 3600
                      for t in trades if t.exit_time]
        result.avg_hold_hours = round(np.mean(hold_hours), 1) if hold_hours else 0

    # Sharpe and Sortino from equity curve
    if len(equity_curve) > 10:
        equities = pd.Series([e["equity"] for e in equity_curve])
        returns = equities.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            # Annualize: hourly returns × sqrt(8760 hours/year)
            result.sharpe_ratio = round(
                returns.mean() / returns.std() * np.sqrt(8760), 2
            )
            downside = returns[returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                result.sortino_ratio = round(
                    returns.mean() / downside.std() * np.sqrt(8760), 2
                )

    # Max drawdown
    if equity_curve:
        equities = [e["equity"] for e in equity_curve]
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)
        result.max_drawdown_pct = round(max_dd, 2)

    return result


def print_report(r: BacktestResult):
    """Print backtest results to console."""
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Starting capital:   ${INITIAL_CAPITAL:.2f}")
    print(f"  Final equity:       ${r.final_equity:.2f}")
    print(f"  Bot return (net):   {r.total_return_pct:+.2f}%   ← after fees & slippage")
    print(f"  Bot return (gross): {r.total_return_gross_pct:+.2f}%   ← before costs")
    print(f"  Total fees paid:    ${r.total_fees_usd:.2f}  "
          f"(fee={TRADING_FEE_PCT}% + slip={SLIPPAGE_PCT}% per side)")
    print(f"  Buy & hold return:  {r.buy_and_hold_pct:+.2f}%")
    print(f"  Alpha vs B&H:       {r.total_return_pct - r.buy_and_hold_pct:+.2f}%  (net)")
    print("-" * 60)
    print(f"  Total trades:       {r.total_trades}")
    print(f"  Wins / Losses:      {r.wins} / {r.losses}")
    print(f"  Win rate:           {r.win_rate:.1%}")
    print(f"  Profit factor:      {r.profit_factor:.2f}")
    print(f"  Avg hold time:      {r.avg_hold_hours:.1f}h")
    print("-" * 60)
    print(f"  Avg trade P&L:      {r.avg_trade_pnl_pct:+.3f}%")
    print(f"  Avg win:            {r.avg_win_pct:+.3f}%")
    print(f"  Avg loss:           {r.avg_loss_pct:+.3f}%")
    print(f"  Best trade:         {r.best_trade_pct:+.3f}%")
    print(f"  Worst trade:        {r.worst_trade_pct:+.3f}%")
    print("-" * 60)
    print(f"  Sharpe ratio:       {r.sharpe_ratio:.2f}")
    print(f"  Sortino ratio:      {r.sortino_ratio:.2f}")
    print(f"  Max drawdown:       {r.max_drawdown_pct:.2f}%")
    print("=" * 60)

    if r.sharpe_ratio >= 1.5:
        print("  Assessment: STRONG — strategy shows clear edge")
    elif r.sharpe_ratio >= 1.0:
        print("  Assessment: GOOD — positive risk-adjusted returns")
    elif r.sharpe_ratio >= 0.5:
        print("  Assessment: MARGINAL — edge exists but thin")
    else:
        print("  Assessment: WEAK — insufficient edge, review strategy")
    print()

    # Exit reason breakdown
    if r.trades:
        reasons = {}
        for t in r.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("  Exit reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(r.trades) * 100
            print(f"    {reason:15s} {count:4d} ({pct:.1f}%)")
        print()


def save_trades_csv(r: BacktestResult, path: str = "backtest_trades.csv"):
    """Full trade ledger for hand-auditing entries against the real chart."""
    if not r.trades:
        return
    rows = [{
        "entry_time": t.entry_time.isoformat(),
        "exit_time": t.exit_time.isoformat() if t.exit_time else "",
        "action": t.action,
        "signal_prob": round(t.signal_prob, 4),
        "entry_price": round(t.entry_price, 2),
        "exit_price": round(t.exit_price, 2),
        "stop_loss": round(t.stop_loss, 2),
        "take_profit": round(t.take_profit, 2),
        "amount_usd": round(t.amount_usd, 2),
        "pnl_pct": round(t.pnl_pct, 4),
        "pnl_usd": round(t.pnl_usd, 4),
        "fees_paid_usd": round(t.fees_paid_usd, 4),
        "exit_reason": t.exit_reason,
    } for t in r.trades]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Trade ledger saved to {path} ({len(rows)} trades)")


def save_equity_csv(r: BacktestResult, path: str = "backtest_equity.csv"):
    """Save equity curve to CSV for charting."""
    if not r.equity_curve:
        return
    df = pd.DataFrame(r.equity_curve)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.to_csv(path, index=False)
    print(f"  Equity curve saved to {path}")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Backtest the trading bot strategy")
    parser.add_argument("--months", type=int, default=3, help="Months of history to test")
    parser.add_argument("--csv", action="store_true", help="Save equity curve CSV")
    parser.add_argument("--no-db", action="store_true", help="Skip saving results to the dashboard DB")
    parser.add_argument("--holdout", action="store_true",
                        help="Out-of-sample: retrain a model on data ending --months ago, "
                             "then backtest those months with candles the model never saw")
    parser.add_argument("--placebo", type=int, nargs="?", const=24, default=0,
                        metavar="BARS",
                        help="Leakage test: lag predictions by BARS (default 24). "
                             "A real edge must die; good results here = harness bug")
    parser.add_argument("--offset", type=int, default=0, metavar="MONTHS",
                        help="Shift the backtest window MONTHS into the past so runs "
                             "with different offsets form disjoint evidence windows "
                             "(e.g. --months 3 --offset 3 tests months 6..3 ago)")
    args = parser.parse_args()

    import ccxt
    exchange = ccxt.binance({
        "apiKey": config.BINANCE_API_KEY,
        "secret": config.BINANCE_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    if config.TESTNET:
        exchange.set_sandbox_mode(True)

    model_path = meta_path = None
    if args.holdout:
        from ml_signal import MODEL_DIR, train_model
        model_path = MODEL_DIR / "btc_ensemble_v2_holdout.joblib"
        meta_path = MODEL_DIR / "model_meta_v2_holdout.joblib"
        train_cutoff = args.months + args.offset
        print(f"\n=== HOLDOUT: training on data ending {train_cutoff} months ago "
              "(causal rule-based regimes; live model untouched) ===")
        train_model(None, cutoff_months=train_cutoff, regime_method="rules",
                    model_path=model_path, meta_path=meta_path)
        print("=== Training complete — running OUT-OF-SAMPLE simulation ===\n")

    result = run_backtest(exchange, months=args.months,
                          model_path=model_path, meta_path=meta_path,
                          placebo_lag=args.placebo,
                          offset_months=args.offset)
    print_report(result)
    if args.holdout:
        print("  NOTE: out-of-sample run — the model never saw these candles.")
    if args.placebo:
        print(f"  PLACEBO RUN (predictions lagged {args.placebo} bars): "
              "strong results here mean the harness leaks — not an edge.")

    save_trades_csv(result)
    if args.csv:
        save_equity_csv(result)

    if not args.no_db and not args.placebo:
        try:
            import database
            curve = result.equity_curve
            day = lambda point: pd.Timestamp(point["ts"], unit="ms").strftime("%Y-%m-%d")
            database.save_backtest_run(
                period_months=args.months,
                start_date=day(curve[0]) if curve else None,
                end_date=day(curve[-1]) if curve else None,
                total_trades=result.total_trades,
                wins=result.wins,
                losses=result.losses,
                win_rate=result.win_rate,
                sharpe_ratio=result.sharpe_ratio,
                sortino_ratio=result.sortino_ratio,
                max_drawdown_pct=result.max_drawdown_pct,
                profit_factor=result.profit_factor,
                total_return_pct=result.total_return_pct,
                equity_curve=[round(p["equity"], 2) for p in curve],
                config_snapshot={
                    "months": args.months,
                    "initial_capital": INITIAL_CAPITAL,
                    "trading_fee_pct": TRADING_FEE_PCT,
                    "slippage_pct": SLIPPAGE_PCT,
                    "testnet": bool(config.TESTNET),
                    "holdout": bool(args.holdout),
                    "offset_months": args.offset,
                    "entry_fill": "next_bar_open",
                },
            )
            print("  Results saved — visible on the dashboard Backtests page.")
        except Exception as exc:
            print(f"  WARNING: could not save results to dashboard DB: {exc}")
