"""
Risk management engine — the most important module in the bot.

Handles:
  1. Kelly Criterion position sizing (optimal bet based on win rate + payoff)
  2. ATR-based volatility scaling (reduce size in chaos, increase in calm)
  3. Per-trade stop-loss / take-profit calculation (ATR-based)
  4. Trailing stop management
  5. Portfolio-level exposure limits
  6. Trade history stats for Kelly formula inputs

A 55% accurate model with proper sizing beats a 65% model with fixed sizing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import database as db
import config
import rl_position
import trade_pnl

logger = logging.getLogger(__name__)


@dataclass
class TradeRisk:
    """Output of risk assessment for a single trade."""
    recommended_usd: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_distance: float
    risk_reward_ratio: float
    kelly_fraction: float
    atr_multiplier: float
    position_rationale: str


# ── Trade History Statistics ─────────────────────────────────────────────────

def _get_trade_stats() -> dict:
    """
    Win rate + avg win/loss PERCENTAGES from recent trades, for Kelly.

    Previously this counted the ±2% correct/wrong label and — critically —
    put each trade's POSITION SIZE ($) into avg_win_pct/avg_loss_pct. That
    made Kelly's payoff ratio b = avg_win/avg_loss ≈ 1.0 always (every
    position ~$6) and win_rate ~0.5, so Kelly was pinned at ~0 by
    construction and position sizing never responded to a real edge.

    Now it uses each trade's NET directional P&L % (trade_pnl), so the
    inputs are genuine percentages and Kelly reflects the actual edge.
    """
    trades = db.get_recent_trades(50)
    default = {"win_rate": 0.5, "avg_win_pct": 1.5, "avg_loss_pct": 1.5,
               "total_trades": 0, "wins": 0, "losses": 0}
    if not trades:
        return default

    wins, losses = [], []   # win/loss magnitudes in PERCENT
    for t in trades:
        pnl = trade_pnl.decision_pnl_pct(t.get("action"), t.get("price"), t.get("price_after_4h"))
        if pnl is None:
            continue
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(-pnl)

    total = len(wins) + len(losses)
    if total == 0:
        return {**default, "total_trades": len(trades)}

    return {
        "win_rate": round(len(wins) / total, 3),
        "avg_win_pct": round(sum(wins) / len(wins), 2) if wins else 1.5,
        "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else 1.5,
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
    }


# ── Kelly Criterion ──────────────────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   total_trades: int = 100) -> float:
    """
    Calculate Kelly fraction: optimal bet size for maximum long-term growth.
    f* = (p * b - q) / b

    Uses Quarter-Kelly (f * 0.25) — crypto-appropriate for high volatility.
    Full Kelly and even Half-Kelly produce dangerously large sizes on BTC/ETH
    (often 40-120% of account). Quarter-Kelly cuts volatility 75% while
    retaining ~60% of theoretical long-run growth.

    Win rate is discounted by 10pp to account for overconfidence bias in
    short trade histories. Strategy must have 50+ trades for reliable inputs.
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0

    b = avg_win / avg_loss  # payoff ratio
    # Discount win rate 10pp to guard against overconfidence / small samples
    p = max(0.0, win_rate - 0.10)
    q = 1 - p

    f = (p * b - q) / b

    # Quarter-Kelly — the crypto-standard fractional Kelly for high-vol assets
    f = f * 0.25

    # Below 50 trades the win-rate estimate is noisy — extra-conservative cap
    if total_trades < 50:
        f = min(f, 0.03)

    # Hard cap: never exceed 20% of portfolio regardless of Kelly output
    return max(0.0, min(0.20, f))


# ── Consecutive Loss Detector ────────────────────────────────────────────────

def _get_consecutive_losses() -> int:
    """Count trailing consecutive 'wrong' outcomes from recent evaluated trades."""
    trades = db.get_recent_trades(10)
    count = 0
    for t in trades:
        outcome = t.get("outcome")
        if outcome == "wrong":
            count += 1
        elif outcome == "correct":
            break
        # trades with no outcome yet (unevaluated) are skipped
    return count


# ── Regime-Aware ATR Stop Multipliers ────────────────────────────────────────

def _atr_stop_multipliers(atr_pct: float, consecutive_losses: int) -> tuple[float, float]:
    """
    Return (sl_multiplier, tp_multiplier) tuned to the current volatility regime.

    Rules from 2025-2026 research:
      Consecutive-loss mode  → tighter stops to limit further damage
      High vol (ATR > 3%)    → wider stops to avoid premature shake-out
      Normal (ATR 1.5-3%)    → standard swing multipliers
      Calm (ATR < 1.5%)      → standard stops, smaller ATR so absolute $ still small
    """
    if consecutive_losses >= 3:
        return 1.5, 2.5   # tighten when bot is underperforming
    elif atr_pct > 3.0:
        return 3.0, 4.5   # high volatility — wider stops needed
    elif atr_pct > 1.5:
        return 2.0, 3.0   # normal swing trading
    else:
        return 1.5, 2.5   # calm market


# ── ATR Volatility Scaling ───────────────────────────────────────────────────

def atr_scale_factor(atr_pct: float) -> float:
    """
    Scale position size inversely to volatility.
    Low ATR (calm) → larger positions; High ATR (chaos) → smaller positions.

    Returns a multiplier (0.3 to 1.5):
      ATR% < 0.8%  → 1.5x (very calm, size up)
      ATR% 0.8-1.5% → 1.0x (normal)
      ATR% 1.5-3.0% → 0.7x (elevated volatility)
      ATR% 3.0-5.0% → 0.4x (high volatility)
      ATR% > 5.0%   → 0.3x (extreme — minimize exposure)
    """
    if atr_pct < 0.8:
        return 1.5
    elif atr_pct < 1.5:
        return 1.0
    elif atr_pct < 3.0:
        return 0.7
    elif atr_pct < 5.0:
        return 0.4
    else:
        return 0.3


# ── Stop Loss / Take Profit / Trailing Stop ──────────────────────────────────

def calculate_stops(price: float, atr: float, action: str,
                    sl_multiplier: float = 1.5,
                    tp_multiplier: float = 2.5) -> dict:
    """
    Calculate ATR-based stop-loss and take-profit levels.

    For BUY:  stop below entry, target above
    For SELL: stop above entry, target below

    Default: 1.5 ATR stop-loss, 2.5 ATR take-profit → 1.67 risk/reward ratio
    """
    if action == "buy":
        stop_loss = price - (atr * sl_multiplier)
        take_profit = price + (atr * tp_multiplier)
        trailing_distance = atr * 1.0
    elif action == "sell":
        stop_loss = price + (atr * sl_multiplier)
        take_profit = price - (atr * tp_multiplier)
        trailing_distance = atr * 1.0
    else:
        return {
            "stop_loss": 0, "take_profit": 0,
            "trailing_distance": 0, "risk_reward": 0,
        }

    risk = abs(price - stop_loss)
    reward = abs(take_profit - price)
    rr = reward / risk if risk > 0 else 0

    return {
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "trailing_distance": round(trailing_distance, 2),
        "risk_reward": round(rr, 2),
    }


# ── Confidence Scaling ───────────────────────────────────────────────────────

def confidence_multiplier(confidence: float) -> float:
    """
    Scale position by Claude's confidence level.
    Low confidence → smaller trade; High confidence → full size.

    0.5 confidence → 0.5x
    0.7 confidence → 0.85x
    0.9 confidence → 1.0x
    """
    return min(1.0, max(0.3, confidence * 1.2))


# ── Main Risk Assessment ─────────────────────────────────────────────────────

def assess_trade(action: str, confidence: float, snapshot: dict,
                 portfolio: dict, stats: dict | None = None,
                 use_rl: bool = True,
                 consecutive_losses: int | None = None) -> TradeRisk:
    """
    Full risk assessment for a proposed trade.
    Returns recommended size, stop/take-profit levels, and rationale.

    Backtests must pass `stats` (neutral defaults), `use_rl=False` and
    `consecutive_losses=0`: the defaults read the LIVE database and RL
    state, which would leak live-account history into simulations.
    """
    price = snapshot["price"]
    atr = snapshot.get("atr", price * 0.015)  # fallback 1.5% if missing
    atr_pct = snapshot.get("atr_pct", 1.5)
    total_usd = portfolio["usdt"] + portfolio["btc"] * price

    if action == "hold":
        return TradeRisk(
            recommended_usd=0, stop_loss_price=0, take_profit_price=0,
            trailing_stop_distance=0, risk_reward_ratio=0,
            kelly_fraction=0, atr_multiplier=1.0,
            position_rationale="Hold — no trade",
        )

    # 1. Get trade statistics for Kelly
    stats = stats or _get_trade_stats()
    kf = kelly_fraction(
        stats["win_rate"], stats["avg_win_pct"], stats["avg_loss_pct"],
        total_trades=stats["total_trades"],
    )

    # 2. ATR volatility scaling
    atr_scale = atr_scale_factor(atr_pct)

    # 3. Confidence scaling
    conf_scale = confidence_multiplier(confidence)

    # 4. Calculate base position from Kelly
    kelly_usd = total_usd * kf if kf > 0 else config.BASE_TRADE_USD

    # 5. Apply all scaling factors
    sized_usd = kelly_usd * atr_scale * conf_scale

    # 6. RL position management adjustment
    rl_scale = 1.0
    rl_action = "hold"
    try:
        if not use_rl:
            raise LookupError("RL disabled (backtest)")
        rl_rec = rl_position.get_recommendation(snapshot, portfolio)
        rl_action = rl_rec.action
        if rl_action == "increase" and action == "buy":
            rl_scale = 1.2
        elif rl_action == "decrease" and action == "buy":
            rl_scale = 0.6
        elif rl_action == "close" and action == "buy":
            rl_scale = 0.3
        elif rl_action == "tighten_stop":
            rl_scale = 0.9
    except Exception:
        pass

    sized_usd *= rl_scale

    # 7. Clamp to config limits
    final_usd = max(config.MIN_TRADE_USD, min(config.MAX_TRADE_USD, sized_usd))

    # 8. Regime-aware stops
    consec_losses = consecutive_losses if consecutive_losses is not None else _get_consecutive_losses()
    sl_mult, tp_mult = _atr_stop_multipliers(atr_pct, consec_losses)
    stops = calculate_stops(price, atr, action,
                            sl_multiplier=sl_mult, tp_multiplier=tp_mult)

    rationale_parts = [
        f"Kelly={kf:.1%} (WR={stats['win_rate']:.0%}→{max(0,stats['win_rate']-0.10):.0%} disc, {stats['total_trades']} trades)",
        f"ATR={atr_pct:.1f}%→{atr_scale:.1f}x (stops {sl_mult}×/{tp_mult}×)",
        f"Conf={confidence:.0%}→{conf_scale:.1f}x",
        f"RL={rl_action}→{rl_scale:.1f}x",
        f"Base=${kelly_usd:.1f}→Final=${final_usd:.1f}",
        f"R:R={stops['risk_reward']:.1f}",
    ]

    logger.info(
        "Risk: %s $%.2f | %s",
        action.upper(), final_usd, " | ".join(rationale_parts),
    )

    return TradeRisk(
        recommended_usd=round(final_usd, 2),
        stop_loss_price=stops["stop_loss"],
        take_profit_price=stops["take_profit"],
        trailing_stop_distance=stops["trailing_distance"],
        risk_reward_ratio=stops["risk_reward"],
        kelly_fraction=round(kf, 4),
        atr_multiplier=round(atr_scale, 2),
        position_rationale=" | ".join(rationale_parts),
    )
