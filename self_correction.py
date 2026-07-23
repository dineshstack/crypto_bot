"""
Self-correction module.

At the start of each cycle, evaluates the outcome of the previous actionable
trade (buy/sell) and — if it was wrong — asks Claude to generate a one-sentence
lesson that gets stored and injected into future prompts.

Outcome thresholds (applied config.OUTCOME_HORIZON_HOURS after the trade):
  buy  + price now  -2 %+ → "wrong"         (bought into a drop)
  buy  + price now  +2 %+ → "correct"
  sell + price now  +2 %+ → "wrong"         (sold before a rally)
  sell + price now  -2 %+ → "correct"
  hold + |change|   >3 %  → "missed_opportunity"
  otherwise               → "neutral"

These correct/wrong/neutral LABELS drive the lesson loop + RL. Continuous
per-trade P&L (for win rate / Kelly / analytics) is computed separately in
trade_pnl.py from the entry and horizon prices stored here.
"""
from __future__ import annotations

import json
import logging
import anthropic
import config
import database as db

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

_data_ex = None  # public production Binance — evaluation prices, never sandbox


def _market_price(symbol: str, at_ts_ms: int | None = None) -> float:
    """
    Price of `symbol` at a moment in time (1h candle close), or last price
    when at_ts_ms is None/now-ish. Evaluating a weeks-old backlog decision
    against TODAY's price would be meaningless — outcomes are defined 4h
    after the decision, so we fetch the candle at that hour.
    """
    global _data_ex
    if _data_ex is None:
        import ccxt
        _data_ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    if at_ts_ms is not None:
        hour_ms = at_ts_ms - (at_ts_ms % 3_600_000)
        candles = _data_ex.fetch_ohlcv(symbol, "1h", since=hour_ms, limit=1)
        if candles:
            return float(candles[0][4])
    return float(_data_ex.fetch_ticker(symbol)["last"])

WRONG_THRESHOLD_PCT  = 2.0
MISSED_THRESHOLD_PCT = 3.0


def _outcome(action: str, trade_price: float, now_price: float) -> tuple[str, float]:
    pct = (now_price - trade_price) / trade_price * 100
    if action == "buy":
        if pct < -WRONG_THRESHOLD_PCT:  return "wrong",              pct
        if pct >  WRONG_THRESHOLD_PCT:  return "correct",            pct
    elif action == "sell":
        if pct >  WRONG_THRESHOLD_PCT:  return "wrong",              pct
        if pct < -WRONG_THRESHOLD_PCT:  return "correct",            pct
    elif action == "hold":
        if abs(pct) > MISSED_THRESHOLD_PCT: return "missed_opportunity", pct
    return "neutral", pct


def _generate_lesson(trade: dict, pct: float, result: str, symbol: str) -> str:
    # MySQL JSON columns arrive as strings — parse before .get()
    m = trade.get("market") or {}
    if isinstance(m, str):
        m = json.loads(m or "{}")
    d = trade.get("decision") or {}
    if isinstance(d, str):
        d = json.loads(d or "{}")
    horizon = config.OUTCOME_HORIZON_HOURS
    if result == "missed_opportunity":
        headline = (
            f"You chose to HOLD, but {symbol} moved {pct:+.1f}% over the next {horizon} h — "
            f"a missed opportunity."
        )
    else:
        headline = f"You made a wrong {trade['action']} trade."
    prompt = (
        f"{headline} Write ONE lesson (max 20 words) "
        f"starting with 'Avoid', 'Do not', or 'Only'.\n\n"
        f"Decision: {trade['action'].upper()} ${float(trade.get('amount_usd') or 0):.0f} "
        f"at {symbol} ${float(trade['price']):,}\n"
        f"RSI {m.get('rsi','?')} | F&G {m.get('fear_greed','?')}/100 | "
        f"vs SMA20 {m.get('vs_sma20_pct','?')}%\n"
        f"Your reasoning: \"{d.get('reason','')}\"\n"
        f"Price moved {pct:+.1f}% over the next {horizon} h.\n\n"
        f"Lesson:"
    )
    r = _client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip().strip('"').strip("'")


def evaluate_and_learn(current_price: float | None = None,
                       max_trades: int = 10) -> list[str]:
    """
    Evaluate up to max_trades unevaluated decisions (buys, sells AND holds)
    that are past the outcome window (config.OUTCOME_HORIZON_HOURS).
    Symbol-aware: each decision is scored against its own symbol's price at
    the horizon, so ETH holds are never compared to the BTC price and
    week-old backlog rows are scored at their historical window, not today.
    Returns the list of newly generated lessons (wrong trades and missed
    opportunities both teach).
    """
    trades = db.get_unevaluated_trades(limit=max_trades)
    lessons: list[str] = []

    for trade in trades:
        market = trade.get("market") or {}
        if isinstance(market, str):
            market = json.loads(market or "{}")
        symbol = market.get("symbol") or "BTC/USDT"

        try:
            eval_ts_ms = (int(trade["created_at"].timestamp() * 1000)
                          + config.OUTCOME_HORIZON_HOURS * 3_600_000)
            now_price = _market_price(symbol, at_ts_ms=eval_ts_ms)
        except Exception as exc:
            logger.warning("Outcome eval: no price for %s (trade #%s): %s",
                           symbol, trade["id"], exc)
            continue

        result, pct = _outcome(trade["action"], float(trade["price"]), now_price)
        db.update_trade_outcome(trade["id"], result, now_price)
        logger.info("Trade #%s (%s %s) outcome: %s (%+.1f%%)",
                    trade["id"], symbol, trade["action"], result, pct)

        if result in ("wrong", "missed_opportunity"):
            try:
                lesson = _generate_lesson(trade, pct, result, symbol)
                db.save_lesson(lesson, "self_correction", trade["id"])
                logger.info("New lesson: %s", lesson)
                lessons.append(lesson)
            except Exception as exc:
                logger.warning("Lesson generation failed (trade #%s): %s",
                               trade["id"], exc)

    return lessons
