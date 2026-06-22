"""
Self-correction module.

At the start of each cycle, evaluates the outcome of the previous actionable
trade (buy/sell) and — if it was wrong — asks Claude to generate a one-sentence
lesson that gets stored and injected into future prompts.

Outcome thresholds (applied 4 h after the trade, i.e. the next cycle):
  buy  + price now  -2 %+ → "wrong"         (bought into a drop)
  buy  + price now  +2 %+ → "correct"
  sell + price now  +2 %+ → "wrong"         (sold before a rally)
  sell + price now  -2 %+ → "correct"
  hold + |change|   >3 %  → "missed_opportunity"
  otherwise               → "neutral"
"""
from __future__ import annotations

import logging
import anthropic
import config
import database as db

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

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


def _generate_lesson(trade: dict, pct: float) -> str:
    m = trade.get("market") or {}
    d = trade.get("decision") or {}
    prompt = (
        f"You made a wrong {trade['action']} trade. Write ONE lesson (max 20 words) "
        f"starting with 'Avoid', 'Do not', or 'Only'.\n\n"
        f"Decision: {trade['action'].upper()} ${trade.get('amount_usd',0):.0f} "
        f"at BTC ${trade['price']:,}\n"
        f"RSI {m.get('rsi','?')} | F&G {m.get('fear_greed','?')}/100 | "
        f"vs SMA20 {m.get('vs_sma20_pct','?')}%\n"
        f"Your reasoning: \"{d.get('reason','')}\"\n"
        f"Price moved {pct:+.1f}% over the next 4 h — WRONG direction.\n\n"
        f"Lesson:"
    )
    r = _client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip().strip('"').strip("'")


def evaluate_and_learn(current_price: float) -> str | None:
    """
    Evaluate the most recent unevaluated trade against the current price.
    Returns a new lesson string if the trade was 'wrong', else None.
    """
    trade = db.get_last_unevaluated_trade()
    if not trade:
        return None

    result, pct = _outcome(trade["action"], float(trade["price"]), current_price)
    db.update_trade_outcome(trade["id"], result, current_price)

    logger.info(
        "Trade %s (%s) outcome: %s (%.1f%%)",
        trade["id"][:8], trade["action"], result, pct,
    )

    if result == "wrong":
        lesson = _generate_lesson(trade, pct)
        db.save_lesson(lesson, "self_correction", trade["id"])
        logger.info("New lesson: %s", lesson)
        return lesson

    return None
