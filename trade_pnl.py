"""
Per-trade decision-quality P&L — one pure function, no side effects.

Lives on its own so both analytics (performance metrics) and risk_manager
(Kelly inputs) can import it without an import cycle.

It marks an actionable decision to market at the outcome horizon:

  buy  profits when price rose   → pnl = +move%
  sell (a defensive 10% trim) 'profits' when price fell → pnl = -move%

...then subtracts a one-way execution cost so the number is NET, not gross.

This measures DECISION QUALITY — did the market move the bot's way by more
than it costs to trade — which is the correct input to win rate and Kelly.
It is deliberately NOT the realized portfolio P&L (that is tracked
separately from portfolio snapshots in analytics). A buy that the bot never
closes still has a decision-quality P&L: was buying at that moment right?
"""
from __future__ import annotations

import config


def decision_pnl_pct(action: str, entry_price, exit_price) -> float | None:
    """
    Net directional P&L (%) of a buy/sell decision, marked to market at the
    outcome horizon. Returns None when it can't be computed yet (no exit
    price recorded, or bad inputs). Holds return None — they have no
    directional P&L.
    """
    if action not in ("buy", "sell"):
        return None
    if exit_price is None:
        return None
    try:
        entry = float(entry_price)
        exit_ = float(exit_price)
    except (TypeError, ValueError):
        return None
    if entry <= 0:
        return None

    move = (exit_ - entry) / entry * 100.0
    directional = move if action == "buy" else -move
    return round(directional - config.DECISION_COST_PCT, 4)
