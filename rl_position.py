"""
Reinforcement Learning position management.

A tabular Q-learning agent that learns optimal position sizing adjustments
and exit timing from actual trade outcomes. Complements Kelly Criterion
with experience-driven behavior.

State space (discretized):
  - RSI bucket:       oversold / neutral / overbought (3 states)
  - Trend:            bearish / neutral / bullish (3 states)
  - Volatility:       low / medium / high (3 states)
  - Position status:  no_position / small / medium / large (4 states)
  - PnL status:       losing / flat / winning (3 states)
  Total: 3×3×3×4×3 = 324 states

Action space:
  - 0: hold (do nothing)
  - 1: increase position (buy more)
  - 2: decrease position (partial sell)
  - 3: close position (full exit)
  - 4: tighten stop-loss

Reward:
  - Realized PnL from trade (scaled)
  - Penalty for excessive trading (to discourage churn)
  - Bonus for holding through profitable moves

Persistence: Q-table saved to disk as JSON, loaded on startup.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

Q_TABLE_PATH = Path("ml_models/rl_q_table.json")
N_ACTIONS = 5
ACTIONS = ["hold", "increase", "decrease", "close", "tighten_stop"]

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.15        # exploration rate
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
TRADE_PENALTY = -0.2  # small penalty per trade to discourage churn


@dataclass
class RLState:
    """Discretized market + position state."""
    rsi_bucket: int       # 0=oversold, 1=neutral, 2=overbought
    trend_bucket: int     # 0=bearish, 1=neutral, 2=bullish
    vol_bucket: int       # 0=low, 1=medium, 2=high
    position_bucket: int  # 0=none, 1=small, 2=medium, 3=large
    pnl_bucket: int       # 0=losing, 1=flat, 2=winning

    def to_key(self) -> str:
        return f"{self.rsi_bucket}_{self.trend_bucket}_{self.vol_bucket}_{self.position_bucket}_{self.pnl_bucket}"


@dataclass
class RLRecommendation:
    """RL agent's recommendation for position management."""
    action: str           # from ACTIONS
    confidence: float     # how confident (based on Q-value spread)
    rationale: str
    q_values: list[float]


# ── Q-table management ────────────────────────────────────────────────────

_q_table: dict[str, list[float]] = {}
_epsilon: float = EPSILON
_last_state: RLState | None = None
_last_action: int | None = None


def _load_q_table():
    """Load Q-table from disk."""
    global _q_table, _epsilon
    if Q_TABLE_PATH.exists():
        try:
            with open(Q_TABLE_PATH) as f:
                data = json.load(f)
            _q_table = data.get("q_table", {})
            _epsilon = data.get("epsilon", EPSILON)
            logger.info("RL Q-table loaded: %d states, epsilon=%.3f", len(_q_table), _epsilon)
        except Exception as exc:
            logger.warning("Failed to load Q-table: %s", exc)
            _q_table = {}


def _save_q_table():
    """Persist Q-table to disk."""
    try:
        Q_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(Q_TABLE_PATH, "w") as f:
            json.dump({"q_table": _q_table, "epsilon": _epsilon}, f)
    except Exception as exc:
        logger.warning("Failed to save Q-table: %s", exc)


def _get_q_values(state_key: str) -> list[float]:
    """Get Q-values for a state, initializing if needed."""
    if state_key not in _q_table:
        _q_table[state_key] = [0.0] * N_ACTIONS
    return _q_table[state_key]


# ── State discretization ───────────────────────────────────────────────────

def discretize_state(snapshot: dict, portfolio: dict, entry_price: float | None = None) -> RLState:
    """Convert continuous market data into discrete RL state."""
    rsi = snapshot.get("rsi", 50)
    if rsi < 30:
        rsi_bucket = 0  # oversold
    elif rsi > 70:
        rsi_bucket = 2  # overbought
    else:
        rsi_bucket = 1  # neutral

    macd_trend = snapshot.get("macd_trend", "neutral")
    ichimoku = snapshot.get("ichimoku_signal", "")
    if macd_trend == "bullish" and "bullish" in ichimoku:
        trend_bucket = 2
    elif macd_trend == "bearish" and "bearish" in ichimoku:
        trend_bucket = 0
    else:
        trend_bucket = 1

    atr_pct = snapshot.get("atr_pct", 1.5)
    if atr_pct < 1.0:
        vol_bucket = 0
    elif atr_pct < 2.5:
        vol_bucket = 1
    else:
        vol_bucket = 2

    # Position size relative to portfolio
    price = snapshot.get("price", 1)
    btc_val = portfolio.get("btc", 0) * price
    total = portfolio.get("usdt", 0) + btc_val
    alloc = btc_val / total if total > 0 else 0

    if alloc < 0.05:
        position_bucket = 0
    elif alloc < 0.25:
        position_bucket = 1
    elif alloc < 0.50:
        position_bucket = 2
    else:
        position_bucket = 3

    # PnL since entry
    if entry_price and entry_price > 0 and position_bucket > 0:
        pnl_pct = (price / entry_price - 1) * 100
        if pnl_pct < -1:
            pnl_bucket = 0
        elif pnl_pct > 1:
            pnl_bucket = 2
        else:
            pnl_bucket = 1
    else:
        pnl_bucket = 1  # flat / no position

    return RLState(rsi_bucket, trend_bucket, vol_bucket, position_bucket, pnl_bucket)


# ── Action selection ───────────────────────────────────────────────────────

def select_action(state: RLState) -> tuple[int, list[float]]:
    """Epsilon-greedy action selection."""
    state_key = state.to_key()
    q_values = _get_q_values(state_key)

    if random.random() < _epsilon:
        action = random.randint(0, N_ACTIONS - 1)
    else:
        action = q_values.index(max(q_values))

    return action, q_values


def get_recommendation(snapshot: dict, portfolio: dict,
                       entry_price: float | None = None) -> RLRecommendation:
    """
    Get RL agent's position management recommendation.
    Called by the executor/risk manager to adjust position sizing.
    """
    global _last_state, _last_action

    if not _q_table:
        _load_q_table()

    state = discretize_state(snapshot, portfolio, entry_price)
    action_idx, q_values = select_action(state)

    _last_state = state
    _last_action = action_idx

    action_name = ACTIONS[action_idx]
    q_spread = max(q_values) - min(q_values)
    confidence = min(1.0, q_spread / 2.0) if q_spread > 0 else 0.1

    rationale_parts = [
        f"State: RSI={'OS' if state.rsi_bucket == 0 else 'OB' if state.rsi_bucket == 2 else 'N'}",
        f"Trend={'bull' if state.trend_bucket == 2 else 'bear' if state.trend_bucket == 0 else 'flat'}",
        f"Vol={'H' if state.vol_bucket == 2 else 'L' if state.vol_bucket == 0 else 'M'}",
        f"Pos={'none' if state.position_bucket == 0 else state.position_bucket}",
        f"PnL={'win' if state.pnl_bucket == 2 else 'loss' if state.pnl_bucket == 0 else 'flat'}",
    ]

    return RLRecommendation(
        action=action_name,
        confidence=confidence,
        rationale=f"RL({', '.join(rationale_parts)}) → {action_name} (Q={q_values[action_idx]:.2f})",
        q_values=q_values,
    )


# ── Learning (reward feedback) ─────────────────────────────────────────────

def learn(reward: float, new_snapshot: dict, portfolio: dict,
          entry_price: float | None = None):
    """
    Update Q-table based on observed reward.
    Called after a trade outcome is known.
    """
    global _epsilon, _last_state, _last_action

    if _last_state is None or _last_action is None:
        return

    old_key = _last_state.to_key()
    old_q = _get_q_values(old_key)

    new_state = discretize_state(new_snapshot, portfolio, entry_price)
    new_key = new_state.to_key()
    new_q = _get_q_values(new_key)

    # Q-learning update
    best_next = max(new_q)
    old_q[_last_action] += LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * best_next - old_q[_last_action]
    )

    _q_table[old_key] = old_q

    # Decay exploration
    _epsilon = max(EPSILON_MIN, _epsilon * EPSILON_DECAY)

    _save_q_table()

    logger.debug(
        "RL update: state=%s action=%s reward=%.2f → Q=%.3f (ε=%.3f)",
        old_key, ACTIONS[_last_action], reward, old_q[_last_action], _epsilon,
    )

    _last_state = None
    _last_action = None


def compute_reward(action: str, pnl_pct: float, holding_periods: int = 1) -> float:
    """
    Compute reward for a completed trade.
    action: the RL action that was taken
    pnl_pct: realized PnL percentage
    holding_periods: how many cycles the position was held
    """
    # Base reward = PnL (scaled to [-1, 1] range)
    reward = max(-1.0, min(1.0, pnl_pct / 3.0))

    # Penalty for trading (discourage churn)
    if action in ("increase", "decrease", "close"):
        reward += TRADE_PENALTY

    # Bonus for holding through profitable moves
    if action == "hold" and pnl_pct > 0.5:
        reward += 0.3

    # Penalty for holding losers too long
    if action == "hold" and pnl_pct < -2.0:
        reward -= 0.5

    return reward


def get_rl_context() -> str:
    """Format RL state for Claude's prompt."""
    if not _q_table:
        return ""

    states_explored = len(_q_table)
    if states_explored < 10:
        return ""

    return (
        f"RL AGENT (Q-learning, {states_explored} states explored, ε={_epsilon:.2f}):\n"
        f"  The RL position manager is active and learning from trade outcomes.\n"
        f"  It adjusts position sizing and exit timing based on experience."
    )
