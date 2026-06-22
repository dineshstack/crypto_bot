"""
Grid/DCA strategy for sideways (range-bound) market regimes.

When the market regime detector flags "sideways", the standard trend-following
approach generates mostly HOLD signals. Grid/DCA exploits the range instead:

  - Grid trading:  place buy orders at support levels, sell at resistance
  - DCA (Dollar-Cost Averaging): accumulate at regular intervals when price
    is in the lower half of the range

Strategy selection:
  - Sideways + low volatility (ATR% < 1.5%) → Grid (profit from range)
  - Sideways + moderate volatility → DCA (accumulate on dips)
  - Non-sideways regime → disabled, returns no orders

Safety:
  - Max grid orders: 3 buys + 3 sells
  - Each order: $2-5 (small)
  - Grid only active when Bollinger Bands width confirms range
  - Auto-cancels all grid orders if regime changes
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import config

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    """A single grid price level."""
    price: float
    side: str         # "buy" or "sell"
    amount_usd: float
    order_id: str | None = None
    filled: bool = False


@dataclass
class GridPlan:
    """Complete grid/DCA execution plan."""
    strategy: str               # "grid" | "dca" | "none"
    levels: list[GridLevel] = field(default_factory=list)
    range_low: float = 0.0
    range_high: float = 0.0
    rationale: str = ""


# ── Grid state tracking ────────────────────────────────────────────────────

_active_grid: GridPlan | None = None
_active_orders: list[str] = []  # exchange order IDs to cancel on regime change


def get_active_grid() -> GridPlan | None:
    return _active_grid


def is_grid_active() -> bool:
    return _active_grid is not None and _active_grid.strategy != "none"


# ── Strategy computation ───────────────────────────────────────────────────

def compute_grid_plan(snapshot: dict, portfolio: dict) -> GridPlan:
    """
    Determine if grid/DCA is appropriate and compute price levels.

    Uses Bollinger Bands to define the range:
      - Range low  = BB lower band
      - Range high = BB upper band
      - Grid spacing = divide range into equal steps
    """
    price = snapshot["price"]
    bb_upper = snapshot.get("bb_upper", 0)
    bb_lower = snapshot.get("bb_lower", 0)
    atr_pct = snapshot.get("atr_pct", 2.0)
    rsi = snapshot.get("rsi", 50)

    # Check if conditions favor grid/DCA
    if bb_upper <= 0 or bb_lower <= 0:
        return GridPlan(strategy="none", rationale="Missing Bollinger Band data")

    bb_width_pct = (bb_upper - bb_lower) / price * 100

    # BB width < 5% = tight range = good for grid
    # BB width > 8% = wide range = trending, not suitable
    if bb_width_pct > 8:
        return GridPlan(strategy="none",
                        rationale=f"BB width {bb_width_pct:.1f}% too wide — trending market")

    usdt_available = portfolio.get("usdt", 0) if isinstance(portfolio, dict) else 0
    if usdt_available < config.MIN_TRADE_USD * 2:
        return GridPlan(strategy="none",
                        rationale=f"Insufficient USDT (${usdt_available:.2f})")

    # Choose strategy based on volatility
    if atr_pct < 1.5 and bb_width_pct < 5:
        return _build_grid_plan(price, bb_lower, bb_upper, atr_pct, usdt_available)
    else:
        return _build_dca_plan(price, bb_lower, bb_upper, rsi, usdt_available)


def _build_grid_plan(price: float, range_low: float, range_high: float,
                     atr_pct: float, usdt: float) -> GridPlan:
    """Build a symmetric grid of buy/sell orders within the range."""
    range_mid = (range_low + range_high) / 2
    grid_spacing = (range_high - range_low) / 6  # 6 steps = 3 buys + 3 sells

    levels = []

    # Buy levels: below current price, toward range low
    order_size = min(3.0, usdt / 4)  # conservative: $3 per level
    for i in range(1, 4):
        buy_price = range_mid - (grid_spacing * i)
        if buy_price >= range_low * 0.99:  # stay within range
            levels.append(GridLevel(
                price=round(buy_price, 2),
                side="buy",
                amount_usd=round(order_size, 2),
            ))

    # Sell levels: above current price, toward range high
    for i in range(1, 4):
        sell_price = range_mid + (grid_spacing * i)
        if sell_price <= range_high * 1.01:
            levels.append(GridLevel(
                price=round(sell_price, 2),
                side="sell",
                amount_usd=round(order_size, 2),
            ))

    buys = [l for l in levels if l.side == "buy"]
    sells = [l for l in levels if l.side == "sell"]

    return GridPlan(
        strategy="grid",
        levels=levels,
        range_low=round(range_low, 2),
        range_high=round(range_high, 2),
        rationale=(
            f"Grid: {len(buys)} buys / {len(sells)} sells in "
            f"${range_low:,.0f}–${range_high:,.0f} range "
            f"(ATR {atr_pct:.1f}%, ${order_size:.2f}/level)"
        ),
    )


def _build_dca_plan(price: float, range_low: float, range_high: float,
                    rsi: float, usdt: float) -> GridPlan:
    """
    Build DCA plan: accumulate on dips in the lower half of the range.
    Only creates buy orders; relies on the main trading loop for sells.
    """
    range_mid = (range_low + range_high) / 2
    levels = []

    # DCA: buy at 3 levels below midpoint
    dca_spacing = (range_mid - range_low) / 3
    order_size = min(2.5, usdt / 4)

    for i in range(1, 4):
        buy_price = range_mid - (dca_spacing * i * 0.9)
        if buy_price >= range_low * 0.99:
            # More $ at lower prices (scale: 0.8x, 1.0x, 1.2x)
            scale = 0.8 + (i * 0.2)
            levels.append(GridLevel(
                price=round(buy_price, 2),
                side="buy",
                amount_usd=round(order_size * scale, 2),
            ))

    return GridPlan(
        strategy="dca",
        levels=levels,
        range_low=round(range_low, 2),
        range_high=round(range_high, 2),
        rationale=(
            f"DCA: {len(levels)} buy levels in "
            f"${range_low:,.0f}–${range_mid:,.0f} (lower half), "
            f"RSI {rsi:.0f}, ${order_size:.2f}/level"
        ),
    )


# ── Grid execution ──────────────────────────────────────────────────────────

def execute_grid(exchange, plan: GridPlan) -> dict:
    """Place grid/DCA limit orders on the exchange."""
    global _active_grid, _active_orders

    if plan.strategy == "none":
        return {"status": "skipped", "reason": plan.rationale}

    placed = 0
    failed = 0
    order_ids = []

    for level in plan.levels:
        try:
            side = level.side
            qty = level.amount_usd / level.price

            min_qty = (exchange.markets.get(config.SYMBOL, {})
                       .get("limits", {}).get("amount", {}).get("min", 0.00001))
            if qty < min_qty:
                continue

            order = exchange.create_limit_order(
                config.SYMBOL,
                side,
                qty,
                level.price,
                params={"timeInForce": "GTC"},
            )
            level.order_id = order.get("id")
            order_ids.append(level.order_id)
            placed += 1
            logger.info(
                "Grid %s: %s %.6f @ $%,.0f ($%.2f)",
                plan.strategy.upper(), side.upper(), qty, level.price, level.amount_usd,
            )
        except Exception as exc:
            failed += 1
            logger.warning("Grid order failed: %s %s @ $%.0f — %s",
                           level.side, level.amount_usd, level.price, exc)

    _active_grid = plan
    _active_orders = order_ids

    return {
        "status": "active",
        "strategy": plan.strategy,
        "placed": placed,
        "failed": failed,
        "levels": len(plan.levels),
        "rationale": plan.rationale,
    }


def cancel_grid(exchange) -> int:
    """Cancel all active grid orders. Returns count of cancelled orders."""
    global _active_grid, _active_orders

    if not _active_orders:
        _active_grid = None
        return 0

    cancelled = 0
    for order_id in _active_orders:
        try:
            exchange.cancel_order(order_id, config.SYMBOL)
            cancelled += 1
        except Exception as exc:
            logger.debug("Grid cancel failed for %s: %s", order_id, exc)

    logger.info("Grid cancelled: %d/%d orders", cancelled, len(_active_orders))
    _active_grid = None
    _active_orders = []
    return cancelled


def get_grid_context() -> str:
    """Format active grid info for Claude's prompt."""
    if not _active_grid or _active_grid.strategy == "none":
        return ""

    g = _active_grid
    buy_levels = [l for l in g.levels if l.side == "buy" and not l.filled]
    sell_levels = [l for l in g.levels if l.side == "sell" and not l.filled]

    lines = [f"ACTIVE {g.strategy.upper()} STRATEGY:"]
    lines.append(f"  Range: ${g.range_low:,.0f} – ${g.range_high:,.0f}")
    if buy_levels:
        prices = ", ".join(f"${l.price:,.0f}" for l in buy_levels)
        lines.append(f"  Buy levels:  {prices}")
    if sell_levels:
        prices = ", ".join(f"${l.price:,.0f}" for l in sell_levels)
        lines.append(f"  Sell levels: {prices}")
    lines.append(f"  Note: {g.rationale}")

    return "\n".join(lines)
