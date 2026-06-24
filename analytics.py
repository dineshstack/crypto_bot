"""
Performance analytics engine — tracks and reports trading metrics.

Computes from Supabase trade history and portfolio snapshots:
  - PnL (realized + unrealized)
  - Win rate, profit factor, expectancy
  - Sharpe ratio, Sortino ratio
  - Max drawdown (peak-to-trough)
  - Streak tracking (consecutive wins/losses)
  - Per-asset breakdown (BTC vs ETH)
  - Rolling performance (7d, 30d, all-time)

Exposed via Telegram commands and injected into weekly review context.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import database as db

logger = logging.getLogger(__name__)


def _get_all_trades(days: int | None = None) -> list[dict]:
    """Fetch trades, optionally limited to last N days."""
    if days:
        start = datetime.now(timezone.utc) - timedelta(days=days)
        end = datetime.now(timezone.utc)
        return db.get_trades_in_period(start, end)
    return db.get_recent_trades(500)


def _get_snapshots(days: int | None = None) -> list[dict]:
    """Fetch portfolio snapshots for equity curve."""
    try:
        since = None
        if days:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        return db.get_snapshots(limit=2000, since=since)
    except Exception:
        return []


def compute_metrics_for_period(start: datetime, end: datetime) -> dict:
    """
    Compute metrics for an explicit time window [start, end].
    Used for previous-period delta comparisons (e.g. 'last 7 days vs prior 7 days').
    """
    trades = db.get_trades_in_period(start, end)
    since_str = start.strftime("%Y-%m-%d %H:%M:%S")

    # Fetch snapshots in range
    try:
        rows = db._execute(
            """SELECT created_at, total_usd FROM portfolio_snapshots
               WHERE created_at >= %s AND created_at <= %s
               ORDER BY created_at""",
            (since_str, end.strftime("%Y-%m-%d %H:%M:%S")),
            fetch="all",
        )
        snapshots = [{"created_at": str(r["created_at"]), "total_usd": r["total_usd"]} for r in rows]
    except Exception:
        snapshots = []

    actionable = [t for t in trades if t["action"] in ("buy", "sell") and t["success"]]
    wins = [t for t in actionable if t.get("outcome") == "correct"]
    losses = [t for t in actionable if t.get("outcome") == "wrong"]
    evaluated = len(wins) + len(losses)
    win_rate = len(wins) / evaluated if evaluated > 0 else 0

    pnl = 0.0
    pnl_pct = 0.0
    if len(snapshots) >= 2:
        first = float(snapshots[0]["total_usd"])
        last = float(snapshots[-1]["total_usd"])
        pnl = last - first
        pnl_pct = (last / first - 1) * 100 if first > 0 else 0

    max_dd_pct = 0.0
    peak = 0.0
    for s in snapshots:
        val = float(s["total_usd"])
        peak = max(peak, val)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - val) / peak * 100)

    daily_returns = _compute_daily_returns(snapshots)
    sharpe = _sharpe_ratio(daily_returns)

    return {
        "pnl_usd": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "win_rate": round(win_rate, 3),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": len(actionable),
    }


def compute_metrics(days: int | None = None) -> dict:
    """
    Compute comprehensive performance metrics.
    days=None → all-time, days=7 → last week, days=30 → last month.
    """
    trades = _get_all_trades(days)
    snapshots = _get_snapshots(days)

    actionable = [t for t in trades if t["action"] in ("buy", "sell") and t["success"]]
    wins = [t for t in actionable if t.get("outcome") == "correct"]
    losses = [t for t in actionable if t.get("outcome") == "wrong"]
    evaluated = len(wins) + len(losses)

    # Win rate
    win_rate = len(wins) / evaluated if evaluated > 0 else 0

    # Average trade amounts
    win_amounts = [t.get("amount_usd", 0) for t in wins]
    loss_amounts = [t.get("amount_usd", 0) for t in losses]
    avg_win = sum(win_amounts) / len(win_amounts) if win_amounts else 0
    avg_loss = sum(loss_amounts) / len(loss_amounts) if loss_amounts else 0

    # Profit factor
    total_wins = sum(win_amounts)
    total_losses = sum(loss_amounts)
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf") if total_wins > 0 else 0

    # Expectancy (expected $ per trade)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if evaluated > 0 else 0

    # PnL from snapshots
    pnl = 0.0
    pnl_pct = 0.0
    if len(snapshots) >= 2:
        first = float(snapshots[0]["total_usd"])
        last = float(snapshots[-1]["total_usd"])
        pnl = last - first
        pnl_pct = (last / first - 1) * 100 if first > 0 else 0

    # Max drawdown from equity curve
    max_dd = 0.0
    max_dd_pct = 0.0
    peak = 0.0
    for s in snapshots:
        val = float(s["total_usd"])
        if val > peak:
            peak = val
        if peak > 0:
            dd_pct = (peak - val) / peak * 100
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd = peak - val

    # Sharpe & Sortino from daily returns
    daily_returns = _compute_daily_returns(snapshots)
    sharpe = _sharpe_ratio(daily_returns)
    sortino = _sortino_ratio(daily_returns)

    # Streaks
    current_streak, max_win_streak, max_loss_streak = _compute_streaks(actionable)

    # Per-asset breakdown
    per_asset = _per_asset_breakdown(actionable)

    # Trade frequency
    if days:
        total_days = days
    elif snapshots:
        try:
            ts = str(snapshots[0]["created_at"]).replace("Z", "+00:00")
            first_dt = datetime.fromisoformat(ts) if "+" in ts or "T" in ts else datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            total_days = max(1, (datetime.now(timezone.utc) - first_dt).days)
        except Exception:
            total_days = 1
    else:
        total_days = 1
    trades_per_day = len(actionable) / total_days

    # BTC buy-and-hold benchmark: % change in BTC price over the same period
    btc_benchmark_pct = None
    if len(snapshots) >= 2:
        try:
            first_price = float(snapshots[0].get("price") or 0)
            last_price = float(snapshots[-1].get("price") or 0)
            if first_price > 0 and last_price > 0:
                btc_benchmark_pct = round((last_price / first_price - 1) * 100, 2)
        except Exception:
            pass

    return {
        "period_days": days or total_days,
        "total_trades": len(actionable),
        "holds": len([t for t in trades if t["action"] == "hold"]),
        "evaluated": evaluated,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 3),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "expectancy_usd": round(expectancy, 2),
        "pnl_usd": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "max_drawdown_usd": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "current_streak": current_streak,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "trades_per_day": round(trades_per_day, 1),
        "per_asset": per_asset,
        "btc_benchmark_pct": btc_benchmark_pct,
    }


def _compute_daily_returns(snapshots: list[dict]) -> list[float]:
    """Compute daily returns from portfolio snapshots."""
    if len(snapshots) < 2:
        return []

    by_day: dict[str, float] = {}
    for s in snapshots:
        day = str(s["created_at"])[:10]
        by_day[day] = float(s["total_usd"])

    days = sorted(by_day.keys())
    returns = []
    for i in range(1, len(days)):
        prev = by_day[days[i - 1]]
        curr = by_day[days[i]]
        if prev > 0:
            returns.append(float(curr / prev - 1))
    return returns


def _sharpe_ratio(returns: list[float], risk_free_daily: float = 0.0001) -> float:
    """Annualized Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free_daily for r in returns]
    mean = sum(excess) / len(excess)
    variance = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
    std = math.sqrt(variance) if variance > 0 else 0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(365)


def _sortino_ratio(returns: list[float], risk_free_daily: float = 0.0001) -> float:
    """Annualized Sortino ratio (only penalizes downside volatility)."""
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free_daily for r in returns]
    mean = sum(excess) / len(excess)
    downside = [r for r in excess if r < 0]
    if not downside:
        return 0.0 if mean <= 0 else 99.0
    down_var = sum(r ** 2 for r in downside) / len(downside)
    down_std = math.sqrt(down_var)
    if down_std == 0:
        return 0.0
    return (mean / down_std) * math.sqrt(365)


def _compute_streaks(trades: list[dict]) -> tuple[str, int, int]:
    """Compute current streak and max win/loss streaks."""
    if not trades:
        return "none", 0, 0

    current_type = None
    current_len = 0
    max_win = 0
    max_loss = 0

    for t in reversed(trades):  # oldest first
        outcome = t.get("outcome", "")
        if outcome == "correct":
            if current_type == "win":
                current_len += 1
            else:
                current_type = "win"
                current_len = 1
            max_win = max(max_win, current_len)
        elif outcome == "wrong":
            if current_type == "loss":
                current_len += 1
            else:
                current_type = "loss"
                current_len = 1
            max_loss = max(max_loss, current_len)

    streak_str = f"{current_len} {'win' if current_type == 'win' else 'loss'}" if current_type else "none"
    return streak_str, max_win, max_loss


def _per_asset_breakdown(trades: list[dict]) -> dict:
    """Group trade stats by asset (BTC, ETH)."""
    by_asset = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "volume_usd": 0})

    for t in trades:
        market = t.get("market") or {}
        symbol = market.get("symbol", "BTC/USDT")
        base = symbol.split("/")[0] if "/" in symbol else "BTC"

        by_asset[base]["trades"] += 1
        by_asset[base]["volume_usd"] += t.get("amount_usd", 0)
        if t.get("outcome") == "correct":
            by_asset[base]["wins"] += 1
        elif t.get("outcome") == "wrong":
            by_asset[base]["losses"] += 1

    result = {}
    for asset, stats in by_asset.items():
        evaluated = stats["wins"] + stats["losses"]
        result[asset] = {
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": round(stats["wins"] / evaluated, 3) if evaluated > 0 else 0,
            "volume_usd": round(stats["volume_usd"], 2),
        }
    return result


# ── Formatted reports ───────────────────────────────────────────────────────

def format_report(days: int | None = None) -> str:
    """Generate a formatted performance report for Telegram."""
    m = compute_metrics(days)

    period = f"Last {m['period_days']}d" if days else "All-Time"

    lines = [
        f"📊 *Performance Report — {period}*",
        "",
        f"💰 PnL: ${m['pnl_usd']:+.2f} ({m['pnl_pct']:+.1f}%)",
        f"📉 Max Drawdown: ${m['max_drawdown_usd']:.2f} ({m['max_drawdown_pct']:.1f}%)",
        "",
        f"🎯 Win Rate: {m['win_rate']:.0%} ({m['wins']}W / {m['losses']}L of {m['evaluated']} evaluated)",
        f"📊 Profit Factor: {m['profit_factor']}",
        f"💵 Expectancy: ${m['expectancy_usd']:+.2f}/trade",
        f"📈 Avg Win: ${m['avg_win_usd']:.2f}  |  Avg Loss: ${m['avg_loss_usd']:.2f}",
        "",
        f"📐 Sharpe Ratio: {m['sharpe_ratio']}",
        f"📐 Sortino Ratio: {m['sortino_ratio']}",
        "",
        f"🔥 Streak: {m['current_streak']}",
        f"   Best: {m['max_win_streak']} wins  |  Worst: {m['max_loss_streak']} losses",
        "",
        f"📋 Total: {m['total_trades']} trades + {m['holds']} holds ({m['trades_per_day']:.1f}/day)",
    ]

    if m["per_asset"]:
        lines.append("")
        lines.append("*Per Asset:*")
        for asset, stats in m["per_asset"].items():
            lines.append(
                f"  {asset}: {stats['trades']} trades, "
                f"{stats['win_rate']:.0%} WR, "
                f"${stats['volume_usd']:.2f} vol"
            )

    return "\n".join(lines)


def format_compact_report() -> str:
    """One-line summary for injection into the main cycle notification."""
    m = compute_metrics(7)
    return (
        f"7d: {m['pnl_pct']:+.1f}% | WR {m['win_rate']:.0%} | "
        f"Sharpe {m['sharpe_ratio']} | DD {m['max_drawdown_pct']:.1f}%"
    )
