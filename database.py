"""
Supabase-backed data layer.
Replaces portfolio.py — all trade logging, snapshots, lessons, and reviews live here.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from supabase import create_client, Client
import config

logger = logging.getLogger(__name__)

_client: Client | None = None


def _db() -> Client:
    global _client
    if _client is None:
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _client


# ── Bootstrap ───────────────────────────────────────────────────────────────

def init():
    """Verify Supabase connection is reachable."""
    try:
        _db().table("portfolio_snapshots").select("id").limit(1).execute()
        logger.info("Supabase connected OK")
    except Exception as exc:
        logger.error("Supabase connection failed: %s", exc)
        raise


# ── Trades ───────────────────────────────────────────────────────────────────

def log_trade(result: dict, decision: dict, snapshot: dict) -> str:
    """Insert a trade record; return its UUID."""
    row = {
        "action":     result["action"],
        "amount_usd": result.get("amount_usd", 0),
        "btc_qty":    result.get("btc_amount", 0),
        "price":      snapshot["price"],
        "decision":   decision,
        "market":     snapshot,
        "success":    result["success"],
        "error":      result.get("error"),
    }
    r = _db().table("trades").insert(row).execute()
    return r.data[0]["id"]


def get_recent_trades(n: int = 5) -> list[dict]:
    r = (
        _db().table("trades")
        .select("created_at,action,amount_usd,price,success,error,outcome,decision")
        .order("created_at", desc=True)
        .limit(n)
        .execute()
    )
    return r.data or []


def get_last_unevaluated_trade() -> dict | None:
    """Most recent successful buy/sell that hasn't been outcome-evaluated yet."""
    r = (
        _db().table("trades")
        .select("*")
        .neq("action", "hold")
        .is_("outcome", "null")
        .eq("success", True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return r.data[0] if r.data else None


def update_trade_outcome(trade_id: str, outcome: str, price_after: float):
    _db().table("trades").update({
        "outcome":          outcome,
        "price_after_4h":   price_after,
        "lesson_generated": outcome == "wrong",
    }).eq("id", trade_id).execute()


def get_trades_in_period(start: datetime, end: datetime) -> list[dict]:
    r = (
        _db().table("trades")
        .select("*")
        .gte("created_at", start.isoformat())
        .lte("created_at", end.isoformat())
        .order("created_at")
        .execute()
    )
    return r.data or []


# ── Portfolio snapshots ───────────────────────────────────────────────────────

def save_snapshot(usdt: float, btc: float, price: float) -> float:
    """Save portfolio snapshot, return total USD value."""
    total = usdt + btc * price
    _db().table("portfolio_snapshots").insert({
        "usdt": usdt, "btc": btc, "price": price, "total_usd": total,
    }).execute()
    return total


def get_first_snapshot_total() -> float | None:
    r = (
        _db().table("portfolio_snapshots")
        .select("total_usd")
        .order("created_at")
        .limit(1)
        .execute()
    )
    return r.data[0]["total_usd"] if r.data else None


# ── Lessons ───────────────────────────────────────────────────────────────────

def save_lesson(text: str, source: str, trade_id: str | None = None) -> str:
    r = _db().table("lessons").insert({
        "lesson":   text,
        "source":   source,
        "trade_id": trade_id,
        "active":   True,
    }).execute()
    return r.data[0]["id"]


def get_active_lessons(limit: int = 5) -> list[str]:
    r = (
        _db().table("lessons")
        .select("lesson")
        .eq("active", True)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [row["lesson"] for row in (r.data or [])]


# ── Weekly reviews ────────────────────────────────────────────────────────────

def save_weekly_review(
    period_start: datetime,
    period_end:   datetime,
    total:        int,
    correct:      int,
    wrong:        int,
    pnl:          float,
    review_text:  str,
):
    _db().table("weekly_reviews").insert({
        "period_start":   period_start.isoformat(),
        "period_end":     period_end.isoformat(),
        "total_trades":   total,
        "correct_trades": correct,
        "wrong_trades":   wrong,
        "pnl_usd":        pnl,
        "review_text":    review_text,
    }).execute()


def get_last_weekly_review_date() -> datetime | None:
    r = (
        _db().table("weekly_reviews")
        .select("created_at")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if r.data:
        ts = r.data[0]["created_at"]
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return None


# ── Pending confirmations ─────────────────────────────────────────────────────

def save_pending_confirmation(
    conf_id:   str,
    decision:  dict,
    market:    dict,
    portfolio: dict,
):
    _db().table("pending_confirmations").insert({
        "id":        conf_id,
        "decision":  decision,
        "market":    market,
        "portfolio": portfolio,
        "status":    "pending",
    }).execute()


def update_confirmation_status(conf_id: str, status: str):
    _db().table("pending_confirmations").update(
        {"status": status}
    ).eq("id", conf_id).execute()


# ── Bot events (activity log for web dashboard) ──────────────────────────────

def log_event(event: str, message: str, level: str = "info", data: dict = None):
    """Log a structured event for the web dashboard activity feed."""
    try:
        _db().table("bot_events").insert({
            "event":   event,
            "message": message,
            "level":   level,
            "data":    data or {},
        }).execute()
    except Exception as exc:
        logger.debug("Event log failed: %s", exc)


def get_events(limit: int = 50, level: str = None) -> list[dict]:
    q = _db().table("bot_events").select("*").order("created_at", desc=True)
    if level:
        q = q.eq("level", level)
    return (q.limit(limit).execute()).data or []
