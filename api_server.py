"""
FastAPI REST API — exposes bot data to the Next.js dashboard.

Runs on the VPS alongside the trading bot. The dashboard on Vercel
calls these endpoints to display trades, snapshots, analytics, etc.

Start: uvicorn api_server:app --host 0.0.0.0 --port 8100
Or via systemd: see deploy/install.sh

Authentication: API_SECRET_KEY header required on every request.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import config
import database as db
import analytics

logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Bot API", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _auth(x_api_key: str = Header(None)):
    if not config.API_SECRET_KEY:
        raise HTTPException(500, "API_SECRET_KEY not configured on server")
    if x_api_key != config.API_SECRET_KEY:
        raise HTTPException(401, "Invalid API key")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Snapshots ─────────────────────────────────────────────────────────────────

@app.get("/api/snapshots")
def get_snapshots(
    limit: int = Query(200, le=2000),
    days: int = Query(None),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    since = None
    if days:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    return db.get_snapshots(limit=limit, since=since)


# ── Trades ────────────────────────────────────────────────────────────────────

@app.get("/api/trades")
def get_trades(
    limit: int = Query(100, le=500),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    rows = db.get_recent_trades(limit)
    for r in rows:
        if isinstance(r.get("decision"), str):
            r["decision"] = json.loads(r["decision"])
        if isinstance(r.get("market"), str):
            r["market"] = json.loads(r["market"])
    return rows


@app.get("/api/trades/period")
def get_trades_period(
    days: int = Query(7),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    end = datetime.now(timezone.utc)
    return db.get_trades_in_period(start, end)


# ── Lessons ───────────────────────────────────────────────────────────────────

@app.get("/api/lessons")
def get_lessons(
    limit: int = Query(30),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    rows = db._execute(
        "SELECT * FROM lessons ORDER BY created_at DESC LIMIT %s",
        (limit,),
        fetch="all",
    )
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return rows


# ── Weekly Reviews ────────────────────────────────────────────────────────────

@app.get("/api/reviews")
def get_reviews(
    limit: int = Query(10),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    rows = db._execute(
        "SELECT * FROM weekly_reviews ORDER BY created_at DESC LIMIT %s",
        (limit,),
        fetch="all",
    )
    for r in rows:
        r["created_at"] = str(r["created_at"])
        r["period_start"] = str(r["period_start"])
        r["period_end"] = str(r["period_end"])
    return rows


# ── Bot Events ────────────────────────────────────────────────────────────────

@app.get("/api/events")
def get_events(
    limit: int = Query(100, le=500),
    level: str = Query(None),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    return db.get_events(limit=limit, level=level)


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.get("/api/analytics")
def get_analytics(
    days: int = Query(None),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    return analytics.compute_metrics(days)


@app.get("/api/analytics/compact")
def get_analytics_compact(x_api_key: str = Header(None)):
    _auth(x_api_key)
    return {"summary": analytics.format_compact_report()}


# ── Coin Research ─────────────────────────────────────────────────────────────

@app.get("/api/research")
def get_research(
    limit: int = Query(20),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    rows = db._execute(
        "SELECT * FROM coin_research ORDER BY created_at DESC LIMIT %s",
        (limit,),
        fetch="all",
    )
    for r in rows:
        r["created_at"] = str(r["created_at"])
        if isinstance(r.get("risks"), str):
            r["risks"] = json.loads(r["risks"])
        if isinstance(r.get("opportunities"), str):
            r["opportunities"] = json.loads(r["opportunities"])
    return rows


@app.get("/api/watchlist")
def get_watchlist(x_api_key: str = Header(None)):
    _auth(x_api_key)
    return db.get_watchlist()


# ── Dashboard summary (single call for homepage) ─────────────────────────────

@app.get("/api/dashboard")
def get_dashboard(x_api_key: str = Header(None)):
    """All data needed for the dashboard homepage in one request."""
    _auth(x_api_key)

    snapshots = db.get_snapshots(limit=200)
    trades = db.get_recent_trades(100)
    lessons = db.get_active_lessons(5)
    metrics = analytics.compute_metrics(7)

    latest = snapshots[-1] if snapshots else None
    actionable = [t for t in trades if t["action"] != "hold" and t.get("success")]
    correct = len([t for t in actionable if t.get("outcome") == "correct"])
    wrong = len([t for t in actionable if t.get("outcome") == "wrong"])
    win_rate = round(correct / len(actionable) * 100) if actionable else 0

    return {
        "snapshots": snapshots,
        "trades": trades[:20],
        "lessons": lessons,
        "latest": latest,
        "win_rate": win_rate,
        "correct": correct,
        "wrong": wrong,
        "actionable_count": len(actionable),
        "metrics_7d": metrics,
    }


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    db.init()
    logger.info("API server started — MySQL connected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
