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
from decimal import Decimal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


def _clean(obj):
    """Recursively convert Decimal/datetime to JSON-safe types."""
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

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
    return _clean(db.get_snapshots(limit=limit, since=since))


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
    return _clean(rows)


@app.get("/api/trades/period")
def get_trades_period(
    days: int = Query(7),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    end = datetime.now(timezone.utc)
    return _clean(db.get_trades_in_period(start, end))


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
    return _clean(db.get_events(limit=limit, level=level))


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.get("/api/analytics")
def get_analytics(
    days: int = Query(None),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    return _clean(analytics.compute_metrics(days))


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
    return _clean(rows)


@app.get("/api/watchlist")
def get_watchlist(x_api_key: str = Header(None)):
    _auth(x_api_key)
    return _clean(db.get_watchlist())


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

    return _clean({
        "snapshots": snapshots,
        "trades": trades[:20],
        "lessons": lessons,
        "latest": latest,
        "win_rate": win_rate,
        "correct": correct,
        "wrong": wrong,
        "actionable_count": len(actionable),
        "metrics_7d": metrics,
    })


# ── Bot Health (circuit breaker state) ────────────────────────────────────────

@app.get("/api/health/bot")
def get_bot_health(x_api_key: str = Header(None)):
    """Current risk state: drawdown, daily P&L, consecutive losses, streak."""
    _auth(x_api_key)

    snapshots = db.get_snapshots(limit=200)
    trades = db.get_recent_trades(20)

    # Current portfolio value
    latest_total = float(snapshots[-1]["total_usd"]) if snapshots else 0

    # Session peak + drawdown
    peak = 0.0
    for s in snapshots:
        v = float(s["total_usd"])
        if v > peak:
            peak = v
    drawdown_pct = round((peak - latest_total) / peak * 100, 2) if peak > 0 else 0

    # Daily P&L
    import datetime as dt
    today = dt.date.today().isoformat()
    daily_start = latest_total
    for s in snapshots:
        if str(s["created_at"])[:10] == today:
            daily_start = float(s["total_usd"])
            break
    daily_pnl = round(latest_total - daily_start, 2)
    daily_pnl_pct = round((latest_total / daily_start - 1) * 100, 2) if daily_start > 0 else 0

    # Consecutive losses
    consec_losses = 0
    for t in trades:
        if t.get("outcome") == "wrong":
            consec_losses += 1
        elif t.get("outcome") == "correct":
            break

    # Current streak
    streak_type = "none"
    streak_len = 0
    for t in trades:
        o = t.get("outcome")
        if o == "correct":
            if streak_type == "win":
                streak_len += 1
            elif streak_type == "none":
                streak_type = "win"
                streak_len = 1
            else:
                break
        elif o == "wrong":
            if streak_type == "loss":
                streak_len += 1
            elif streak_type == "none":
                streak_type = "loss"
                streak_len = 1
            else:
                break

    # Risk level
    risk_level = "normal"
    if drawdown_pct >= 20:
        risk_level = "halt"
    elif drawdown_pct >= 10:
        risk_level = "reduced"
    elif consec_losses >= 5:
        risk_level = "paused"
    elif daily_pnl_pct <= -3:
        risk_level = "daily_halt"
    elif drawdown_pct >= 5 or consec_losses >= 3:
        risk_level = "caution"

    return _clean({
        "portfolio_usd": latest_total,
        "session_peak_usd": peak,
        "drawdown_pct": drawdown_pct,
        "daily_pnl_usd": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "consecutive_losses": consec_losses,
        "streak": f"{streak_len} {'win' if streak_type == 'win' else 'loss'}" if streak_type != "none" else "none",
        "risk_level": risk_level,
        "thresholds": {
            "daily_loss_halt": config.DAILY_LOSS_HALT_PCT * 100,
            "drawdown_reduce": config.DRAWDOWN_REDUCE_PCT * 100,
            "drawdown_halt": config.DRAWDOWN_HALT_PCT * 100,
            "consec_loss_halt": config.CONSECUTIVE_LOSS_HALT,
        },
    })


# ── Single Trade Detail ──────────────────────────────────────────────────────

@app.get("/api/trades/{trade_id}")
def get_trade_detail(trade_id: int, x_api_key: str = Header(None)):
    """Full trade detail with all agent reasoning, market data, risk data."""
    _auth(x_api_key)
    row = db._execute(
        "SELECT * FROM trades WHERE id = %s", (trade_id,), fetch="one"
    )
    if not row:
        raise HTTPException(404, "Trade not found")
    if isinstance(row.get("decision"), str):
        row["decision"] = json.loads(row["decision"])
    if isinstance(row.get("market"), str):
        row["market"] = json.loads(row["market"])
    row["created_at"] = str(row["created_at"])
    return _clean(row)


# ── Live Derivatives Data ────────────────────────────────────────────────────

@app.get("/api/derivatives")
def get_derivatives(x_api_key: str = Header(None)):
    """Live BTC + ETH derivatives data from Binance Futures (public, no key)."""
    _auth(x_api_key)
    import requests

    result = {}
    for symbol, label in [("BTCUSDT", "btc"), ("ETHUSDT", "eth")]:
        data = {"funding_rate": None, "oi": None, "long_short_ratio": None,
                "long_pct": None, "short_pct": None}
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                             params={"symbol": symbol}, timeout=5)
            if r.ok:
                d = r.json()
                rate = float(d.get("lastFundingRate", 0))
                data["funding_rate"] = round(rate * 100, 4)
                data["funding_annual"] = round(rate * 3 * 365 * 100, 1)
        except Exception:
            pass
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/openInterest",
                             params={"symbol": symbol}, timeout=5)
            if r.ok:
                data["oi"] = round(float(r.json().get("openInterest", 0)), 2)
        except Exception:
            pass
        try:
            r = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                             params={"symbol": symbol, "period": "4h", "limit": 1}, timeout=5)
            if r.ok and r.json():
                d = r.json()[0]
                data["long_short_ratio"] = round(float(d.get("longShortRatio", 1)), 3)
                data["long_pct"] = round(float(d.get("longAccount", 0.5)) * 100, 1)
                data["short_pct"] = round(float(d.get("shortAccount", 0.5)) * 100, 1)
        except Exception:
            pass

        # Derivatives pressure label
        fr = data["funding_rate"] or 0
        if fr > 0.05:
            data["pressure"] = "overheated_longs"
        elif fr < -0.01:
            data["pressure"] = "short_squeeze_risk"
        else:
            data["pressure"] = "neutral"

        result[label] = data

    return result


# ── Backtest Runs ────────────────────────────────────────────────────────────

# ── Claude API Logs ──────────────────────────────────────────────────────────

@app.get("/api/claude-logs")
def get_claude_logs(
    limit: int = Query(50, le=200),
    cycle_id: str = Query(None),
    x_api_key: str = Header(None),
):
    """View every prompt sent to Claude and every response back."""
    _auth(x_api_key)
    return _clean(db.get_claude_logs(limit=limit, cycle_id=cycle_id))


# ── Coin Screening ───────────────────────────────────────────────────────────

@app.get("/api/screening")
def get_screening(x_api_key: str = Header(None)):
    """Latest coin screening results (top 50 by momentum)."""
    _auth(x_api_key)
    import coin_screener
    return _clean(coin_screener.get_latest_screening())


@app.get("/api/screening/run")
def run_screening(x_api_key: str = Header(None)):
    """Trigger a new screening scan."""
    _auth(x_api_key)
    import coin_screener
    results = coin_screener.run_screening(50)
    return _clean({"count": len(results), "top": results[:5] if results else []})


# ── Market Reports ───────────────────────────────────────────────────────────

@app.get("/api/reports")
def get_reports(
    limit: int = Query(10),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    import report_generator
    return _clean(report_generator.get_latest_reports(limit))


@app.get("/api/backtests")
def get_backtests(
    limit: int = Query(20),
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    rows = db._execute(
        "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT %s",
        (limit,),
        fetch="all",
    )
    for r in rows:
        r["created_at"] = str(r["created_at"])
        r["start_date"] = str(r.get("start_date", ""))
        r["end_date"] = str(r.get("end_date", ""))
        if isinstance(r.get("equity_curve"), str):
            r["equity_curve"] = json.loads(r["equity_curve"])
        if isinstance(r.get("config_snapshot"), str):
            r["config_snapshot"] = json.loads(r["config_snapshot"])
    return _clean(rows)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    db.init()
    logger.info("API server started — MySQL connected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
