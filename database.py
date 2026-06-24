"""
MySQL-backed data layer.
All trade logging, snapshots, lessons, and reviews live here.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pymysql
from pymysql.cursors import DictCursor

import config

logger = logging.getLogger(__name__)

_pool: pymysql.Connection | None = None


def _db() -> pymysql.Connection:
    """Get or create a MySQL connection (auto-reconnects)."""
    global _pool
    if _pool is None or not _pool.open:
        _pool = pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=True,
            connect_timeout=10,
            read_timeout=30,
            write_timeout=30,
        )
    else:
        _pool.ping(reconnect=True)
    return _pool


def _execute(sql: str, params: tuple = (), fetch: str = "none") -> list | dict | int:
    """Execute SQL and return results. fetch: 'none', 'one', 'all'."""
    conn = _db()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        if fetch == "one":
            return cur.fetchone()
        elif fetch == "all":
            return cur.fetchall() or []
        return cur.lastrowid


# ── Bootstrap ───────────────────────────────────────────────────────────────

def _column_exists(table: str, column: str) -> bool:
    """Check if a column exists in the given table (MySQL-safe)."""
    row = _execute(
        """SELECT COUNT(*) AS cnt FROM INFORMATION_SCHEMA.COLUMNS
           WHERE TABLE_SCHEMA = DATABASE()
             AND TABLE_NAME = %s AND COLUMN_NAME = %s""",
        (table, column),
        fetch="one",
    )
    return bool(row and row.get("cnt", 0))


def _migrate_lessons():
    """
    Idempotent migration — adds new columns to the lessons table.
    Safe to run on every startup; skips columns that already exist.

    New columns:
      trade_id      INT NULL        — ID of the trade that triggered the lesson
      category      VARCHAR(50) NULL — timing | risk_management | technical | sentiment | macro
      times_applied INT DEFAULT 0   — how many times Claude cited this lesson in a decision
    """
    migrations = [
        ("trade_id",      "ALTER TABLE lessons ADD COLUMN trade_id INT NULL"),
        ("category",      "ALTER TABLE lessons ADD COLUMN category VARCHAR(50) NULL"),
        ("times_applied", "ALTER TABLE lessons ADD COLUMN times_applied INT NOT NULL DEFAULT 0"),
    ]
    for col, sql in migrations:
        if not _column_exists("lessons", col):
            try:
                _execute(sql)
                logger.info("DB migration: added lessons.%s", col)
            except Exception as exc:
                logger.warning("DB migration failed for lessons.%s: %s", col, exc)


def init():
    """Verify MySQL connection and run idempotent migrations."""
    try:
        _execute("SELECT 1", fetch="one")
        logger.info("MySQL connected OK (%s@%s/%s)",
                     config.MYSQL_USER, config.MYSQL_HOST, config.MYSQL_DATABASE)
        _migrate_lessons()
    except Exception as exc:
        logger.error("MySQL connection failed: %s", exc)
        raise


# ── Trades ───────────────────────────────────────────────────────────────────

def log_trade(result: dict, decision: dict, snapshot: dict) -> str:
    """Insert a trade record; return its ID."""
    row_id = _execute(
        """INSERT INTO trades (action, amount_usd, btc_qty, price, decision, market, success, error)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            result["action"],
            result.get("amount_usd", 0),
            result.get("btc_amount", 0),
            snapshot["price"],
            json.dumps(decision),
            json.dumps(snapshot),
            1 if result["success"] else 0,
            result.get("error"),
        ),
    )
    return str(row_id)


def get_recent_trades(n: int = 5) -> list[dict]:
    rows = _execute(
        """SELECT id, created_at, action, amount_usd, price, success, error,
                  outcome, decision
           FROM trades ORDER BY created_at DESC LIMIT %s""",
        (n,),
        fetch="all",
    )
    for r in rows:
        if isinstance(r.get("decision"), str):
            r["decision"] = json.loads(r["decision"])
        r["created_at"] = str(r["created_at"])
    return rows


def get_last_unevaluated_trade() -> dict | None:
    """Most recent successful buy/sell that hasn't been outcome-evaluated yet."""
    return _execute(
        """SELECT * FROM trades
           WHERE action != 'hold' AND outcome IS NULL AND success = 1
           ORDER BY created_at DESC LIMIT 1""",
        fetch="one",
    )


def update_trade_outcome(trade_id: str, outcome: str, price_after: float):
    _execute(
        """UPDATE trades SET outcome = %s, price_after_4h = %s,
                  lesson_generated = %s WHERE id = %s""",
        (outcome, price_after, 1 if outcome == "wrong" else 0, trade_id),
    )


def get_trades_in_period(start: datetime, end: datetime) -> list[dict]:
    rows = _execute(
        """SELECT * FROM trades
           WHERE created_at >= %s AND created_at <= %s
           ORDER BY created_at""",
        (start, end),
        fetch="all",
    )
    for r in rows:
        if isinstance(r.get("decision"), str):
            r["decision"] = json.loads(r["decision"])
        if isinstance(r.get("market"), str):
            r["market"] = json.loads(r["market"])
        r["created_at"] = str(r["created_at"])
    return rows


# ── Portfolio snapshots ───────────────────────────────────────────────────────

def save_snapshot(usdt: float, btc: float, price: float) -> float:
    """Save portfolio snapshot, return total USD value."""
    total = usdt + btc * price
    _execute(
        "INSERT INTO portfolio_snapshots (usdt, btc, price, total_usd) VALUES (%s, %s, %s, %s)",
        (usdt, btc, price, total),
    )
    return total


def get_first_snapshot_total() -> float | None:
    row = _execute(
        "SELECT total_usd FROM portfolio_snapshots ORDER BY created_at LIMIT 1",
        fetch="one",
    )
    return float(row["total_usd"]) if row else None


# ── Lessons ───────────────────────────────────────────────────────────────────

def save_lesson(
    text: str,
    source: str,
    trade_id: str | None = None,
    category: str | None = None,
) -> str:
    """
    Save a lesson. category is one of:
      timing | risk_management | technical | sentiment | macro
    """
    row_id = _execute(
        "INSERT INTO lessons (lesson, source, trade_id, category, active) VALUES (%s, %s, %s, %s, 1)",
        (text, source, trade_id, category),
    )
    return str(row_id)


def increment_lesson_applied(lesson_id: int):
    """Increment times_applied counter when a lesson is cited in a decision."""
    try:
        _execute(
            "UPDATE lessons SET times_applied = times_applied + 1 WHERE id = %s",
            (lesson_id,),
        )
    except Exception as exc:
        logger.debug("increment_lesson_applied failed: %s", exc)


def get_active_lessons(limit: int = 5) -> list[str]:
    rows = _execute(
        "SELECT lesson FROM lessons WHERE active = 1 ORDER BY created_at DESC LIMIT %s",
        (limit,),
        fetch="all",
    )
    return [r["lesson"] for r in rows]


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
    _execute(
        """INSERT INTO weekly_reviews
           (period_start, period_end, total_trades, correct_trades, wrong_trades, pnl_usd, review_text)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (period_start, period_end, total, correct, wrong, pnl, review_text),
    )


def get_last_weekly_review_date() -> datetime | None:
    row = _execute(
        "SELECT created_at FROM weekly_reviews ORDER BY created_at DESC LIMIT 1",
        fetch="one",
    )
    if row:
        ts = row["created_at"]
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        return ts.replace(tzinfo=timezone.utc)
    return None


# ── Pending confirmations ─────────────────────────────────────────────────────

def save_pending_confirmation(
    conf_id:   str,
    decision:  dict,
    market:    dict,
    portfolio: dict,
):
    _execute(
        """INSERT INTO pending_confirmations (id, decision, market, portfolio, status)
           VALUES (%s, %s, %s, %s, 'pending')""",
        (conf_id, json.dumps(decision), json.dumps(market), json.dumps(portfolio)),
    )


def update_confirmation_status(conf_id: str, status: str):
    _execute(
        "UPDATE pending_confirmations SET status = %s WHERE id = %s",
        (status, conf_id),
    )


# ── Bot events ────────────────────────────────────────────────────────────────

def log_event(event: str, message: str, level: str = "info", data: dict = None):
    """Log a structured event for the dashboard activity feed."""
    try:
        _execute(
            "INSERT INTO bot_events (event, message, level, data) VALUES (%s, %s, %s, %s)",
            (event, message, level, json.dumps(data or {})),
        )
    except Exception as exc:
        logger.debug("Event log failed: %s", exc)


def get_events(limit: int = 50, level: str = None) -> list[dict]:
    if level:
        rows = _execute(
            "SELECT * FROM bot_events WHERE level = %s ORDER BY created_at DESC LIMIT %s",
            (level, limit),
            fetch="all",
        )
    else:
        rows = _execute(
            "SELECT * FROM bot_events ORDER BY created_at DESC LIMIT %s",
            (limit,),
            fetch="all",
        )
    for r in rows:
        if isinstance(r.get("data"), str):
            r["data"] = json.loads(r["data"])
        r["created_at"] = str(r["created_at"])
    return rows


# ── Claude API logs ───────────────────────────────────────────────────────────

def log_claude_call(cycle_id: str, agent: str, model: str,
                    prompt: str, response: str,
                    tokens_in: int = 0, tokens_out: int = 0,
                    duration_ms: int = 0):
    """Log a Claude API call for the audit trail."""
    try:
        _execute(
            """INSERT INTO claude_api_logs
               (cycle_id, agent, model, prompt, response, tokens_in, tokens_out, duration_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (cycle_id, agent, model, prompt[:10000], response[:10000],
             tokens_in, tokens_out, duration_ms),
        )
    except Exception as exc:
        logger.debug("Claude log failed: %s", exc)


def get_claude_logs(limit: int = 50, cycle_id: str = None) -> list[dict]:
    if cycle_id:
        rows = _execute(
            "SELECT * FROM claude_api_logs WHERE cycle_id = %s ORDER BY created_at",
            (cycle_id,),
            fetch="all",
        )
    else:
        rows = _execute(
            "SELECT * FROM claude_api_logs ORDER BY created_at DESC LIMIT %s",
            (limit,),
            fetch="all",
        )
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return rows


# ── Coin research (used by coin_researcher.py) ───────────────────────────────

def insert_coin_research(data: dict) -> int:
    """Insert a coin research row, return its ID."""
    return _execute(
        """INSERT INTO coin_research
           (coin_id, symbol, name, investment_score, team_score, technology_score,
            market_score, tokenomics_score, usecase_score, verdict, suggested_usd,
            hold_months, risks, opportunities, summary, price_usd, market_cap_usd,
            volume_24h_usd, price_change_7d, github_commits_4w, twitter_followers, raw_data)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            data.get("coin_id"), data.get("symbol"), data.get("name"),
            data.get("investment_score", 0), data.get("team_score", 0),
            data.get("technology_score", 0), data.get("market_score", 0),
            data.get("tokenomics_score", 0), data.get("usecase_score", 0),
            data.get("verdict"), data.get("suggested_usd", 0),
            data.get("hold_months", 0),
            json.dumps(data.get("risks", [])),
            json.dumps(data.get("opportunities", [])),
            data.get("summary"),
            data.get("price_usd"), data.get("market_cap_usd"),
            data.get("volume_24h_usd"), data.get("price_change_7d"),
            data.get("github_commits_4w"), data.get("twitter_followers"),
            json.dumps(data.get("raw_data")) if data.get("raw_data") else None,
        ),
    )


def upsert_watchlist(data: dict):
    """Insert or update a watchlist entry."""
    _execute(
        """INSERT INTO coin_watchlist (coin_id, symbol, name, entry_price, target_usd, research_id, active)
           VALUES (%s, %s, %s, %s, %s, %s, 1)
           ON DUPLICATE KEY UPDATE
             symbol = VALUES(symbol), name = VALUES(name),
             entry_price = VALUES(entry_price), target_usd = VALUES(target_usd),
             research_id = VALUES(research_id), active = 1""",
        (
            data["coin_id"], data["symbol"], data.get("name"),
            data.get("entry_price", 0), data.get("target_usd", 0),
            data.get("research_id"),
        ),
    )


def update_research_watchlist(research_id: int):
    """Mark a research row as watchlisted."""
    _execute("UPDATE coin_research SET on_watchlist = 1 WHERE id = %s", (research_id,))


def get_watchlist() -> list[dict]:
    rows = _execute(
        "SELECT * FROM coin_watchlist WHERE active = 1 ORDER BY created_at DESC",
        fetch="all",
    )
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return rows


def get_recent_research(limit: int = 10) -> list[dict]:
    rows = _execute(
        """SELECT id, symbol, name, investment_score, verdict, created_at, on_watchlist, summary
           FROM coin_research ORDER BY created_at DESC LIMIT %s""",
        (limit,),
        fetch="all",
    )
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return rows


def get_research_by_id(research_id: int) -> dict | None:
    return _execute(
        "SELECT coin_id, name, suggested_usd FROM coin_research WHERE id = %s",
        (research_id,),
        fetch="one",
    )


# ── Analytics helpers (used by analytics.py) ──────────────────────────────────

def get_snapshots(limit: int = 2000, since: str = None) -> list[dict]:
    """Get portfolio snapshots for equity curve (includes BTC price for benchmark)."""
    if since:
        rows = _execute(
            """SELECT created_at, total_usd, price FROM portfolio_snapshots
               WHERE created_at >= %s ORDER BY created_at LIMIT %s""",
            (since, limit),
            fetch="all",
        )
    else:
        rows = _execute(
            "SELECT created_at, total_usd, price FROM portfolio_snapshots ORDER BY created_at LIMIT %s",
            (limit,),
            fetch="all",
        )
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return rows
