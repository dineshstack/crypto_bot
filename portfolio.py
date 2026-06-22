"""SQLite-backed trade log and portfolio snapshot store."""
import json
import sqlite3
import os
from datetime import datetime

DB = os.path.join(os.path.dirname(__file__), "trades.db")


def init():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT,
                action      TEXT,
                amount_usd  REAL,
                btc_qty     REAL,
                price       REAL,
                decision    TEXT,
                market      TEXT,
                success     INTEGER,
                error       TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT,
                usdt        REAL,
                btc         REAL,
                price       REAL,
                total_usd   REAL
            )""")


def log_trade(result: dict, decision: dict, snapshot: dict):
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO trades (ts,action,amount_usd,btc_qty,price,decision,market,success,error) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(),
                result["action"],
                result.get("amount_usd", 0),
                result.get("btc_amount", 0),
                snapshot["price"],
                json.dumps(decision),
                json.dumps(snapshot),
                int(result["success"]),
                result.get("error"),
            ),
        )


def save_snapshot(usdt: float, btc: float, price: float) -> float:
    total = usdt + btc * price
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO snapshots (ts,usdt,btc,price,total_usd) VALUES (?,?,?,?,?)",
            (datetime.utcnow().isoformat(), usdt, btc, price, total),
        )
    return total


def recent_trades(n: int = 5) -> list:
    with sqlite3.connect(DB) as conn:
        return conn.execute(
            "SELECT ts,action,amount_usd,price,success,error FROM trades ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()


def first_snapshot_total() -> float | None:
    with sqlite3.connect(DB) as conn:
        row = conn.execute("SELECT total_usd FROM snapshots ORDER BY ts ASC LIMIT 1").fetchone()
        return row[0] if row else None
