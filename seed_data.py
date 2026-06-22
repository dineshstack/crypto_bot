"""
Seed Supabase with sample data so the dashboard has something to show.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

import database as db

now = datetime.now(timezone.utc)


def seed():
    db.init()
    print("Seeding sample data...\n")

    # Portfolio snapshots (simulating 7 days of 4h cycles = 42 snapshots)
    base_price = 63000.0
    usdt = 180.0
    btc = 0.0003
    prices = [
        63000, 63200, 63500, 63100, 62800, 62500,
        62200, 62600, 63000, 63400, 63800, 64200,
        64000, 63700, 63500, 63800, 64100, 64500,
        64800, 65000, 64700, 64300, 64600, 64900,
        65200, 65500, 65100, 64800, 64500, 64200,
        64600, 64900, 65200, 65500, 65800, 66000,
        65700, 65400, 65100, 65400, 65700, 64100,
    ]

    snapshots_inserted = 0
    for i, price in enumerate(prices):
        t = now - timedelta(hours=(len(prices) - i) * 4)
        total = usdt + btc * price

        # Simulate portfolio changes from trades
        if i == 6:   usdt -= 5; btc += 5 / price
        if i == 12:  usdt -= 8; btc += 8 / price
        if i == 18:  btc -= 0.00005; usdt += 0.00005 * price
        if i == 24:  usdt -= 10; btc += 10 / price
        if i == 30:  usdt -= 5; btc += 5 / price
        if i == 36:  btc -= 0.0001; usdt += 0.0001 * price

        total = usdt + btc * price
        db._db().table("portfolio_snapshots").insert({
            "created_at": t.isoformat(),
            "usdt": round(usdt, 2),
            "btc": round(btc, 8),
            "price": price,
            "total_usd": round(total, 2),
        }).execute()
        snapshots_inserted += 1

    print(f"  ✅ {snapshots_inserted} portfolio snapshots")

    # Trades
    sample_trades = [
        {
            "offset_hours": 160, "action": "buy", "amount_usd": 5.0,
            "price": 62500, "success": True, "outcome": "correct",
            "price_after_4h": 63000,
            "reason": "RSI oversold at 28, Fear & Greed extreme fear — DCA opportunity",
            "confidence": 0.78, "signals": ["rsi_oversold", "extreme_fear", "below_sma20"],
        },
        {
            "offset_hours": 144, "action": "hold", "amount_usd": 0,
            "price": 63000, "success": True, "outcome": None,
            "price_after_4h": None,
            "reason": "RSI recovering, wait for confirmation above SMA20",
            "confidence": 0.60, "signals": ["rsi_neutral", "approaching_sma20"],
        },
        {
            "offset_hours": 120, "action": "buy", "amount_usd": 8.0,
            "price": 63700, "success": True, "outcome": "correct",
            "price_after_4h": 64200,
            "reason": "Broke above SMA20, bullish ETF inflow news, moderate fear",
            "confidence": 0.82, "signals": ["above_sma20", "bullish_news", "moderate_fear"],
        },
        {
            "offset_hours": 96, "action": "sell", "amount_usd": 3.25,
            "price": 65000, "success": True, "outcome": "wrong",
            "price_after_4h": 65500,
            "reason": "RSI approaching overbought at 68, take partial profit",
            "confidence": 0.55, "signals": ["rsi_high", "near_bb_upper"],
        },
        {
            "offset_hours": 72, "action": "hold", "amount_usd": 0,
            "price": 64800, "success": True, "outcome": None,
            "price_after_4h": None,
            "reason": "Mixed signals — RSI neutral, news conflicting, capital preservation",
            "confidence": 0.65, "signals": ["rsi_neutral", "conflicting_news"],
        },
        {
            "offset_hours": 48, "action": "buy", "amount_usd": 10.0,
            "price": 64200, "success": True, "outcome": "correct",
            "price_after_4h": 65200,
            "reason": "Gold rallying as safe haven, BTC following — strong buy signal",
            "confidence": 0.85, "signals": ["gold_rally", "safe_haven_flow", "below_sma20"],
        },
        {
            "offset_hours": 24, "action": "buy", "amount_usd": 5.0,
            "price": 65500, "success": True, "outcome": "neutral",
            "price_after_4h": 65700,
            "reason": "Continuation of uptrend, DCA into strength with moderate size",
            "confidence": 0.70, "signals": ["uptrend", "above_sma20", "moderate_volume"],
        },
        {
            "offset_hours": 8, "action": "sell", "amount_usd": 6.55,
            "price": 65400, "success": True, "outcome": "correct",
            "price_after_4h": 64100,
            "reason": "RSI overbought at 72, negative macro news — take profit before pullback",
            "confidence": 0.75, "signals": ["rsi_overbought", "bearish_macro", "near_resistance"],
        },
    ]

    trades_inserted = 0
    for t in sample_trades:
        created = now - timedelta(hours=t["offset_hours"])
        decision = {
            "action": t["action"],
            "trade_usd": t["amount_usd"],
            "confidence": t["confidence"],
            "risk": "medium",
            "reason": t["reason"],
            "signals": t["signals"],
        }
        btc_qty = t["amount_usd"] / t["price"] if t["action"] != "hold" else 0
        row = {
            "created_at": created.isoformat(),
            "action": t["action"],
            "amount_usd": t["amount_usd"],
            "btc_qty": round(btc_qty, 8),
            "price": t["price"],
            "decision": decision,
            "market": {"price": t["price"], "rsi": 50, "fear_greed": 35, "fear_greed_lbl": "Fear"},
            "success": t["success"],
            "outcome": t["outcome"],
            "price_after_4h": t.get("price_after_4h"),
            "lesson_generated": t["outcome"] == "wrong",
        }
        db._db().table("trades").insert(row).execute()
        trades_inserted += 1

    print(f"  ✅ {trades_inserted} trades")

    # Lessons
    sample_lessons = [
        ("Avoid selling when RSI is between 55-70 — premature profit-taking misses larger moves", "self_correction"),
        ("Do not act on news sentiment alone when technicals are neutral — prefer HOLD", "weekly_review"),
        ("Only buy during extreme fear when RSI confirms oversold below 35", "self_correction"),
    ]
    for text, source in sample_lessons:
        db.save_lesson(text, source)
    print(f"  ✅ {len(sample_lessons)} lessons")

    # Weekly review
    db.save_weekly_review(
        period_start=now - timedelta(days=7),
        period_end=now,
        total=8,
        correct=4,
        wrong=1,
        pnl=3.25,
        review_text=(
            "SUMMARY: Solid week with conservative approach. 4 correct trades, 1 wrong "
            "(premature sell at $65K). Portfolio grew from $200 to $203.25. The self-correction "
            "system caught the early sell mistake and generated a useful lesson about RSI ranges. "
            "News integration helped identify the gold-driven rally correctly. Main area for "
            "improvement: avoid selling into strength just because RSI approaches 70."
        ),
    )
    print("  ✅ 1 weekly review")

    # Bot events
    events = [
        (48, "info", "bot_start", "Bot started — baseline $200.00"),
        (47, "info", "cycle_start", "Analysis cycle started"),
        (47, "info", "trade", "BUY $10.00"),
        (43, "info", "cycle_start", "Analysis cycle started"),
        (43, "info", "trade", "HOLD $0.00"),
        (39, "info", "cycle_start", "Analysis cycle started"),
        (39, "info", "trade", "BUY $5.00"),
        (39, "info", "lesson", "Avoid selling when RSI is between 55-70"),
        (35, "info", "cycle_start", "Analysis cycle started"),
        (35, "info", "trade", "HOLD $0.00"),
        (24, "warning", "stop_loss", "Portfolio approached stop-loss threshold"),
        (20, "info", "cycle_start", "Analysis cycle started"),
        (20, "info", "trade", "SELL $6.55"),
        (8, "info", "cycle_start", "Analysis cycle started"),
        (8, "info", "trade", "HOLD $0.00"),
        (4, "error", "error", "RSS feed timeout — CoinTelegraph unreachable"),
        (2, "info", "cycle_start", "Analysis cycle started"),
    ]
    for offset_h, level, event, message in events:
        created = now - timedelta(hours=offset_h)
        db._db().table("bot_events").insert({
            "created_at": created.isoformat(),
            "level": level,
            "event": event,
            "message": message,
            "data": {},
        }).execute()
    print(f"  ✅ {len(events)} bot events")

    print("\n✅ All sample data seeded! Refresh your dashboard at http://localhost:3000")


if __name__ == "__main__":
    seed()
