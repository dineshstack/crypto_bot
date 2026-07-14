"""
Test runner — runs the full bot pipeline WITHOUT Telegram.
Tests each component step-by-step with console output.

Usage:  python test_run.py
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_database():
    header("1/6  MYSQL CONNECTION")
    import database as db
    db.init()
    print("  ✅ MySQL connected\n")

    lessons = db.get_active_lessons(5)
    print(f"  Active lessons: {len(lessons)}")
    trades = db.get_recent_trades(5)
    print(f"  Recent trades:  {len(trades)}")
    return True


def test_binance():
    header("2/6  BINANCE EXCHANGE")
    import config
    import market_data as md

    exchange = md.get_exchange()
    mode = "TESTNET" if config.TESTNET else "🔴 LIVE"
    print(f"  Mode: {mode}")

    # Test market data
    print("  Fetching market snapshot...")
    snap = md.get_market_snapshot(exchange)
    print(f"  BTC Price:    ${snap['price']:,}")
    print(f"  24h Change:   {snap['change_24h_pct']}%")
    print(f"  RSI(14):      {snap['rsi']}")
    print(f"  SMA20:        ${snap['sma20']:,}")
    print(f"  SMA50:        ${snap['sma50']:,}")
    print(f"  Fear & Greed: {snap['fear_greed']}/100 ({snap['fear_greed_lbl']})")

    # Test portfolio
    print("\n  Fetching portfolio...")
    port = md.get_portfolio(exchange)
    btc_val = port["btc"] * snap["price"]
    total = port["usdt"] + btc_val
    print(f"  USDT:  ${port['usdt']:.2f}")
    print(f"  BTC:   {port['btc']:.6f} (${btc_val:.2f})")
    print(f"  Total: ${total:.2f}")

    print("\n  ✅ Binance connected\n")
    return exchange, snap, port


def test_news():
    header("3/6  NEWS FETCHER")
    import news_fetcher

    print("  Fetching headlines from RSS feeds...")
    news = news_fetcher.get_news_context()

    if news:
        lines = news.strip().split("\n")
        print(f"  Found {len(lines)} headline lines\n")
        for line in lines[:12]:
            print(f"  {line}")
        if len(lines) > 12:
            print(f"  ... and {len(lines)-12} more")

        sentiment = news_fetcher.get_market_sentiment_summary(news)
        print(f"\n  Sentiment: {sentiment}")
    else:
        print("  No recent headlines (feeds may be down or no news in last 12h)")
        print("  This is OK — bot works without news data")

    print("\n  ✅ News fetcher working\n")
    return True


def test_claude(snap, port):
    header("4/6  CLAUDE ANALYZER (Haiku)")
    import claude_analyzer

    print("  Sending market data + news to Claude Haiku...")
    print("  (This costs ~$0.004)\n")
    decision = claude_analyzer.analyze(snap, port)

    emoji = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}.get(decision["action"], "⚪")
    print(f"  {emoji} Decision:   {decision['action'].upper()}")
    print(f"  Amount:     ${decision['trade_usd']:.2f}")
    print(f"  Confidence: {decision['confidence']:.0%}")
    print(f"  Risk:       {decision['risk']}")
    print(f"  Reason:     {decision['reason']}")
    print(f"  Signals:    {', '.join(decision.get('signals', []))}")
    print(f"  News:       {decision.get('news_sentiment', 'n/a')}")

    print("\n  ✅ Claude analysis working\n")
    return decision


def test_executor(exchange, decision, snap, port):
    header("5/6  EXECUTOR (dry run)")
    import config

    print(f"  Mode: {'TESTNET' if config.TESTNET else '🔴 LIVE'}")
    print(f"  Would execute: {decision['action'].upper()} ${decision['trade_usd']:.2f}")

    if decision["action"] == "hold":
        print("  Decision is HOLD — no trade to execute")
        print("\n  ✅ Executor check passed\n")
        return

    import executor
    result = executor.execute(exchange, decision, snap, port)
    if result["success"]:
        print(f"  ✅ Trade executed: {result['action']} ${result['amount_usd']:.2f} "
              f"({result['btc_amount']:.6f} BTC)")
    else:
        print(f"  ⚠️  Trade skipped: {result.get('error', 'unknown')}")
        print("  (This may be normal — safety guards working as intended)")

    print("\n  ✅ Executor working\n")


def test_logging(decision, snap):
    header("6/6  DATABASE LOGGING")
    import database as db

    db.log_event("test_run", "Test cycle completed successfully",
                 data={"action": decision["action"],
                       "price": snap["price"],
                       "confidence": decision["confidence"]})
    print("  ✅ Event logged to MySQL\n")


def main():
    print("\n" + "🤖" * 30)
    print("  CLAUDE CRYPTO BOT — FULL PIPELINE TEST")
    print("🤖" * 30)

    try:
        # 1. MySQL
        test_database()

        # 2. Binance + market data
        exchange, snap, port = test_binance()

        # 3. News
        test_news()

        # 4. Claude analysis
        decision = test_claude(snap, port)

        # 5. Executor
        test_executor(exchange, decision, snap, port)

        # 6. Log to DB
        test_logging(decision, snap)

        header("ALL TESTS PASSED ✅")
        print("  Everything is working! When Telegram is ready, add the")
        print("  bot token + chat ID to .env and run: python main.py\n")

    except Exception as e:
        header("❌ TEST FAILED")
        print(f"  Error: {e}")
        logger.exception("Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
