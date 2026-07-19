import os
from dotenv import load_dotenv

load_dotenv()

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BINANCE_API_KEY   = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET    = os.getenv("BINANCE_SECRET", "")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")

# MySQL (local on VPS — replaces Supabase)
MYSQL_HOST     = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER     = os.getenv("MYSQL_USER", "crypto_bot")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "crypto_bot")

# API server key (for dashboard → VPS API authentication)
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")

# News research (optional — NewsAPI.org free tier: 100 req/day)
# Leave empty to rely on RSS feeds only (completely free, no API key)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# CoinGecko — optional demo key for higher rate limits (free at coingecko.com)
# Without a key, the free public API works fine (~30 req/min)
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")

# LunarCrush — social sentiment intelligence (lunarcrush.com/developers)
LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")

# Exchange
TESTNET = os.getenv("TESTNET", "true").lower() == "true"
SYMBOL  = "BTC/USDT"

# Multi-asset trading — ETH alongside BTC
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
ASSET_CONFIG = {
    "BTC/USDT": {
        "base": "BTC",
        "base_trade_usd": 5.0,
        "min_trade_usd": 2.0,
        "max_trade_usd": 15.0,
        "max_alloc_pct": 0.60,
        "ws_symbol": "btcusdt",
        "futures_symbol": "BTCUSDT",
    },
    "ETH/USDT": {
        "base": "ETH",
        "base_trade_usd": 3.0,
        "min_trade_usd": 2.0,
        "max_trade_usd": 10.0,
        "max_alloc_pct": 0.25,
        "ws_symbol": "ethusdt",
        "futures_symbol": "ETHUSDT",
    },
}

# Trading safety limits — code enforces these, Claude just suggests
BASE_TRADE_USD     = 5.0    # Claude's default suggestion
MIN_TRADE_USD      = 2.0    # Absolute floor per trade
MAX_TRADE_USD      = 15.0   # Absolute ceiling per trade (7.5% of $200)
MAX_BTC_ALLOC_PCT  = 0.60   # Stop buying if BTC > 60% of portfolio
MAX_TOTAL_CRYPTO_PCT = 0.80 # Max total crypto allocation (BTC+ETH) = 80%
STOP_LOSS_PCT      = 0.09   # Pause bot if total portfolio drops 9%

# Alpha-sleeve sizing: a validated ML gate (91.9% OOS win rate across 8
# disjoint windows — see ROADMAP.md Phase 0 exit) earns a larger clip than
# agent-led trades. Quarter-Kelly at the measured edge supports far more
# than 3x; the ramp stays conservative until the 300-trade bar (G3).
GATE_TRADE_MULT    = 3.0   # multiplier on risk-managed size when a gate fires
MAX_GATE_TRADE_USD = 45.0  # absolute ceiling for gate trades

# Circuit breaker thresholds (Phase-1 risk improvements)
DAILY_LOSS_HALT_PCT   = 0.03  # Pause for the day if portfolio drops 3% intraday
WEEKLY_LOSS_HALT_PCT  = 0.06  # Halt if portfolio drops 6% from the ISO week's start
DRAWDOWN_REDUCE_PCT   = 0.10  # Halve position sizing at 10% drawdown from session peak
DRAWDOWN_HALT_PCT     = 0.20  # Full halt at 20% drawdown from session peak
CONSECUTIVE_LOSS_HALT = 5     # Pause after 5 consecutive losing trades
CYCLE_FAILURE_HALT    = 3     # Halt after 3 consecutive failed analysis cycles

# Bot behaviour
ANALYSIS_INTERVAL_HOURS = 4   # Run Claude analysis every 4 hours

# Claude models
CLAUDE_MODEL         = "claude-haiku-4-5"   # frequent analysis — cheap
CLAUDE_DEEP_MODEL    = "claude-fable-5"     # deep reasoning: weekly review, thesis, reports, coin research
CLAUDE_DEEP_FALLBACK = "claude-opus-4-8"    # auto-serves the request if Fable's safety classifiers decline
CLAUDE_MAX_TOKENS = 512
