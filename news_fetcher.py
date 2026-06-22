"""
News research layer for the trading bot.

Fetches recent headlines from free RSS feeds (always) and optionally from
NewsAPI.org (if NEWS_API_KEY is set in .env — free tier: 100 req/day).

Sources:
  CRYPTO  — CoinDesk, CoinTelegraph, Bitcoin Magazine
  MACRO   — Reuters Business, CNBC Markets, MarketWatch
  GOLD    — Kitco News (gold often leads BTC as a safe-haven signal)

Headlines from the last LOOKBACK_HOURS are extracted, deduplicated,
and returned as a compact string for injection into Claude's prompt.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests
import config

logger = logging.getLogger(__name__)

LOOKBACK_HOURS  = 12       # Only show headlines newer than this
MAX_PER_CAT     = 4        # Max headlines per category in the prompt
REQUEST_TIMEOUT = 8        # Seconds per feed fetch

# ── Feed definitions ────────────────────────────────────────────────────────
# All free RSS feeds — no API key required
RSS_FEEDS = {
    "crypto": [
        ("CoinDesk",       "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph",  "https://cointelegraph.com/rss"),
        ("Bitcoin Mag",    "https://bitcoinmagazine.com/feed"),
    ],
    "macro": [
        ("Reuters Biz",    "https://feeds.reuters.com/reuters/businessNews"),
        ("CNBC Markets",   "https://www.cnbc.com/id/10001147/device/rss/rss.html"),
        ("MarketWatch",    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
    ],
    "gold": [
        ("Kitco",          "https://www.kitco.com/news/kitco-rss.xml"),
        ("Investing Gold", "https://www.investing.com/rss/news_14.rss"),
    ],
}

# NewsAPI.org query terms (optional — requires NEWS_API_KEY)
NEWSAPI_QUERIES = {
    "crypto": "bitcoin OR cryptocurrency OR BTC",
    "macro":  "federal reserve OR inflation OR interest rate OR economic",
    "gold":   "gold price OR safe haven OR commodities",
}

NEWSAPI_URL = "https://newsapi.org/v2/everything"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)


def _parse_date(entry) -> datetime | None:
    """Try to extract a timezone-aware datetime from a feedparser entry."""
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
    # Fallback: assume now if no date (treat as fresh)
    return datetime.now(timezone.utc)


def _clean(title: str) -> str:
    """Remove HTML entities and excess whitespace from a headline."""
    title = re.sub(r"<[^>]+>", "", title)
    title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return " ".join(title.split())


# ── RSS fetcher ──────────────────────────────────────────────────────────────

def _fetch_rss(label: str, url: str, cutoff: datetime) -> list[tuple[datetime, str, str]]:
    """
    Fetch one RSS feed. Returns list of (published_dt, source_label, title).
    Never raises — returns [] on failure.
    """
    try:
        # feedparser can be slow; use requests with timeout then parse bytes
        resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; CryptoBot/1.0)"})
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as exc:
        logger.debug("RSS fetch failed [%s]: %s", label, exc)
        return []

    results = []
    for entry in feed.entries:
        dt = _parse_date(entry)
        if dt and dt >= cutoff:
            title = _clean(getattr(entry, "title", ""))
            if title:
                results.append((dt, label, title))

    return results


# ── NewsAPI fetcher (optional) ───────────────────────────────────────────────

def _fetch_newsapi(category: str, query: str, cutoff: datetime) -> list[tuple[datetime, str, str]]:
    """Call NewsAPI if NEWS_API_KEY is configured. Returns same tuple format."""
    if not config.NEWS_API_KEY:
        return []
    try:
        from_dt = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        resp = requests.get(
            NEWSAPI_URL,
            params={
                "q":        query,
                "from":     from_dt,
                "sortBy":   "publishedAt",
                "language": "en",
                "pageSize": 10,
                "apiKey":   config.NEWS_API_KEY,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for art in data.get("articles", []):
            raw_dt = art.get("publishedAt", "")
            try:
                dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now(timezone.utc)
            title = _clean(art.get("title") or "")
            source = art.get("source", {}).get("name", "NewsAPI")
            if title and "[Removed]" not in title:
                results.append((dt, source, title))
        return results
    except Exception as exc:
        logger.debug("NewsAPI fetch failed [%s]: %s", category, exc)
        return []


# ── Public API ───────────────────────────────────────────────────────────────

def get_news_context() -> str:
    """
    Fetch recent headlines from all sources. Returns a compact multi-line
    string ready to drop into Claude's prompt, or empty string if all feeds fail.

    Example output:
      CRYPTO NEWS (last 12h):
        [CoinDesk] Bitcoin ETF inflows hit record $500M amid institutional surge
        [CoinTelegraph] SEC delays Ethereum spot ETF decision to August
      MACRO NEWS (last 12h):
        [Reuters Biz] Fed signals rate hold through summer as inflation cools
        [CNBC Markets] US jobs data beats expectations — dollar strengthens
      GOLD NEWS (last 12h):
        [Kitco] Gold slides $18 as risk appetite returns to equity markets
    """
    cutoff = _cutoff()
    sections = []

    category_labels = {
        "crypto": "CRYPTO NEWS",
        "macro":  "MACRO / MARKETS NEWS",
        "gold":   "GOLD NEWS",
    }

    for cat, feeds in RSS_FEEDS.items():
        items: list[tuple[datetime, str, str]] = []

        # RSS (always attempted)
        for label, url in feeds:
            items.extend(_fetch_rss(label, url, cutoff))

        # NewsAPI (only if key configured and RSS gave few results)
        if len(items) < 2:
            items.extend(_fetch_newsapi(cat, NEWSAPI_QUERIES[cat], cutoff))

        if not items:
            continue

        # Sort newest first, deduplicate similar titles, cap count
        items.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        unique: list[tuple[datetime, str, str]] = []
        for dt, src, title in items:
            key = title[:40].lower()
            if key not in seen:
                seen.add(key)
                unique.append((dt, src, title))
            if len(unique) >= MAX_PER_CAT:
                break

        label = category_labels.get(cat, cat.upper())
        lines = [f"  [{src}] {title}" for _, src, title in unique]
        sections.append(f"{label} (last {LOOKBACK_HOURS}h):\n" + "\n".join(lines))

    if not sections:
        return ""

    return "\n\n".join(sections)


def get_market_sentiment_summary(news_context: str) -> str:
    """
    Returns a one-line sentiment indicator derived from news keywords.
    Used for logging; Claude interprets the full headlines itself.
    """
    if not news_context:
        return "no news data"

    text = news_context.lower()
    bullish_kw = ["surge", "rally", "record", "etf inflow", "adoption",
                  "bullish", "upside", "gain", "rise", "institutional"]
    bearish_kw = ["crash", "drop", "ban", "hack", "regulation", "sell-off",
                  "decline", "bearish", "fear", "recession", "rate hike"]

    bull = sum(1 for w in bullish_kw if w in text)
    bear = sum(1 for w in bearish_kw if w in text)

    if bull > bear + 1:
        return f"bullish ({bull}↑ vs {bear}↓ signals)"
    if bear > bull + 1:
        return f"bearish ({bear}↓ vs {bull}↑ signals)"
    return f"neutral ({bull}↑ {bear}↓ signals)"
