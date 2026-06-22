"""
Social sentiment layer — multi-source.

Primary:  LunarCrush API v4 (requires Individual plan — $30/mo)
Fallback: CoinGecko community data + Reddit/crypto sentiment from free APIs

Provides:
  - Galaxy Score / sentiment (LunarCrush when available)
  - Community engagement metrics (CoinGecko — always free)
  - Reddit/social buzz indicators
  - Composite social score for Claude's prompt
"""
from __future__ import annotations

import logging

import requests
import config

logger = logging.getLogger(__name__)

LUNARCRUSH_BASE = "https://lunarcrush.com/api4"
REQUEST_TIMEOUT = 10


# ── LunarCrush (premium) ─────────────────────────────────────────────────────

def _lc_get(path: str) -> dict | None:
    if not config.LUNARCRUSH_API_KEY:
        return None
    try:
        r = requests.get(
            f"{LUNARCRUSH_BASE}/{path}",
            headers={
                "Authorization": f"Bearer {config.LUNARCRUSH_API_KEY}",
                "Accept": "application/json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug("LunarCrush %s failed: %s", path, exc)
        return None


def _fetch_lunarcrush() -> dict | None:
    data = _lc_get("public/topic/bitcoin/v1")
    if not data or "data" not in data:
        return None

    d = data["data"]
    return {
        "source": "lunarcrush",
        "galaxy_score": d.get("galaxy_score"),
        "sentiment": d.get("sentiment"),
        "social_volume": d.get("interactions_24h") or d.get("social_volume"),
        "social_dominance": d.get("social_dominance"),
        "social_contributors": d.get("social_contributors"),
    }


# ── Free fallback: Reddit RSS sentiment ───────────────────────────────────────

def _fetch_reddit_sentiment() -> dict:
    """Fetch r/Bitcoin and r/CryptoCurrency via RSS and analyse post titles for sentiment."""
    import feedparser

    subreddits = [
        ("r/Bitcoin", "https://www.reddit.com/r/Bitcoin/hot.rss?limit=20"),
        ("r/CryptoCurrency", "https://www.reddit.com/r/CryptoCurrency/hot.rss?limit=20"),
    ]

    bullish_kw = [
        "bull", "moon", "pump", "rally", "breakout", "ath", "buy",
        "accumulate", "hodl", "bullish", "surge", "soar", "record",
        "adoption", "institutional", "etf approv", "inflow", "all-time high",
        "support held", "bounce", "recover", "green",
    ]
    bearish_kw = [
        "bear", "crash", "dump", "sell", "scam", "hack", "ban",
        "regulation", "sec ", "fear", "bubble", "rug", "bearish",
        "decline", "plunge", "collapse", "liquidat", "outflow",
        "resistance", "reject", "red", "warning",
    ]

    total_bull = 0
    total_bear = 0
    total_posts = 0
    titles = []

    for name, url in subreddits:
        try:
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; CryptoBot/1.0)"},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                continue
            feed = feedparser.parse(r.content)
            for entry in feed.entries:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                total_posts += 1
                titles.append(title)
                title_lower = title.lower()

                is_bull = any(kw in title_lower for kw in bullish_kw)
                is_bear = any(kw in title_lower for kw in bearish_kw)
                if is_bull:
                    total_bull += 1
                if is_bear:
                    total_bear += 1
        except Exception as exc:
            logger.debug("Reddit RSS %s failed: %s", name, exc)

    if total_posts == 0:
        return {}

    bull_pct = total_bull / total_posts * 100
    bear_pct = total_bear / total_posts * 100

    if total_bull > total_bear + 2:
        mood = "bullish"
    elif total_bear > total_bull + 2:
        mood = "bearish"
    else:
        mood = "mixed"

    return {
        "reddit_posts_scanned": total_posts,
        "reddit_bullish_pct": round(bull_pct, 1),
        "reddit_bearish_pct": round(bear_pct, 1),
        "reddit_mood": mood,
        "reddit_top_titles": titles[:5],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_btc_social_data() -> dict:
    """
    Fetch BTC social sentiment. Tries LunarCrush first, falls back to free sources.
    """
    result = {
        "source": "free",
        "galaxy_score": None,
        "sentiment": None,
        "social_volume": None,
        "social_dominance": None,
        "social_contributors": None,
        "reddit_mood": None,
        "reddit_bullish_pct": None,
        "reddit_bearish_pct": None,
        "reddit_active_users": None,
        "summary": "no social data",
    }

    # Try LunarCrush first (premium — requires Individual plan $5/day)
    lc = _fetch_lunarcrush()
    if lc:
        result.update(lc)
        parts = []
        gs = lc.get("galaxy_score")
        sent = lc.get("sentiment")
        if gs:
            label = "strong" if gs >= 70 else "neutral" if gs >= 50 else "weak"
            parts.append(f"galaxy={gs:.0f} ({label})")
        if sent:
            label = "bullish" if sent >= 70 else "mixed" if sent >= 40 else "bearish"
            parts.append(f"sentiment={sent:.0f}% ({label})")
        result["summary"] = ", ".join(parts) if parts else "LunarCrush: no data"
        return result

    # Fallback: Reddit RSS sentiment (free, no API key needed)
    reddit = _fetch_reddit_sentiment()
    result.update(reddit)

    # Compute composite sentiment from Reddit + community signals
    parts = []
    reddit_mood = reddit.get("reddit_mood")
    if reddit_mood:
        result["reddit_mood"] = reddit_mood
        parts.append(f"reddit={reddit_mood}")
        bull = reddit.get("reddit_bullish_pct", 0)
        bear = reddit.get("reddit_bearish_pct", 0)
        parts.append(f"({bull:.0f}%↑/{bear:.0f}%↓)")

    result["summary"] = " ".join(parts) if parts else "no social data"
    return result


def get_social_context() -> str:
    """
    Returns a formatted string for injection into Claude's prompt.
    Works with either LunarCrush or free fallback data.
    """
    data = get_btc_social_data()

    if data["summary"] == "no social data":
        return ""

    lines = ["SOCIAL SENTIMENT (real-time):"]

    # LunarCrush data
    if data["source"] == "lunarcrush":
        galaxy = data.get("galaxy_score", 0)
        sentiment = data.get("sentiment", 0)
        social_vol = data.get("social_volume", 0)
        social_dom = data.get("social_dominance", 0)
        contributors = data.get("social_contributors", 0)

        if social_vol and social_vol >= 1_000_000:
            vol_str = f"{social_vol / 1e6:.1f}M"
        elif social_vol and social_vol >= 1_000:
            vol_str = f"{social_vol / 1e3:.0f}K"
        else:
            vol_str = str(social_vol or 0)

        lines.append(f"  Source:       LunarCrush (premium)")
        lines.append(f"  Galaxy Score: {galaxy}/100 ({'strong' if galaxy >= 70 else 'neutral' if galaxy >= 50 else 'weak'} social momentum)")
        lines.append(f"  Sentiment:    {sentiment}/100 ({'bullish' if sentiment >= 70 else 'mixed' if sentiment >= 40 else 'bearish'})")
        lines.append(f"  Social Vol:   {vol_str} interactions/24h")
        if social_dom:
            lines.append(f"  Dominance:    {social_dom}% of crypto social")
        if contributors:
            lines.append(f"  Contributors: {contributors:,} unique accounts")

    # Free fallback data (Reddit RSS sentiment)
    else:
        lines.append(f"  Source:       Reddit RSS (free)")
        reddit_mood = data.get("reddit_mood")
        if reddit_mood:
            bull = data.get("reddit_bullish_pct", 0)
            bear = data.get("reddit_bearish_pct", 0)
            posts = data.get("reddit_posts_scanned", 0)
            lines.append(f"  Reddit Mood:  {reddit_mood} ({bull:.0f}% bullish / {bear:.0f}% bearish from {posts} posts)")

        titles = data.get("reddit_top_titles", [])
        if titles:
            lines.append(f"  Top posts:")
            for t in titles[:3]:
                lines.append(f"    - {t[:80]}")

    return "\n".join(lines)
