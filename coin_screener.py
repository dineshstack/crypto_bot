"""
Multi-coin screening pipeline — daily scan of top 50 cryptocurrencies.

Fetches market data from CoinGecko (free, no API key required) and scores
each coin on momentum, volume, and trend. Classifies into risk tiers.

For advisors: "Which coins look strongest right now?" answered with data.

Risk Tiers:
  Tier 1: Market cap > $50B (BTC, ETH, SOL, XRP, BNB)
  Tier 2: Market cap $1B - $50B (established mid-caps)
  Tier 3: Market cap < $1B (high risk, high potential)

Momentum Score (0-100):
  Based on 7d change, 30d change, volume/cap ratio, and relative strength.
  Higher = stronger momentum. Used to rank coins within each tier.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

import requests

import database as db

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
REQUEST_TIMEOUT = 15


def _fetch_top_coins(limit: int = 50) -> list[dict]:
    """Fetch top coins by market cap from CoinGecko."""
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": "true",
                "price_change_percentage": "24h,7d,30d",
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.error("CoinGecko top coins fetch failed: %s", exc)
        return []


def _compute_momentum_score(coin: dict) -> int:
    """
    Score 0-100 based on multiple momentum signals.
    Higher = stronger upward momentum.
    """
    score = 50  # neutral baseline

    # 7-day price change (-20 to +20 points)
    change_7d = coin.get("price_change_percentage_7d_in_currency") or 0
    score += max(-20, min(20, change_7d * 2))

    # 30-day price change (-15 to +15 points)
    change_30d = coin.get("price_change_percentage_30d_in_currency") or 0
    score += max(-15, min(15, change_30d * 0.5))

    # Volume/market cap ratio (liquidity signal, 0 to +10)
    vol = coin.get("total_volume") or 0
    cap = coin.get("market_cap") or 1
    vol_ratio = vol / cap * 100
    if vol_ratio > 10:
        score += 10
    elif vol_ratio > 5:
        score += 5

    # Distance from ATH (closer = stronger, 0 to +5)
    ath_change = coin.get("ath_change_percentage") or -100
    if ath_change > -10:
        score += 5
    elif ath_change > -30:
        score += 3

    return max(0, min(100, int(score)))


def _classify_tier(market_cap: float) -> str:
    if market_cap >= 50_000_000_000:
        return "tier1"
    elif market_cap >= 1_000_000_000:
        return "tier2"
    return "tier3"


def run_screening(limit: int = 50) -> list[dict]:
    """
    Run a full coin screening. Fetches top N coins, scores them,
    stores results in MySQL, and returns the ranked list.
    """
    coins = _fetch_top_coins(limit)
    if not coins:
        return []

    today = date.today().isoformat()
    results = []

    for coin in coins:
        cap = coin.get("market_cap") or 0
        momentum = _compute_momentum_score(coin)
        tier = _classify_tier(cap)

        sparkline = coin.get("sparkline_in_7d", {}).get("price", [])
        # Downsample sparkline to 24 points for storage
        if len(sparkline) > 24:
            step = len(sparkline) // 24
            sparkline = [round(sparkline[i], 2) for i in range(0, len(sparkline), step)][:24]

        row = {
            "scan_date": today,
            "coin_id": coin.get("id", ""),
            "symbol": (coin.get("symbol") or "").upper(),
            "name": coin.get("name", ""),
            "price_usd": coin.get("current_price") or 0,
            "market_cap": cap,
            "volume_24h": coin.get("total_volume") or 0,
            "change_24h_pct": coin.get("price_change_percentage_24h") or 0,
            "change_7d_pct": coin.get("price_change_percentage_7d_in_currency") or 0,
            "change_30d_pct": coin.get("price_change_percentage_30d_in_currency") or 0,
            "momentum_score": momentum,
            "risk_tier": tier,
            "category": "",
            "sparkline_7d": sparkline,
        }
        results.append(row)

    # Store in MySQL (upsert by date + coin_id)
    for r in results:
        try:
            db._execute(
                """INSERT INTO coin_screenings
                   (scan_date, coin_id, symbol, name, price_usd, market_cap, volume_24h,
                    change_24h_pct, change_7d_pct, change_30d_pct, momentum_score,
                    risk_tier, category, sparkline_7d)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON DUPLICATE KEY UPDATE
                     price_usd=VALUES(price_usd), market_cap=VALUES(market_cap),
                     volume_24h=VALUES(volume_24h), change_24h_pct=VALUES(change_24h_pct),
                     change_7d_pct=VALUES(change_7d_pct), change_30d_pct=VALUES(change_30d_pct),
                     momentum_score=VALUES(momentum_score), sparkline_7d=VALUES(sparkline_7d)""",
                (
                    r["scan_date"], r["coin_id"], r["symbol"], r["name"],
                    r["price_usd"], r["market_cap"], r["volume_24h"],
                    r["change_24h_pct"], r["change_7d_pct"], r["change_30d_pct"],
                    r["momentum_score"], r["risk_tier"], r["category"],
                    json.dumps(r["sparkline_7d"]),
                ),
            )
        except Exception as exc:
            logger.debug("Screening insert failed for %s: %s", r["symbol"], exc)

    # Sort by momentum score descending
    results.sort(key=lambda x: x["momentum_score"], reverse=True)

    logger.info("Coin screening: %d coins scanned, top: %s (%d)",
                len(results), results[0]["symbol"] if results else "none",
                results[0]["momentum_score"] if results else 0)

    return results


def get_latest_screening() -> list[dict]:
    """Get the most recent screening results from MySQL."""
    rows = db._execute(
        """SELECT * FROM coin_screenings
           WHERE scan_date = (SELECT MAX(scan_date) FROM coin_screenings)
           ORDER BY momentum_score DESC""",
        fetch="all",
    )
    for r in rows:
        r["scan_date"] = str(r["scan_date"])
        r["created_at"] = str(r["created_at"])
        if isinstance(r.get("sparkline_7d"), str):
            r["sparkline_7d"] = json.loads(r["sparkline_7d"])
    return rows


def format_screening_telegram(results: list[dict], top_n: int = 10) -> str:
    """Format top N screening results for Telegram."""
    if not results:
        return "No screening data available. Run /screen first."

    tier_emoji = {"tier1": "🔵", "tier2": "🟡", "tier3": "🔴"}
    lines = [f"📊 *Coin Screening — Top {top_n}*\n"]

    for i, r in enumerate(results[:top_n], 1):
        te = tier_emoji.get(r["risk_tier"], "⚪")
        change_7d = r.get("change_7d_pct") or 0
        change_color = "+" if change_7d >= 0 else ""
        cap = r["market_cap"]
        cap_str = f"${cap/1e9:.1f}B" if cap >= 1e9 else f"${cap/1e6:.0f}M"

        lines.append(
            f"{i}\\. {te} *{r['symbol']}* — Score: {r['momentum_score']}/100\n"
            f"   ${r['price_usd']:,.2f} \\| 7d: {change_color}{change_7d:.1f}% \\| Cap: {cap_str}"
        )

    lines.append(f"\n_Tiers: 🔵 >$50B  🟡 $1\\-50B  🔴 <$1B_")
    return "\n".join(lines)
