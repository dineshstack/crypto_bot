"""
Investment thesis generator — deep analysis of any cryptocurrency.

Unlike /research (which scores new coins), /thesis generates a complete
investment thesis that an advisor can share with clients. Includes:
  - Fundamental analysis
  - Technical setup (current chart position)
  - Risk factors with probability assessment
  - Entry/exit levels
  - Position size suggestion based on portfolio size
  - Timeframe and conviction level

Uses Claude Opus for the deepest analysis quality.
"""
from __future__ import annotations

import json
import logging
from datetime import date

import anthropic
import requests

import config
import database as db

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def _fetch_coin_data(query: str) -> dict | None:
    """Fetch comprehensive coin data from CoinGecko."""
    # Search for coin
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/search",
            params={"query": query},
            timeout=10,
        )
        if not r.ok:
            return None
        results = r.json().get("coins", [])
        if not results:
            return None
        coin_id = results[0]["id"]
    except Exception:
        return None

    # Fetch detailed data
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "true",
            },
            timeout=15,
        )
        if not r.ok:
            return None
        data = r.json()
    except Exception:
        return None

    md = data.get("market_data", {})

    return {
        "id": coin_id,
        "symbol": (data.get("symbol") or "").upper(),
        "name": data.get("name", ""),
        "description": (data.get("description", {}).get("en") or "")[:2000],
        "categories": data.get("categories", []),
        "market_cap_rank": data.get("market_cap_rank"),
        "price_usd": md.get("current_price", {}).get("usd"),
        "market_cap": md.get("market_cap", {}).get("usd"),
        "volume_24h": md.get("total_volume", {}).get("usd"),
        "change_24h": md.get("price_change_percentage_24h"),
        "change_7d": md.get("price_change_percentage_7d"),
        "change_30d": md.get("price_change_percentage_30d"),
        "change_1y": md.get("price_change_percentage_1y"),
        "ath": md.get("ath", {}).get("usd"),
        "ath_change_pct": md.get("ath_change_percentage", {}).get("usd"),
        "ath_date": md.get("ath_date", {}).get("usd"),
        "circulating_supply": md.get("circulating_supply"),
        "total_supply": md.get("total_supply"),
        "max_supply": md.get("max_supply"),
        "github_commits_4w": (data.get("developer_data") or {}).get("commit_count_4_weeks"),
        "github_stars": (data.get("developer_data") or {}).get("stars"),
        "twitter_followers": (data.get("community_data") or {}).get("twitter_followers"),
        "reddit_subscribers": (data.get("community_data") or {}).get("reddit_subscribers"),
        "genesis_date": data.get("genesis_date"),
        "homepage": (data.get("links", {}).get("homepage") or [""])[0],
    }


def generate_thesis(query: str, portfolio_size: float = 10000) -> dict | None:
    """
    Generate a complete investment thesis for a cryptocurrency.
    Returns structured thesis or None if coin not found.
    """
    coin = _fetch_coin_data(query)
    if not coin:
        return None

    def _n(v, d=0):
        return v if v is not None else d

    cap = _n(coin["market_cap"])
    cap_str = f"${cap/1e9:.1f}B" if cap >= 1e9 else f"${cap/1e6:.0f}M" if cap >= 1e6 else f"${cap/1e3:.0f}K"

    circ = _n(coin["circulating_supply"])
    total = _n(coin["total_supply"])
    max_s = _n(coin["max_supply"])
    dilution = f"{circ/total*100:.0f}% circulating" if total else "unknown dilution"

    prompt = f"""Generate a professional investment thesis for {coin['name']} ({coin['symbol']}) that a crypto investment advisor can share with clients.

CLIENT CONTEXT: The client has a ${portfolio_size:,.0f} crypto portfolio and is asking whether to invest in {coin['symbol']}.

COIN DATA:
  Name:           {coin['name']} ({coin['symbol']})
  Rank:           #{coin.get('market_cap_rank') or '?'}
  Categories:     {', '.join(coin.get('categories', [])[:3]) or 'Unknown'}
  Launch:         {coin.get('genesis_date') or 'Unknown'}
  Price:          ${_n(coin['price_usd']):,.6f}
  Market Cap:     {cap_str}
  24h Volume:     ${_n(coin['volume_24h']):,.0f}

PRICE PERFORMANCE:
  24h:   {_n(coin['change_24h']):+.1f}%
  7d:    {_n(coin['change_7d']):+.1f}%
  30d:   {_n(coin['change_30d']):+.1f}%
  1y:    {_n(coin['change_1y']):+.1f}%
  ATH:   ${_n(coin['ath']):,.4f} ({_n(coin['ath_change_pct']):.0f}% from ATH)

TOKENOMICS:
  Circulating: {circ:,.0f} ({dilution})
  Total:       {total:,.0f}
  Max:         {max_s:,.0f} ({'capped' if max_s else 'uncapped/inflationary'})

DEVELOPMENT:
  GitHub commits (4w): {_n(coin['github_commits_4w'])}
  Stars: {_n(coin['github_stars'])}

COMMUNITY:
  Twitter: {_n(coin['twitter_followers']):,}
  Reddit: {_n(coin['reddit_subscribers']):,}

DESCRIPTION:
{coin.get('description', 'No description available.')[:1500]}

Write a structured investment thesis with these exact sections:

1. THESIS SUMMARY (2-3 sentences — the core investment case, for or against)

2. FUNDAMENTAL ANALYSIS
   - What problem does it solve? Is there product-market fit?
   - Competitive positioning vs alternatives
   - Team and development activity assessment
   - Tokenomics evaluation (supply dynamics, inflation)

3. TECHNICAL SETUP
   - Current price relative to ATH and key levels
   - Short-term and medium-term trend direction
   - Key support and resistance levels to watch

4. RISK FACTORS
   List the top 3-5 risks, each with estimated probability (Low/Medium/High)

5. INVESTMENT RECOMMENDATION
   - Verdict: BUY / ACCUMULATE / HOLD / AVOID
   - Conviction level: HIGH / MEDIUM / LOW
   - Suggested allocation: X% of portfolio (based on ${portfolio_size:,.0f} portfolio)
   - Suggested entry zone: $X - $Y
   - Stop-loss level: $X (% below entry)
   - Take-profit targets: Target 1 ($X), Target 2 ($X)
   - Time horizon: X months

6. WHAT WOULD CHANGE THIS THESIS
   - Bullish catalyst (what would make you more bullish)
   - Bearish catalyst (what would make you sell/avoid)

Write in professional language suitable for client communication. Be specific with numbers."""

    try:
        resp = _client.beta.messages.create(
            model=config.CLAUDE_DEEP_MODEL,
            max_tokens=3000,
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": config.CLAUDE_DEEP_FALLBACK}],
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("thesis request declined by safety filters")
        # Fable always emits thinking blocks first — take the first text block
        thesis_text = next(
            (b.text for b in resp.content if getattr(b, "type", None) == "text"),
            "",
        ).strip()

        db.log_claude_call(
            cycle_id=f"thesis_{coin['symbol']}_{date.today().isoformat()}",
            agent="thesis_generator",
            model=resp.model,
            prompt=prompt[:5000],
            response=thesis_text[:5000],
            tokens_in=resp.usage.input_tokens if resp.usage else 0,
            tokens_out=resp.usage.output_tokens if resp.usage else 0,
        )
    except Exception as exc:
        logger.error("Thesis generation failed: %s", exc)
        return {"error": str(exc)}

    return {
        "symbol": coin["symbol"],
        "name": coin["name"],
        "price": coin["price_usd"],
        "market_cap": coin["market_cap"],
        "thesis": thesis_text,
        "coin_data": coin,
        "portfolio_size": portfolio_size,
        "generated_at": date.today().isoformat(),
    }
