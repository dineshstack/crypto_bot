"""
New coin research module.

Discovers recently listed cryptocurrencies and evaluates their investment
potential using CoinGecko (free API) data + Claude Opus deep analysis.

HOW IT WORKS:
  1. /newcoins  → fetches CoinGecko "recently added" list + trending
  2. Quick filter: removes coins below market cap / volume thresholds
  3. For each candidate, fetches full data: market metrics, developer stats
     (GitHub commits, stars), community data (Twitter, Reddit), description
  4. Sends all structured data to Claude Opus (adaptive thinking) for a
     scored investment report (0-100) with risks, opportunities, verdict
  5. Saves report to MySQL; top candidates can be added to watchlist

SCORING DIMENSIONS (each 0–20, total 0–100):
  Team         — transparency, known backgrounds, team size indicators
  Technology   — GitHub activity, audit status, code quality signals
  Market       — market cap size, liquidity, exchange listings quality
  Tokenomics   — supply cap, distribution fairness, vesting signals
  Use Case     — problem clarity, market size, competitive advantage, timing

VERDICT:
  buy    (score ≥ 70) — strong conviction, recommend small position
  watch  (score 45–69) — promising but wait for confirmation
  avoid  (score < 45)  — too risky or unclear fundamentals
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
import requests
import anthropic
import claude_deep
import config

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

COINGECKO_BASE   = "https://api.coingecko.com/api/v3"
GITHUB_API_BASE  = "https://api.github.com"
REQUEST_TIMEOUT  = 12
RATE_LIMIT_DELAY = 1.5      # CoinGecko free tier: ~30 req/min

# Screening thresholds — below these the coin is auto-rejected
MIN_MARKET_CAP   = 1_000_000    # $1M minimum (filters true micro-caps)
MIN_VOLUME_24H   = 100_000      # $100K daily volume (filters ghost markets)
MAX_NEW_COINS    = 6            # Max coins fetched per scan
MAX_TO_RESEARCH  = 3            # Max deep-researched per scan (Opus is expensive)


# ── CoinGecko API helpers ────────────────────────────────────────────────────

def _cg_headers() -> dict:
    h = {"accept": "application/json"}
    if config.COINGECKO_API_KEY:
        h["x-cg-demo-api-key"] = config.COINGECKO_API_KEY
    return h


def _cg_get(path: str, params: dict = None) -> dict | list | None:
    """GET a CoinGecko endpoint. Returns parsed JSON or None on failure."""
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/{path}",
            params=params or {},
            headers=_cg_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("CoinGecko %s failed: %s", path, exc)
        return None


def _github_repo_stats(repo_url: str) -> dict:
    """Fetch GitHub repo stats from a URL like https://github.com/owner/repo."""
    try:
        match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
        if not match:
            return {}
        slug = match.group(1).rstrip("/").replace(".git", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if hasattr(config, "GITHUB_TOKEN") and config.GITHUB_TOKEN:
            headers["Authorization"] = f"token {config.GITHUB_TOKEN}"
        r = requests.get(
            f"{GITHUB_API_BASE}/repos/{slug}",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()
        return {
            "stars":    d.get("stargazers_count", 0),
            "forks":    d.get("forks_count", 0),
            "watchers": d.get("watchers_count", 0),
            "open_issues": d.get("open_issues_count", 0),
            "last_push": d.get("pushed_at", ""),
            "language": d.get("language", ""),
        }
    except Exception as exc:
        logger.debug("GitHub stats failed for %s: %s", repo_url, exc)
        return {}


# ── Coin discovery ─────────────────────────────────────────────────────────

def get_new_coins() -> list[dict]:
    """
    Fetch recently added coins from CoinGecko and trending coins as fallback.
    Returns a list of dicts with id, symbol, name, market_cap, volume, age_days.
    """
    candidates: list[dict] = []
    seen: set[str] = set()

    # 1. Recently added list (primary source)
    new_raw = _cg_get("coins/list/new")
    if new_raw and isinstance(new_raw, list):
        ids = [c["id"] for c in new_raw[:20] if c.get("id")]
        time.sleep(RATE_LIMIT_DELAY)

        # Batch market data fetch
        market_data = _cg_get("coins/markets", {
            "vs_currency": "usd",
            "ids":         ",".join(ids),
            "order":       "market_cap_desc",
            "per_page":    20,
            "sparkline":   "false",
        })
        if market_data:
            for m in market_data:
                cap = m.get("market_cap") or 0
                vol = m.get("total_volume") or 0
                if cap >= MIN_MARKET_CAP and vol >= MIN_VOLUME_24H:
                    cid = m["id"]
                    if cid not in seen:
                        seen.add(cid)
                        genesis = m.get("atl_date") or ""
                        candidates.append({
                            "id":          cid,
                            "symbol":      m.get("symbol", "").upper(),
                            "name":        m.get("name", ""),
                            "market_cap":  cap,
                            "volume_24h":  vol,
                            "price":       m.get("current_price", 0),
                            "change_7d":   m.get("price_change_percentage_7d_in_currency", 0),
                            "rank":        m.get("market_cap_rank"),
                            "image":       m.get("image", ""),
                            "source":      "new_listing",
                        })

    time.sleep(RATE_LIMIT_DELAY)

    # 2. Trending coins (some are new breakout projects)
    trending = _cg_get("search/trending")
    if trending and "coins" in trending:
        trend_ids = [c["item"]["id"] for c in trending["coins"] if c.get("item", {}).get("id")]
        trend_ids = [t for t in trend_ids if t not in seen]
        if trend_ids:
            time.sleep(RATE_LIMIT_DELAY)
            mkt = _cg_get("coins/markets", {
                "vs_currency": "usd",
                "ids":         ",".join(trend_ids),
                "order":       "market_cap_desc",
                "per_page":    15,
                "sparkline":   "false",
            })
            if mkt:
                for m in mkt:
                    cap = m.get("market_cap") or 0
                    vol = m.get("total_volume") or 0
                    rank = m.get("market_cap_rank") or 9999
                    # Trending + under rank 500 + decent metrics = interesting
                    if cap >= MIN_MARKET_CAP and vol >= MIN_VOLUME_24H and rank < 500:
                        cid = m["id"]
                        if cid not in seen:
                            seen.add(cid)
                            candidates.append({
                                "id":         cid,
                                "symbol":     m.get("symbol", "").upper(),
                                "name":       m.get("name", ""),
                                "market_cap": cap,
                                "volume_24h": vol,
                                "price":      m.get("current_price", 0),
                                "change_7d":  m.get("price_change_percentage_7d_in_currency", 0),
                                "rank":       rank,
                                "image":      m.get("image", ""),
                                "source":     "trending",
                            })

    # Sort by volume/market_cap ratio (liquidity proxy) descending
    for c in candidates:
        c["liquidity_ratio"] = c["volume_24h"] / c["market_cap"] if c["market_cap"] else 0

    candidates.sort(key=lambda x: (x["liquidity_ratio"]), reverse=True)
    return candidates[:MAX_NEW_COINS]


# ── Detailed data fetch ────────────────────────────────────────────────────

def fetch_coin_detail(coin_id: str) -> dict | None:
    """
    Fetch complete CoinGecko data for one coin: market, developer, community,
    description, links. Also enriches with direct GitHub stats if repo is found.
    """
    time.sleep(RATE_LIMIT_DELAY)
    data = _cg_get(f"coins/{coin_id}", {
        "localization":    "false",
        "tickers":         "false",
        "market_data":     "true",
        "community_data":  "true",
        "developer_data":  "true",
        "sparkline":       "false",
    })
    if not data:
        return None

    # Extract and flatten key fields
    md = data.get("market_data", {})
    cd = data.get("community_data", {})
    dd = data.get("developer_data", {})
    links = data.get("links", {})

    github_repos = links.get("repos_url", {}).get("github", [])
    gh_stats = {}
    if github_repos:
        time.sleep(RATE_LIMIT_DELAY)
        gh_stats = _github_repo_stats(github_repos[0])

    description = (data.get("description", {}).get("en", "") or "")[:2000]
    categories  = data.get("categories", [])

    return {
        # Identity
        "id":               data.get("id"),
        "symbol":           data.get("symbol", "").upper(),
        "name":             data.get("name"),
        "categories":       categories,
        "description":      description,
        "genesis_date":     data.get("genesis_date"),
        "hashing_algorithm": data.get("hashing_algorithm"),
        "block_time":       data.get("block_time_in_minutes"),

        # Links
        "homepage":         (links.get("homepage") or [""])[0],
        "whitepaper":       links.get("whitepaper") or "",
        "twitter":          links.get("twitter_screen_name") or "",
        "reddit":           links.get("subreddit_url") or "",
        "github_repos":     github_repos,

        # Market data
        "price_usd":        (md.get("current_price") or {}).get("usd", 0),
        "market_cap_usd":   (md.get("market_cap") or {}).get("usd", 0),
        "volume_24h_usd":   (md.get("total_volume") or {}).get("usd", 0),
        "ath_usd":          (md.get("ath") or {}).get("usd", 0),
        "ath_change_pct":   (md.get("ath_change_percentage") or {}).get("usd", 0),
        "change_24h_pct":   (md.get("price_change_percentage_24h") or 0),
        "change_7d_pct":    (md.get("price_change_percentage_7d") or 0),
        "change_30d_pct":   (md.get("price_change_percentage_30d") or 0),
        "total_supply":     md.get("total_supply"),
        "max_supply":       md.get("max_supply"),
        "circulating_supply": md.get("circulating_supply"),
        "market_cap_rank":  data.get("market_cap_rank"),

        # Community
        "twitter_followers":  cd.get("twitter_followers", 0),
        "reddit_subscribers": cd.get("reddit_subscribers", 0),
        "telegram_users":     cd.get("telegram_channel_user_count", 0),

        # Developer (CoinGecko aggregated)
        "github_stars":         dd.get("stars", 0),
        "github_forks":         dd.get("forks", 0),
        "github_subscribers":   dd.get("subscribers", 0),
        "github_issues_closed": dd.get("closed_issues", 0),
        "github_prs_merged":    dd.get("pull_requests_merged", 0),
        "github_commits_4w":    dd.get("commit_count_4_weeks", 0),
        "github_contributors":  dd.get("pull_request_contributors", 0),

        # Direct GitHub (may be richer than CoinGecko's aggregation)
        "github_direct": gh_stats,

        # Coingecko scores
        "coingecko_score":   data.get("coingecko_score", 0),
        "developer_score":   data.get("developer_score", 0),
        "community_score":   data.get("community_score", 0),
        "liquidity_score":   data.get("liquidity_score", 0),
    }


# ── Claude Opus investment analysis ──────────────────────────────────────────

_RESEARCH_SYSTEM = """You are a cryptocurrency investment analyst for a small retail portfolio.
Your job: evaluate a new/trending coin and output a structured investment report in JSON.

SCORING FRAMEWORK (each dimension 0–20, total 0–100):
  team_score       — transparency, known names, team size, track record signals
  technology_score — GitHub activity, audit status, code quality, unique tech
  market_score     — market cap size (sweet spot $5M-$500M), liquidity, exchange quality
  tokenomics_score — supply cap, circulating%, fair distribution, no suspicious concentration
  usecase_score    — clear problem solved, market timing, competitive moat

VERDICT THRESHOLDS:
  buy   (score ≥ 70) — high conviction, recommend small position
  watch (45–69)      — promising but needs more evidence
  avoid (< 45)       — too risky, unclear, or likely scam

RED FLAGS (strongly penalise):
  - Anonymous team with no GitHub contributors
  - Whitepaper is missing or plagiarised
  - Max supply already 95%+ circulating (pre-mine dump risk)
  - No real GitHub activity (0 commits in 4 weeks AND no stars)
  - Listed only on obscure DEXs with < $100K volume
  - Name or description copies an existing popular project

OUTPUT — valid JSON only, no other text:
{
  "investment_score": 72,
  "team_score": 15,
  "technology_score": 18,
  "market_score": 14,
  "tokenomics_score": 13,
  "usecase_score": 12,
  "verdict": "watch",
  "suggested_usd": 25,
  "hold_months": 12,
  "risks": ["anonymous core team", "low liquidity on DEX only"],
  "opportunities": ["first-mover in AI inference market", "active GitHub community"],
  "summary": "Two to three sentence investment thesis or rejection reason."
}"""


def research_coin(detail: dict) -> dict | None:
    """
    Send full coin data to Claude Opus for investment analysis.
    Returns the parsed JSON report dict, or None on failure.
    """
    def _n(val, default=0):
        """Coerce None to default for safe formatting."""
        return val if val is not None else default

    circ = _n(detail.get("circulating_supply"))
    total = _n(detail.get("total_supply"))
    max_s = _n(detail.get("max_supply"))
    circ_pct = round(circ / total * 100, 1) if total else None
    liq_ratio = (
        _n(detail.get("volume_24h_usd")) / _n(detail.get("market_cap_usd"), 1) * 100
        if _n(detail.get("market_cap_usd")) else 0
    )

    categories_str = ", ".join(detail.get("categories", [])) or "Unknown"

    prompt = f"""Research and score this cryptocurrency as a small investment opportunity:

═══ IDENTITY ════════════════════════════════════════════
Name:        {detail['name']} ({detail['symbol']})
Categories:  {categories_str}
Launch date: {detail.get('genesis_date') or 'Unknown'}
Consensus:   {detail.get('hashing_algorithm') or 'Unknown'}

═══ DESCRIPTION (max 2000 chars from CoinGecko) ════════
{detail.get('description') or 'No description available.'}

═══ MARKET METRICS ══════════════════════════════════════
Price:           ${_n(detail.get('price_usd')):,.6f}
Market Cap:      ${_n(detail.get('market_cap_usd')):,.0f}  (rank #{detail.get('market_cap_rank') or '?'})
Volume 24h:      ${_n(detail.get('volume_24h_usd')):,.0f}
Volume/Cap:      {liq_ratio:.1f}%  (>5% = good liquidity)
ATH:             ${_n(detail.get('ath_usd')):,.4f}  ({_n(detail.get('ath_change_pct')):.0f}% from ATH)
24h change:      {_n(detail.get('change_24h_pct')):+.1f}%
7d change:       {_n(detail.get('change_7d_pct')):+.1f}%
30d change:      {_n(detail.get('change_30d_pct')):+.1f}%

═══ TOKENOMICS ══════════════════════════════════════════
Total supply:     {total:,.0f}
Max supply:       {max_s:,.0f}  ({'infinite / inflationary' if not max_s else 'capped'})
Circulating:      {circ:,.0f}  ({f'{circ_pct}%' if circ_pct else 'unknown'} of total)

═══ DEVELOPER ACTIVITY ══════════════════════════════════
GitHub repos:     {', '.join(detail.get('github_repos', [])) or 'None found'}
Commits (4w):     {detail.get('github_commits_4w', 0)}
Stars:            {detail.get('github_stars', 0)}
Forks:            {detail.get('github_forks', 0)}
Contributors:     {detail.get('github_contributors', 0)}
PRs merged:       {detail.get('github_prs_merged', 0)}
Issues closed:    {detail.get('github_issues_closed', 0)}
Direct GH stats:  {json.dumps(detail.get('github_direct', {})) or 'N/A'}

═══ COMMUNITY ════════════════════════════════════════════
Twitter followers: {_n(detail.get('twitter_followers')):,}  (@{detail.get('twitter') or 'none'})
Reddit subscribers:{_n(detail.get('reddit_subscribers')):,}
Telegram users:    {_n(detail.get('telegram_users')):,}

═══ CoinGecko Scores (0–100 each) ═══════════════════════
Developer score:   {_n(detail.get('developer_score')):.1f}
Community score:   {_n(detail.get('community_score')):.1f}
Liquidity score:   {_n(detail.get('liquidity_score')):.1f}

═══ LINKS ════════════════════════════════════════════════
Website:    {detail.get('homepage') or 'N/A'}
Whitepaper: {detail.get('whitepaper') or 'N/A'}

Evaluate all dimensions and return the JSON investment report."""

    try:
        response = claude_deep.call_deep_model(
            _client, max_tokens=1024, thinking=True,
            system=_RESEARCH_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("research request declined by safety filters")
        raw = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            "",
        ).strip()

        try:
            report = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            report = json.loads(m.group()) if m else {}

        # Attach the raw coin data into the report for MySQL storage
        report["coin_id"]      = detail["id"]
        report["symbol"]       = detail["symbol"]
        report["name"]         = detail["name"]
        report["price_usd"]    = detail.get("price_usd", 0)
        report["market_cap"]   = detail.get("market_cap_usd", 0)
        report["volume_24h"]   = detail.get("volume_24h_usd", 0)
        report["change_7d"]    = detail.get("change_7d_pct", 0)
        report["commits_4w"]   = detail.get("github_commits_4w", 0)
        report["twitter_flw"]  = detail.get("twitter_followers", 0)
        report["raw_data"]     = detail

        return report

    except Exception as exc:
        logger.error("Claude research failed for %s: %s", detail.get("id"), exc)
        return None


# ── MySQL persistence ───────────────────────────────────────────────────

def save_research(report: dict) -> str:
    """Save a research report to MySQL. Returns the row ID."""
    import database as db
    try:
        row_id = db.insert_coin_research({
            "coin_id":          report.get("coin_id"),
            "symbol":           report.get("symbol"),
            "name":             report.get("name"),
            "investment_score": report.get("investment_score"),
            "team_score":       report.get("team_score"),
            "technology_score": report.get("technology_score"),
            "market_score":     report.get("market_score"),
            "tokenomics_score": report.get("tokenomics_score"),
            "usecase_score":    report.get("usecase_score"),
            "verdict":          report.get("verdict"),
            "suggested_usd":    report.get("suggested_usd"),
            "hold_months":      report.get("hold_months"),
            "risks":            report.get("risks", []),
            "opportunities":    report.get("opportunities", []),
            "summary":          report.get("summary"),
            "price_usd":        report.get("price_usd"),
            "market_cap_usd":   report.get("market_cap"),
            "volume_24h_usd":   report.get("volume_24h"),
            "price_change_7d":  report.get("change_7d"),
            "github_commits_4w": report.get("commits_4w"),
            "twitter_followers": report.get("twitter_flw"),
            "raw_data":         report.get("raw_data"),
        })
        return str(row_id)
    except Exception as exc:
        logger.error("Failed to save research: %s", exc)
        return ""


def add_to_watchlist(coin_id: str, symbol: str, name: str,
                     price: float, target_usd: float, research_id: str):
    import database as db
    try:
        db.upsert_watchlist({
            "coin_id":     coin_id,
            "symbol":      symbol,
            "name":        name,
            "entry_price": price,
            "target_usd":  target_usd,
            "research_id": int(research_id) if research_id else None,
        })
        if research_id:
            db.update_research_watchlist(int(research_id))
    except Exception as exc:
        logger.error("Failed to add watchlist: %s", exc)


def get_watchlist() -> list[dict]:
    import database as db
    try:
        return db.get_watchlist()
    except Exception as exc:
        logger.error("Failed to get watchlist: %s", exc)
        return []


def get_recent_research(limit: int = 10) -> list[dict]:
    import database as db
    try:
        return db.get_recent_research(limit)
    except Exception as exc:
        logger.error("Failed to get recent research: %s", exc)
        return []


def find_coin_id(symbol_or_name: str) -> str | None:
    """Search CoinGecko for a coin by symbol or name. Returns its ID."""
    result = _cg_get("search", {"query": symbol_or_name})
    if not result:
        return None
    coins = result.get("coins", [])
    if not coins:
        return None
    # Prefer exact symbol match, then exact name match, else first result
    q = symbol_or_name.lower()
    for c in coins:
        if c.get("symbol", "").lower() == q:
            return c["id"]
    for c in coins:
        if c.get("name", "").lower() == q:
            return c["id"]
    return coins[0]["id"]


# ── High-level orchestrators ───────────────────────────────────────────────

def scan_new_coins() -> list[dict]:
    """
    Full scan: discover new coins → filter → deep research top N.
    Returns list of report dicts sorted by investment_score descending.
    """
    logger.info("Starting new coin scan…")
    candidates = get_new_coins()
    logger.info("Found %d candidates after screening", len(candidates))

    reports = []
    for i, coin in enumerate(candidates[:MAX_TO_RESEARCH]):
        logger.info("Researching %d/%d: %s (%s)",
                    i + 1, min(len(candidates), MAX_TO_RESEARCH),
                    coin["name"], coin["symbol"])
        detail = fetch_coin_detail(coin["id"])
        if not detail:
            continue
        report = research_coin(detail)
        if report:
            rid = save_research(report)
            report["_db_id"] = rid
            reports.append(report)
        time.sleep(RATE_LIMIT_DELAY)

    reports.sort(key=lambda r: r.get("investment_score", 0), reverse=True)
    logger.info("Scan complete — %d reports generated", len(reports))
    return reports


def research_by_query(query: str) -> dict | None:
    """
    On-demand deep research for a specific coin (Telegram /research <symbol>).
    Searches CoinGecko by symbol/name, fetches full data, runs Claude Opus.
    """
    logger.info("On-demand research: %s", query)
    coin_id = find_coin_id(query)
    if not coin_id:
        logger.warning("Coin not found: %s", query)
        return None
    detail = fetch_coin_detail(coin_id)
    if not detail:
        return None
    report = research_coin(detail)
    if report:
        rid = save_research(report)
        report["_db_id"] = rid
    return report


# ── Telegram message formatters ────────────────────────────────────────────

def format_scan_summary(reports: list[dict]) -> str:
    """Short multi-coin scan result for Telegram."""
    if not reports:
        return "No new coins passed the quality filter this scan."

    verdict_emoji = {"buy": "✅", "watch": "👀", "avoid": "❌"}
    lines = ["🔍 *NEW COIN SCAN RESULTS*\n"]
    for i, r in enumerate(reports, 1):
        cap = r.get("market_cap") or 0
        cap_str = f"${cap/1e6:.1f}M" if cap >= 1e6 else f"${cap/1e3:.0f}K"
        ve = verdict_emoji.get(r.get("verdict", "avoid"), "❓")
        lines.append(
            f"{i}\\. {ve} *{r['symbol']}* — {r['name']}\n"
            f"   Score: {r.get('investment_score', 0)}/100  |  Cap: {cap_str}\n"
            f"   _{r.get('summary', '')[:120]}_"
        )
    lines.append("\nUse /research \\<symbol\\> for full deep\\-dive")
    return "\n".join(lines)


def format_deep_report(r: dict) -> str:
    """Full research report for Telegram (Markdown v2)."""
    verdict_emoji = {"buy": "✅ BUY", "watch": "👀 WATCH", "avoid": "❌ AVOID"}
    verdict_str   = verdict_emoji.get(r.get("verdict", "avoid"), "❓ UNKNOWN")

    cap = r.get("market_cap") or 0
    cap_str = f"${cap/1e6:.1f}M" if cap >= 1e6 else f"${cap/1e3:.0f}K"
    vol = r.get("volume_24h") or 0
    vol_str = f"${vol/1e6:.1f}M" if vol >= 1e6 else f"${vol/1e3:.0f}K"

    def esc(t: str) -> str:
        t = str(t)
        for ch in r"\_*[]()~`>#+-=|{}.!":
            t = t.replace(ch, f"\\{ch}")
        return t

    risks = "\n".join(f"  • {esc(x)}" for x in r.get("risks", []))
    opps  = "\n".join(f"  • {esc(x)}" for x in r.get("opportunities", []))

    price_str = f"{r.get('price_usd', 0):,.6f}"
    change_7d_str = f"{r.get('change_7d', 0):+.1f}"

    lines = [
        f"📊 *RESEARCH: {esc(r['symbol'])} — {esc(r['name'])}*",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"🏆 *Investment Score: {r.get('investment_score', 0)}/100*",
        f"Verdict: {esc(verdict_str)}",
        f"",
        f"*Score Breakdown:*",
        f"  Team:       {r.get('team_score', 0)}/20",
        f"  Technology: {r.get('technology_score', 0)}/20",
        f"  Market:     {r.get('market_score', 0)}/20",
        f"  Tokenomics: {r.get('tokenomics_score', 0)}/20",
        f"  Use Case:   {r.get('usecase_score', 0)}/20",
        f"",
        f"📈 *Market*",
        f"  Price: ${esc(price_str)}  |  Cap: {esc(cap_str)}",
        f"  Volume 24h: {esc(vol_str)}  |  7d: {esc(change_7d_str)}%",
        f"",
    ]
    if risks:
        lines += ["⚠️ *Risks:*", risks, ""]
    if opps:
        lines += ["🚀 *Opportunities:*", opps, ""]
    lines += [
        f"💡 *Recommendation:*",
        f"  Position size: ${esc(str(r.get('suggested_usd', 0)))}",
        f"  Hold period:   {esc(str(r.get('hold_months', 0)))} months",
        f"",
        f"_{esc(r.get('summary', ''))}_",
    ]
    if r.get("_db_id"):
        lines.append(
            f"\nUse /watchlist add {esc(r['symbol'])} to track this coin"
        )
    return "\n".join(lines)
