"""
Multi-agent Claude analysis pipeline.

Three specialized agents replace the single mega-prompt:
  1. Market Analyst  (Haiku) — technicals + derivatives → market assessment
  2. Sentiment Analyst (Haiku) — news + social + on-chain → sentiment assessment
  3. Decision Maker  (Haiku) — both assessments + portfolio + lessons → final trade

Agents 1 & 2 run in parallel (independent). Agent 3 synthesizes.
This produces better decisions because each agent focuses on its domain.
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic
import config
import database as db
import news_fetcher
import social_sentiment
import onchain_data
import cross_asset
import options_data
import whale_monitor
import ml_signal
import ws_stream
import grid_dca
import onchain_macro

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
_current_cycle_id: str = ""   # set per analysis cycle for log grouping


def _call_claude(agent_name: str, system: str, prompt: str,
                 max_tokens: int = 300, defaults: dict = None) -> dict:
    """Call Claude API with full logging to MySQL."""
    import time as _time
    import uuid

    start = _time.time()
    try:
        resp = _client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        duration = int((_time.time() - start) * 1000)

        db.log_claude_call(
            cycle_id=_current_cycle_id or str(uuid.uuid4())[:8],
            agent=agent_name,
            model=config.CLAUDE_MODEL,
            prompt=f"[SYSTEM]\n{system[:2000]}\n\n[USER]\n{prompt}",
            response=raw,
            tokens_in=resp.usage.input_tokens if resp.usage else 0,
            tokens_out=resp.usage.output_tokens if resp.usage else 0,
            duration_ms=duration,
        )

        return _parse_json(raw, defaults or {})
    except Exception as exc:
        duration = int((_time.time() - start) * 1000)
        db.log_claude_call(
            cycle_id=_current_cycle_id or "error",
            agent=agent_name,
            model=config.CLAUDE_MODEL,
            prompt=f"[SYSTEM]\n{system[:500]}\n\n[USER]\n{prompt[:500]}",
            response=f"ERROR: {exc}",
            duration_ms=duration,
        )
        logger.error("%s agent failed: %s", agent_name, exc)
        return defaults or {}


# ── Agent 1: Market Analyst ───────────────────────────────────────────────────

_MARKET_SYSTEM = """You are a Bitcoin technical and derivatives analyst.
Analyse the market data and output a JSON assessment. Focus ONLY on technicals and derivatives — ignore news.

INDICATOR RULES:
- RSI < 30 = oversold (bullish); RSI > 70 = overbought (bearish); 30-70 = neutral
- StochRSI K < 20 = oversold; K > 80 = overbought; K crossing D = crossover signal
- MACD bullish + strengthening = uptrend; bearish + weakening = reversal possible
- Price above VWAP = buyers dominate; below = sellers dominate
- ATR% > 3% = high volatility; < 1% = low volatility
- OBV rising + price rising = confirmed trend; divergence = WARNING
- Ichimoku above cloud = bullish; below = bearish; inside = neutral
- Funding rate > 0.05% = overleveraged longs (crash risk)
- Funding rate < -0.01% = short squeeze potential (bullish)
- Long/Short > 2.0 = crowded long (contrarian sell); < 0.7 = crowded short (contrarian buy)

REAL-TIME STREAM RULES (if provided):
- 5m change < -1% = short-term selling pressure; > +1% = short-term buying pressure
- Buy pressure > 65% = aggressive buyers dominating; < 35% = sellers dominating
- Volume SPIKE (>3x avg) = big move imminent, confirms direction of price change
- Liquidation cascade (>$1M in 5m) = forced selling/buying, trend likely continues
- ALERT events (flash crash / breakout) = highest priority signal, override other signals

OUTPUT — valid JSON only:
{
  "trend": "bullish" | "bearish" | "neutral",
  "strength": "strong" | "moderate" | "weak",
  "volatility": "high" | "moderate" | "low",
  "derivatives_bias": "bullish" | "bearish" | "neutral",
  "key_signals": ["rsi_oversold", "macd_bullish", ...],
  "risk_level": "low" | "medium" | "high",
  "summary": "one sentence technical assessment"
}"""


def _run_market_agent(snapshot: dict, ws_context: str = "") -> dict:
    """Agent 1: Analyse technicals + derivatives + real-time stream."""
    fr = snapshot.get("funding_rate")
    fr_str = f"{fr}%" if fr is not None else "N/A"
    fr_ann = snapshot.get("funding_rate_annual")
    fr_ann_str = f"{fr_ann}% APR" if fr_ann is not None else ""
    oi_btc = snapshot.get("open_interest_btc")
    oi_usd = snapshot.get("open_interest_usd")
    oi_str = f"{oi_btc:,.0f} BTC (${oi_usd/1e9:.1f}B)" if oi_btc and oi_usd else "N/A"
    ls_ratio = snapshot.get("long_short_ratio")
    ls_str = f"{ls_ratio}" if ls_ratio else "N/A"
    long_pct = snapshot.get("long_pct")
    short_pct = snapshot.get("short_pct")
    ls_detail = f"({long_pct}% long / {short_pct}% short)" if long_pct else ""

    prompt = f"""Analyse this BTC market data:

PRICE & TREND
  Price:      ${snapshot["price"]:,}
  24h change: {snapshot["change_24h_pct"]}%
  7d change:  {snapshot["change_7d_pct"]}%
  vs SMA20:   {snapshot["vs_sma20_pct"]}%
  vs SMA50:   {snapshot["vs_sma50_pct"]}%
  Ichimoku:   {snapshot.get("ichimoku_signal", "N/A")}

MOMENTUM
  RSI(14):    {snapshot["rsi"]}
  StochRSI:   K={snapshot.get("stoch_rsi_k", "N/A")} D={snapshot.get("stoch_rsi_d", "N/A")}
  MACD:       {snapshot.get("macd_trend", "N/A")} ({snapshot.get("macd_momentum", "N/A")}), histogram={snapshot.get("macd_histogram", "N/A")}

VOLATILITY & VOLUME
  BB upper:   ${snapshot["bb_upper"]:,}  |  BB lower: ${snapshot["bb_lower"]:,}
  ATR(14):    ${snapshot.get("atr", 0):,} ({snapshot.get("atr_pct", 0)}% volatility)
  VWAP:       ${snapshot.get("vwap", 0):,} (price {snapshot.get("vs_vwap_pct", 0):+.1f}% vs VWAP)
  OBV:        {snapshot.get("obv_slope", "N/A")}
  Volume 24h: {snapshot["volume_24h_btc"]} BTC

DERIVATIVES
  Funding Rate:  {fr_str} {fr_ann_str}
  Open Interest: {oi_str}
  Long/Short:    {ls_str} {ls_detail}

Fear & Greed:    {snapshot["fear_greed"]}/100 — {snapshot["fear_greed_lbl"]} (7d trend: {snapshot.get("fear_greed_trend", "unknown")}, 7d avg: {snapshot.get("fear_greed_avg7d", "N/A")})

TIMEFRAME CONSENSUS (1h / 4h / 1d regimes from resampled data):
  1h regime:  {snapshot.get("tf_regime_1h", "N/A")}
  4h regime:  {snapshot.get("tf_regime_4h", "N/A")}
  1d regime:  {snapshot.get("tf_regime_1d", "N/A")}
  Direction:  {snapshot.get("tf_direction", "mixed")} ({snapshot.get("tf_agreement", 0)}/3 timeframes agree)
  Note: low agreement (< 2/3) means mixed signals — favour hold over directional trades"""

    if ws_context:
        prompt += f"\n\n{ws_context}"

    return _call_claude("market", _MARKET_SYSTEM, prompt, max_tokens=400, defaults={
        "trend": "neutral", "strength": "moderate", "volatility": "moderate",
        "derivatives_bias": "neutral", "key_signals": [], "risk_level": "medium",
        "summary": "Unable to assess market",
    })


# ── Agent 2: Sentiment Analyst ────────────────────────────────────────────────

_SENTIMENT_SYSTEM = """You are a Bitcoin sentiment and macro analyst.
Analyse news headlines, social media sentiment, on-chain network data, and cross-asset macro context.
Focus ONLY on sentiment and macro context — ignore price technicals.

INTERPRETATION RULES:
- ETF inflows, institutional adoption, rate cuts = bullish
- Regulatory bans, exchange hacks, rate hikes = bearish
- Gold rallying often leads BTC by hours — early bullish signal
- Reddit bullish mood + rising social volume = FOMO risk if overextended
- Reddit bearish mood + falling social volume = fear (contrarian buy if not structural)
- Hash rate rising = miner confidence; dropping = potential stress
- High mempool fees = heavy usage (bull market signal)
- Low fees = calm network

WHALE ACTIVITY RULES:
- Whale buying on exchange (large buy orders) = bullish, institutional accumulation
- Whale selling on exchange = bearish, distribution/profit-taking
- Large on-chain movements TO exchanges = sell pressure incoming
- Large on-chain movements FROM exchanges to cold storage = accumulation (bullish)
- 5+ large on-chain txs in one block = high whale activity, expect volatility

OPTIONS MARKET RULES (Deribit — often LEADS spot by 12-48 hours):
- Put/Call ratio < 0.7: more calls than puts = bullish positioning
- Put/Call ratio > 1.0: heavy put buying = fear/hedging = bearish
- DVOL < 30: low implied volatility = calm, breakout may be coming
- DVOL 30-50: normal volatility
- DVOL > 60: extreme fear, major move expected
- Max pain: price gravitates toward max pain at expiry — if price is far below max pain, expect upward pull
- Large OI at a strike = magnetic level, acts as support/resistance

CROSS-ASSET RULES (critical — BTC correlations in 2026):
- DXY (dollar) UP = bearish for BTC (inverse correlation -0.90)
- DXY DOWN = bullish for BTC
- S&P 500 UP = generally bullish for BTC (correlation +0.74)
- VIX > 25 = fear/risk-off = short-term bearish for BTC
- VIX < 15 = complacency = BTC can rally
- Gold UP + BTC DOWN = divergence warning (check if money flowing to gold as safe haven)
- US 10Y yield rising sharply = tightening conditions = bearish for risk assets including BTC

ON-CHAIN MACRO RULES (if provided):
- MVRV-Z > 7 = extreme overvaluation — historically precedes 50%+ corrections, reduce exposure
- MVRV-Z 3-7 = elevated — caution, not a sell signal alone but tighten stops
- MVRV-Z 0-3 = fair value — neutral signal
- MVRV-Z < 0 = historically undervalued — strong accumulation zone, increase exposure
- Exchange net inflow = selling pressure building; net outflow = accumulation (bullish)

OUTPUT — valid JSON only:
{
  "news_bias": "bullish" | "bearish" | "neutral",
  "social_bias": "bullish" | "bearish" | "neutral",
  "onchain_bias": "bullish" | "bearish" | "neutral",
  "macro_bias": "bullish" | "bearish" | "neutral",
  "options_bias": "bullish" | "bearish" | "neutral",
  "overall_sentiment": "bullish" | "bearish" | "neutral",
  "confidence": 0.7,
  "key_factors": ["etf_inflows", "dxy_falling", "risk_on", "put_call_bullish", ...],
  "summary": "one sentence sentiment assessment"
}"""


def _run_sentiment_agent(news_section: str, social_section: str,
                         onchain_section: str, macro_section: str = "",
                         options_section: str = "",
                         whale_section: str = "") -> dict:
    """Agent 2: Analyse news + social + on-chain + macro + options + whale activity."""
    parts = []
    if news_section:
        parts.append(f"WORLD NEWS:\n{news_section}")
    else:
        parts.append("WORLD NEWS: No recent headlines available")

    if social_section:
        parts.append(social_section)
    else:
        parts.append("SOCIAL SENTIMENT: No data available")

    if onchain_section:
        parts.append(onchain_section)
    else:
        parts.append("ON-CHAIN DATA: No data available")

    if macro_section:
        parts.append(macro_section)
    else:
        parts.append("CROSS-ASSET MACRO: No data available")

    if options_section:
        parts.append(options_section)
    else:
        parts.append("OPTIONS MARKET: No data available")

    if whale_section:
        parts.append(whale_section)
    else:
        parts.append("WHALE ACTIVITY: No data available")

    prompt = "Analyse this BTC sentiment data:\n\n" + "\n\n".join(parts)

    return _call_claude("sentiment", _SENTIMENT_SYSTEM, prompt, max_tokens=500, defaults={
        "news_bias": "neutral", "social_bias": "neutral",
        "onchain_bias": "neutral", "overall_sentiment": "neutral",
        "confidence": 0.5, "key_factors": [],
        "summary": "Unable to assess sentiment",
    })


# ── Agent 3: Decision Maker ──────────────────────────────────────────────────

_DECISION_SYSTEM = """You are a conservative Bitcoin trading decision maker for a small $200 portfolio.
You receive pre-analysed market assessment and sentiment assessment from specialist analysts.
Your job: synthesise both into ONE trading decision.

HARD RULES (non-negotiable):
- Max trade size: $15 per action
- Do NOT recommend buying if BTC allocation > 55% of portfolio
- Do NOT recommend selling if RSI > 45 (avoid panic-selling uptrends)
- Capital preservation > profit — when market and sentiment conflict, HOLD
- Do not flip between buy and sell on consecutive cycles
- If risk_level is "high" AND sentiment is not clearly bullish, HOLD
- If derivatives show overleveraged longs (funding > 0.08%), do NOT buy

DECISION FRAMEWORK:
- BUY: market trend bullish + sentiment supports + low/medium risk + allocation < 55%
- SELL: market trend bearish + sentiment confirms + portfolio has BTC to sell
- HOLD: any signal conflict, mixed sentiment, high risk, or unclear direction

OUTPUT — valid JSON only:
{
  "action": "buy" | "hold" | "sell",
  "trade_usd": 5.0,
  "confidence": 0.75,
  "risk": "low" | "medium" | "high",
  "reason": "one sentence synthesising market + sentiment",
  "signals": ["rsi_oversold", "news_bullish", "funding_neutral"]
}"""


def _run_decision_agent(market: dict, sentiment: dict,
                        portfolio: dict, price: float,
                        context: str, ml_ctx: str = "") -> dict:
    """Agent 3: Final trade decision from market + sentiment + ML assessments."""
    btc_val = portfolio["btc"] * price
    total = portfolio["usdt"] + btc_val
    alloc = round(btc_val / total * 100, 1) if total > 0 else 0

    ml_section = f"\n\n═══ {ml_ctx} ═══" if ml_ctx else ""

    prompt = f"""Make a trading decision based on these analyst reports:

═══ MARKET ANALYSIS (from Technical Analyst) ════
  Trend:       {market.get("trend", "?")} ({market.get("strength", "?")})
  Volatility:  {market.get("volatility", "?")}
  Derivatives: {market.get("derivatives_bias", "?")}
  Risk Level:  {market.get("risk_level", "?")}
  Signals:     {", ".join(market.get("key_signals", [])) or "none"}
  Assessment:  {market.get("summary", "N/A")}

═══ SENTIMENT ANALYSIS (from Sentiment Analyst) ═
  News:        {sentiment.get("news_bias", "?")}
  Social:      {sentiment.get("social_bias", "?")}
  On-chain:    {sentiment.get("onchain_bias", "?")}
  Macro:       {sentiment.get("macro_bias", "?")}
  Options:     {sentiment.get("options_bias", "?")}
  Overall:     {sentiment.get("overall_sentiment", "?")} (conf {sentiment.get("confidence", 0):.0%})
  Factors:     {", ".join(sentiment.get("key_factors", [])) or "none"}
  Assessment:  {sentiment.get("summary", "N/A")}
{ml_section}

═══ PORTFOLIO ══════════════════════════════════════
  USDT:       ${portfolio["usdt"]:.2f}
  BTC:        {portfolio["btc"]:.6f} BTC (≈${btc_val:.2f})
  Total:      ${total:.2f}
  BTC alloc:  {alloc}%
{context}
Base DCA = $5. Suggest $2–$15 depending on opportunity quality."""

    return _call_claude("decision", _DECISION_SYSTEM, prompt, max_tokens=400, defaults={
        "action": "hold", "trade_usd": 0, "confidence": 0.5,
        "risk": "high", "reason": "Unable to decide — defaulting to hold",
        "signals": [],
    })


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json(raw: str, defaults: dict) -> dict:
    """Parse JSON from Claude's response, stripping markdown fences if present."""
    # Strip ```json ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract first JSON object with regex
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse agent JSON: %s", raw[:200])
    return defaults


def _build_context_sections() -> str:
    """Load lessons and recent decision history from MySQL."""
    sections = []

    lessons = db.get_active_lessons(5)
    if lessons:
        bullet = "\n".join(f"  - {l}" for l in lessons)
        sections.append(f"LESSONS FROM PAST MISTAKES (obey these):\n{bullet}")

    history = db.get_recent_trades(5)
    if history:
        lines = []
        for h in reversed(history):
            d = h.get("decision") or {}
            outcome = f" [{h['outcome']}]" if h.get("outcome") else ""
            lines.append(
                f"  {h['created_at'][:10]}  "
                f"{h['action'].upper():4}  "
                f"${d.get('trade_usd', 0):.0f}  "
                f"@${h.get('price', 0):,}"
                f"{outcome}"
            )
        sections.append("RECENT DECISIONS (oldest→newest):\n" + "\n".join(lines))

    return ("\n\n" + "\n\n".join(sections)) if sections else ""


# ── Public API (called by main.py) ───────────────────────────────────────────

def analyze(snapshot: dict, portfolio: dict, exchange=None) -> dict:
    """
    Multi-agent analysis pipeline:
      1. Market Analyst + Sentiment Analyst run in PARALLEL
      2. ML model prediction (if trained)
      3. Decision Maker synthesises all assessments
    Returns the final parsed JSON decision dict.

    All Claude API calls are logged to claude_api_logs table for audit.
    """
    import uuid
    global _current_cycle_id
    _current_cycle_id = str(uuid.uuid4())[:8]

    price = snapshot["price"]

    # Gather data for sentiment agent (these are fast fetches)
    news = news_fetcher.get_news_context()
    news_sentiment_str = news_fetcher.get_market_sentiment_summary(news)
    logger.info("News sentiment: %s", news_sentiment_str)

    social_ctx = social_sentiment.get_social_context()
    social_data = social_sentiment.get_btc_social_data()
    logger.info("Social sentiment: %s", social_data.get("summary", "N/A"))

    onchain_ctx = onchain_data.get_onchain_context()

    # Cross-asset macro data (DXY, S&P500, Gold, VIX)
    macro_ctx = cross_asset.get_cross_asset_context()
    macro_data = cross_asset.get_cross_asset_data()
    logger.info("Macro context: %s", macro_data.get("summary", "N/A"))

    macro_section = f"\n\n{macro_ctx}" if macro_ctx else ""

    # Options market data (Deribit — put/call ratio, DVOL, max pain)
    options_ctx = options_data.get_options_context()
    opts_data = options_data.get_options_data()
    logger.info("Options: %s", opts_data.get("summary", "N/A"))

    options_section = f"\n\n{options_ctx}" if options_ctx else ""

    # Whale activity monitoring (on-chain + Binance large trades)
    whale_ctx = whale_monitor.get_whale_context()
    whale_data = whale_monitor.get_whale_data()
    logger.info("Whale activity: %s", whale_data.get("summary", "N/A"))

    whale_section = f"\n\n{whale_ctx}" if whale_ctx else ""

    # On-chain macro signals (MVRV-Z score, exchange flow)
    macro_onchain_ctx = onchain_macro.get_onchain_macro_context()
    if macro_onchain_ctx:
        logger.info("MVRV-Z: %s", onchain_macro.get_mvrv_z().get("signal", "N/A"))
        # Append to macro section so sentiment agent sees it
        macro_section += f"\n\n{macro_onchain_ctx}"

    # Real-time WebSocket data (live price, trade flow, liquidations, anomalies)
    ws_symbol = snapshot.get("symbol", "BTC/USDT").replace("/", "").lower()
    ws_ctx = ws_stream.get_ws_context(ws_symbol)
    if ws_ctx:
        logger.info("WebSocket [%s]: connected, live price $%.0f",
                     ws_symbol, ws_stream.get_realtime_price(ws_symbol))

    context = _build_context_sections()

    # Active grid/DCA context
    grid_ctx = grid_dca.get_grid_context()
    if grid_ctx:
        context += f"\n\n{grid_ctx}"

    # ML prediction (fast — local inference, no API call)
    ml_ctx = ""
    ml_data = {}
    if exchange:
        ml_data = ml_signal.predict(exchange)
        ml_ctx = ml_signal.get_ml_context(exchange)

    # Run Agent 1 (Market) and Agent 2 (Sentiment) in PARALLEL
    with ThreadPoolExecutor(max_workers=2) as pool:
        market_future = pool.submit(_run_market_agent, snapshot, ws_ctx)
        sentiment_future = pool.submit(
            _run_sentiment_agent, news, social_ctx, onchain_ctx,
            macro_section, options_section, whale_section,
        )
        market_assessment = market_future.result()
        sentiment_assessment = sentiment_future.result()

    logger.info(
        "Market agent: %s %s | Sentiment agent: %s",
        market_assessment.get("trend"), market_assessment.get("strength"),
        sentiment_assessment.get("overall_sentiment"),
    )

    # Run Agent 3 (Decision) — needs all assessments
    decision = _run_decision_agent(
        market_assessment, sentiment_assessment, portfolio, price, context,
        ml_ctx,
    )

    # Clamp trade size to safety limits
    decision["trade_usd"] = max(
        config.MIN_TRADE_USD,
        min(config.MAX_TRADE_USD, float(decision.get("trade_usd", config.BASE_TRADE_USD))),
    )
    decision.setdefault("action", "hold")
    decision.setdefault("confidence", 0.5)
    decision.setdefault("risk", "low")
    decision.setdefault("reason", "No reason provided")
    decision.setdefault("signals", [])

    # Attach metadata for Telegram message and logging
    decision["news_sentiment"] = news_sentiment_str
    decision["social_sentiment"] = social_data.get("summary", "")
    decision["market_assessment"] = market_assessment.get("summary", "")
    decision["sentiment_assessment"] = sentiment_assessment.get("summary", "")
    decision["ml_prediction"] = ml_data.get("ml_direction", "")
    decision["ml_probability"] = ml_data.get("ml_probability_up")
    # Gate flags persisted with the decision → per-signal attribution later
    decision["ml_buy_signal"] = bool(ml_data.get("ml_buy_signal"))
    decision["ml_sell_signal"] = bool(ml_data.get("ml_sell_signal"))

    logger.info(
        "Decision: %s $%.2f (conf %.0f%%) — %s",
        decision["action"].upper(), decision["trade_usd"],
        decision["confidence"] * 100, decision["reason"],
    )
    return decision
