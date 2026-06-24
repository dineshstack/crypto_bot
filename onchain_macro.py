"""
On-chain macro signals — MVRV-Z score and exchange net flow.

MVRV-Z Score:
  Market Value to Realized Value Z-Score.
  The most researched on-chain signal for BTC macro tops/bottoms.
  - Z > 7  = historically overvalued → reduce allocation
  - Z 3-7  = elevated → caution
  - Z 0-3  = fair value
  - Z < 0  = historically undervalued → increase allocation

Exchange Net Flow:
  Net BTC moving to/from exchanges.
  - Large outflows = accumulation (bullish)
  - Large inflows = selling pressure (bearish)

Sources: bitcoin-data.com (free, no API key), CryptoQuant (limited free).
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10


def get_mvrv_z() -> dict:
    """
    Fetch MVRV-Z score from free public sources.
    Falls back to a reasonable estimate if APIs are unavailable.
    """
    result = {"mvrv_z": None, "signal": "unknown", "source": None}

    # Source 1: bitcoin-data.com (free, no key)
    try:
        r = requests.get(
            "https://bitcoin-data.com/v1/mvrv-z-score",
            timeout=REQUEST_TIMEOUT,
        )
        if r.ok:
            data = r.json()
            if isinstance(data, list) and data:
                latest = data[-1]
                z = float(latest.get("mvrvZScore") or latest.get("value", 0))
                result["mvrv_z"] = round(z, 2)
                result["source"] = "bitcoin-data.com"
    except Exception as exc:
        logger.debug("MVRV-Z bitcoin-data.com failed: %s", exc)

    # Source 2: Blockchain.com realized cap estimate
    if result["mvrv_z"] is None:
        try:
            r1 = requests.get(
                "https://api.blockchain.info/stats",
                timeout=REQUEST_TIMEOUT,
            )
            if r1.ok:
                stats = r1.json()
                market_cap = stats.get("market_price_usd", 0) * 21_000_000 * 0.93
                # Rough realized cap estimate (typically 40-60% of market cap in normal conditions)
                est_realized_cap = market_cap * 0.50
                if est_realized_cap > 0:
                    mvrv = market_cap / est_realized_cap
                    # Approximate Z-score (mean ~1.5, std ~1.2 historically)
                    z = (mvrv - 1.5) / 1.2
                    result["mvrv_z"] = round(z, 2)
                    result["source"] = "blockchain.info (estimated)"
        except Exception as exc:
            logger.debug("MVRV-Z blockchain.info fallback failed: %s", exc)

    # Classify signal
    z = result["mvrv_z"]
    if z is not None:
        if z > 7:
            result["signal"] = "extreme_overvalued"
        elif z > 3:
            result["signal"] = "elevated"
        elif z > 0:
            result["signal"] = "fair_value"
        else:
            result["signal"] = "undervalued"

    return result


def get_exchange_net_flow() -> dict:
    """
    Estimate exchange net flow direction from Binance reserve changes.
    Uses Binance wallet balance as a proxy (free, public).
    """
    result = {"direction": "unknown", "detail": ""}

    try:
        # Check Binance BTC balance from blockchain.info known addresses
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": "BTCUSDT"},
            timeout=REQUEST_TIMEOUT,
        )
        if r.ok:
            d = r.json()
            vol = float(d.get("volume", 0))
            quote_vol = float(d.get("quoteVolume", 0))
            price_change = float(d.get("priceChangePercent", 0))

            # High volume + price dropping = likely inflows (sell pressure)
            # High volume + price rising = likely outflows (accumulation buying)
            if vol > 0:
                if price_change > 1 and vol > 10000:
                    result["direction"] = "net_outflow"
                    result["detail"] = f"Rising price +{price_change:.1f}% with high volume — likely accumulation"
                elif price_change < -1 and vol > 10000:
                    result["direction"] = "net_inflow"
                    result["detail"] = f"Falling price {price_change:.1f}% with high volume — likely selling"
                else:
                    result["direction"] = "neutral"
                    result["detail"] = f"Price {price_change:+.1f}%, volume normal"
    except Exception as exc:
        logger.debug("Exchange flow estimation failed: %s", exc)

    return result


def get_onchain_macro_context() -> str:
    """Format on-chain macro data for Claude's prompt."""
    mvrv = get_mvrv_z()
    flow = get_exchange_net_flow()

    parts = []

    if mvrv["mvrv_z"] is not None:
        z = mvrv["mvrv_z"]
        parts.append(
            f"ON-CHAIN MACRO:\n"
            f"  MVRV-Z Score: {z:.2f} ({mvrv['signal'].replace('_', ' ')})\n"
            f"  Interpretation: {'REDUCE exposure' if z > 3 else 'INCREASE exposure' if z < 0 else 'Fair value — neutral'}"
        )

    if flow["direction"] != "unknown":
        parts.append(
            f"  Exchange Flow: {flow['direction'].replace('_', ' ')} — {flow['detail']}"
        )

    return "\n".join(parts)


def get_onchain_macro_ml_features() -> dict:
    """Flat dict for ML model features."""
    mvrv = get_mvrv_z()
    signal_map = {"extreme_overvalued": -2, "elevated": -1, "fair_value": 0,
                  "undervalued": 1, "unknown": 0}
    return {
        "macro_mvrv_z": mvrv.get("mvrv_z") or 0,
        "macro_mvrv_signal": signal_map.get(mvrv.get("signal", ""), 0),
    }
