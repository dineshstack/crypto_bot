"""
BTC Options market data from Deribit (free public API, no auth needed).

Provides leading indicators that spot/futures markets don't show:
  - Put/Call ratio: crowd positioning (< 0.7 = bullish, > 1.0 = bearish/hedging)
  - DVOL (Deribit Volatility Index): implied volatility expectation
  - Max pain strike: price level where most options expire worthless
  - IV skew: whether puts or calls are more expensive (fear vs greed)
  - Large OI clusters: magnetic price levels

Options market often LEADS spot by 12-48 hours. The Feb 2026 crash was
signaled by 25-delta skew falling to -19.34 before BTC dropped from $90K to $60K.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
REQUEST_TIMEOUT = 12


def _deribit_get(method: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(
            f"{DERIBIT_BASE}/{method}",
            params=params or {},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("result")
    except Exception as exc:
        logger.debug("Deribit %s failed: %s", method, exc)
        return None


def get_options_data() -> dict:
    """
    Fetch comprehensive BTC options market data.
    All from Deribit public API — zero cost, no API key.
    """
    result = {
        "put_call_ratio": None,
        "put_call_signal": None,
        "total_call_oi_btc": None,
        "total_put_oi_btc": None,
        "dvol": None,
        "dvol_signal": None,
        "max_pain_strike": None,
        "top_oi_strikes": [],
        "index_price": None,
        "max_pain_distance_pct": None,
        "summary": "no options data",
    }

    # 1. Put/Call open interest ratio
    book = _deribit_get("get_book_summary_by_currency",
                        {"currency": "BTC", "kind": "option"})
    if book:
        calls = [d for d in book if d.get("instrument_name", "").endswith("C")]
        puts = [d for d in book if d.get("instrument_name", "").endswith("P")]

        total_call_oi = sum(d.get("open_interest", 0) for d in calls)
        total_put_oi = sum(d.get("open_interest", 0) for d in puts)

        result["total_call_oi_btc"] = round(total_call_oi, 0)
        result["total_put_oi_btc"] = round(total_put_oi, 0)

        if total_call_oi > 0:
            pcr = total_put_oi / total_call_oi
            result["put_call_ratio"] = round(pcr, 3)

            if pcr < 0.5:
                result["put_call_signal"] = "very_bullish"
            elif pcr < 0.7:
                result["put_call_signal"] = "bullish"
            elif pcr < 1.0:
                result["put_call_signal"] = "neutral"
            elif pcr < 1.3:
                result["put_call_signal"] = "bearish"
            else:
                result["put_call_signal"] = "very_bearish"

        # Max pain calculation (strike with highest total OI)
        strike_oi = defaultdict(float)
        for d in book:
            name = d.get("instrument_name", "")
            parts = name.split("-")
            if len(parts) >= 3:
                try:
                    strike = float(parts[2])
                    strike_oi[strike] += d.get("open_interest", 0)
                except ValueError:
                    pass

        if strike_oi:
            max_pain = max(strike_oi.items(), key=lambda x: x[1])
            result["max_pain_strike"] = max_pain[0]

            top_strikes = sorted(strike_oi.items(), key=lambda x: -x[1])[:5]
            result["top_oi_strikes"] = [
                {"strike": s, "oi_btc": round(oi, 0)} for s, oi in top_strikes
            ]

    # 2. DVOL (Deribit Volatility Index)
    dvol_data = _deribit_get("get_volatility_index_data", {
        "currency": "BTC",
        "resolution": 3600,
        "start_timestamp": 0,
        "end_timestamp": 9999999999999,
    })
    if dvol_data and dvol_data.get("data"):
        latest = dvol_data["data"][-1]
        dvol = latest[4]  # close value
        result["dvol"] = round(dvol, 1)

        if dvol < 30:
            result["dvol_signal"] = "low_vol_calm"
        elif dvol < 50:
            result["dvol_signal"] = "moderate_vol"
        elif dvol < 70:
            result["dvol_signal"] = "high_vol"
        else:
            result["dvol_signal"] = "extreme_vol_fear"

    # 3. Index price (for distance-to-max-pain calculation)
    idx = _deribit_get("get_index_price", {"index_name": "btc_usd"})
    if idx:
        result["index_price"] = idx.get("index_price")
        if result["max_pain_strike"] and result["index_price"]:
            dist = (result["max_pain_strike"] / result["index_price"] - 1) * 100
            result["max_pain_distance_pct"] = round(dist, 1)

    # Build summary
    parts = []
    pcr = result["put_call_ratio"]
    if pcr is not None:
        parts.append(f"P/C={pcr:.2f} ({result['put_call_signal']})")
    dvol = result["dvol"]
    if dvol:
        parts.append(f"DVOL={dvol:.0f} ({result['dvol_signal']})")
    mp = result["max_pain_strike"]
    if mp:
        dist = result.get("max_pain_distance_pct", 0)
        parts.append(f"MaxPain=${mp:,.0f} ({dist:+.1f}%)")

    result["summary"] = ", ".join(parts) if parts else "no options data"
    return result


def get_options_context() -> str:
    """Format options data for Claude's prompt."""
    data = get_options_data()

    if data["summary"] == "no options data":
        return ""

    lines = ["OPTIONS MARKET (Deribit — leading indicator):"]

    pcr = data["put_call_ratio"]
    if pcr is not None:
        call_oi = data["total_call_oi_btc"]
        put_oi = data["total_put_oi_btc"]
        signal = data["put_call_signal"]
        lines.append(
            f"  Put/Call OI:   {pcr:.2f} ({signal}) — "
            f"{call_oi:,.0f} BTC calls / {put_oi:,.0f} BTC puts"
        )

    dvol = data["dvol"]
    if dvol:
        lines.append(f"  DVOL (IV):     {dvol:.0f}% ({data['dvol_signal']})")

    mp = data["max_pain_strike"]
    if mp:
        dist = data.get("max_pain_distance_pct", 0)
        direction = "above" if dist > 0 else "below"
        lines.append(
            f"  Max Pain:      ${mp:,.0f} (price is {abs(dist):.1f}% {direction} max pain)"
        )

    strikes = data.get("top_oi_strikes", [])
    if strikes:
        strike_str = ", ".join(f"${s['strike']:,.0f}" for s in strikes[:3])
        lines.append(f"  Key OI levels: {strike_str}")

    return "\n".join(lines)


def get_options_ml_features() -> dict:
    """Return flat dict of features for the ML model."""
    data = get_options_data()
    signal_map = {
        "very_bullish": 2, "bullish": 1, "neutral": 0,
        "bearish": -1, "very_bearish": -2,
    }
    vol_map = {
        "low_vol_calm": -1, "moderate_vol": 0,
        "high_vol": 1, "extreme_vol_fear": 2,
    }
    return {
        "opt_put_call_ratio": data.get("put_call_ratio", 0) or 0,
        "opt_pc_signal": signal_map.get(data.get("put_call_signal", ""), 0),
        "opt_dvol": data.get("dvol", 0) or 0,
        "opt_dvol_signal": vol_map.get(data.get("dvol_signal", ""), 0),
        "opt_max_pain_dist_pct": data.get("max_pain_distance_pct", 0) or 0,
    }
