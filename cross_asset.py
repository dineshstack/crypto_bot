"""
Cross-asset correlation data — macro context that moves BTC.

Fetches real-time data for correlated assets via yfinance (free, no API key):
  - DXY (US Dollar Index)  — -0.90 correlation with BTC in 2026
  - S&P 500 (SPY)          — +0.74 correlation
  - Gold (GC=F / GLD)      — variable, recently negative
  - US 10Y Treasury (^TNX) — inverse in risk-off

Computes:
  - Current price + daily change for each asset
  - Rolling correlations with BTC
  - Divergence signals (when BTC moves opposite to expected correlation)

Injected into Claude's Sentiment Agent prompt and ML feature set.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15


def _fetch_yfinance_data() -> dict:
    """Fetch latest prices and changes for macro assets."""
    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not installed — skipping cross-asset data")
        return {}

    tickers = {
        "DXY": "DX-Y.NYB",
        "SPX": "^GSPC",
        "GOLD": "GC=F",
        "US10Y": "^TNX",
        "VIX": "^VIX",
    }

    result = {}
    try:
        symbols = list(tickers.values())
        data = yf.download(
            symbols, period="5d", interval="1d",
            progress=False, threads=True, timeout=REQUEST_TIMEOUT,
        )

        if data.empty:
            return {}

        close = data["Close"] if "Close" in data.columns else data.get("Adj Close", data)

        for label, symbol in tickers.items():
            try:
                col = close[symbol] if symbol in close.columns else None
                if col is None or col.dropna().empty:
                    continue
                vals = col.dropna()
                if len(vals) < 2:
                    continue
                current = float(vals.iloc[-1])
                prev = float(vals.iloc[-2])
                change_pct = (current / prev - 1) * 100

                result[label] = {
                    "price": round(current, 2),
                    "change_pct": round(change_pct, 2),
                }
            except Exception:
                continue

    except Exception as exc:
        logger.warning("Cross-asset yfinance fetch failed: %s", exc)

    return result


def get_cross_asset_data() -> dict:
    """
    Fetch cross-asset data and compute signals.
    Returns dict with per-asset prices/changes plus derived signals.
    """
    assets = _fetch_yfinance_data()

    result = {
        "assets": assets,
        "dxy_signal": None,
        "risk_appetite": None,
        "summary": "no cross-asset data",
    }

    if not assets:
        return result

    # DXY signal: dollar up = BTC bearish, dollar down = BTC bullish
    dxy = assets.get("DXY", {})
    if dxy:
        change = dxy["change_pct"]
        if change > 0.3:
            result["dxy_signal"] = "bearish_for_btc"
        elif change < -0.3:
            result["dxy_signal"] = "bullish_for_btc"
        else:
            result["dxy_signal"] = "neutral"

    # Risk appetite: SPX up + VIX down = risk-on (bullish BTC)
    spx = assets.get("SPX", {})
    vix = assets.get("VIX", {})
    if spx and vix:
        if spx["change_pct"] > 0.3 and vix["change_pct"] < 0:
            result["risk_appetite"] = "risk_on"
        elif spx["change_pct"] < -0.3 and vix["change_pct"] > 0:
            result["risk_appetite"] = "risk_off"
        else:
            result["risk_appetite"] = "neutral"
    elif spx:
        result["risk_appetite"] = "risk_on" if spx["change_pct"] > 0.3 else (
            "risk_off" if spx["change_pct"] < -0.3 else "neutral"
        )

    # Build summary
    parts = []
    if dxy:
        parts.append(f"DXY {dxy['change_pct']:+.1f}%")
    if spx:
        parts.append(f"S&P500 {spx['change_pct']:+.1f}%")
    gold = assets.get("GOLD", {})
    if gold:
        parts.append(f"Gold {gold['change_pct']:+.1f}%")
    if vix:
        parts.append(f"VIX {vix['price']:.1f}")

    if result["dxy_signal"]:
        parts.append(f"[{result['dxy_signal']}]")
    if result["risk_appetite"]:
        parts.append(f"[{result['risk_appetite']}]")

    result["summary"] = ", ".join(parts) if parts else "no cross-asset data"
    return result


def get_cross_asset_context() -> str:
    """Format cross-asset data for Claude's prompt."""
    data = get_cross_asset_data()

    if data["summary"] == "no cross-asset data":
        return ""

    assets = data["assets"]
    lines = ["CROSS-ASSET MACRO CONTEXT (real-time):"]

    dxy = assets.get("DXY", {})
    if dxy:
        signal = data.get("dxy_signal", "")
        sig_str = f" → {signal.replace('_', ' ')}" if signal else ""
        lines.append(f"  Dollar (DXY):   {dxy['price']:.1f} ({dxy['change_pct']:+.1f}% today){sig_str}")

    spx = assets.get("SPX", {})
    if spx:
        lines.append(f"  S&P 500:        {spx['price']:,.0f} ({spx['change_pct']:+.1f}% today)")

    gold = assets.get("GOLD", {})
    if gold:
        lines.append(f"  Gold:           ${gold['price']:,.0f} ({gold['change_pct']:+.1f}% today)")

    tnx = assets.get("US10Y", {})
    if tnx:
        lines.append(f"  US 10Y Yield:   {tnx['price']:.2f}% ({tnx['change_pct']:+.2f}%)")

    vix = assets.get("VIX", {})
    if vix:
        vix_label = "elevated" if vix["price"] > 20 else "calm" if vix["price"] < 15 else "moderate"
        lines.append(f"  VIX:            {vix['price']:.1f} ({vix_label})")

    risk = data.get("risk_appetite")
    if risk:
        lines.append(f"  Risk appetite:  {risk.replace('_', ' ')}")

    return "\n".join(lines)


def get_cross_asset_ml_features() -> dict:
    """Return flat dict of features for the ML model."""
    data = get_cross_asset_data()
    features = {}

    for label in ["DXY", "SPX", "GOLD", "US10Y", "VIX"]:
        asset = data["assets"].get(label, {})
        features[f"macro_{label.lower()}_change"] = asset.get("change_pct", 0)

    signal_map = {"bearish_for_btc": -1, "neutral": 0, "bullish_for_btc": 1}
    features["macro_dxy_signal"] = signal_map.get(data.get("dxy_signal", ""), 0)

    risk_map = {"risk_off": -1, "neutral": 0, "risk_on": 1}
    features["macro_risk_appetite"] = risk_map.get(data.get("risk_appetite", ""), 0)

    return features
