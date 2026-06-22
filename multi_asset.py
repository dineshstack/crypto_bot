"""
Multi-asset analysis — run the Claude pipeline for any supported symbol.

Generalizes the BTC-only analysis to support ETH/USDT and future assets.
Each asset gets its own market snapshot, derivatives data, and analysis cycle.
The Decision Agent sees ALL assets' positions when making per-asset decisions.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
import ccxt
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

import config

logger = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"


def get_asset_snapshot(exchange: ccxt.binance, symbol: str) -> dict:
    """
    Build a market snapshot for any supported symbol.
    Same structure as market_data.get_market_snapshot() but symbol-generic.
    """
    ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=168)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    rsi = RSIIndicator(close, window=14).rsi()
    sma20 = SMAIndicator(close, window=20).sma_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    bb = BollingerBands(close, window=20, window_dev=2)

    macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    stoch_rsi = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    atr = AverageTrueRange(high, low, close, window=14)
    obv = OnBalanceVolumeIndicator(close, volume)
    vwap = VolumeWeightedAveragePrice(high, low, close, volume, window=24)
    ichimoku = IchimokuIndicator(high, low, window1=9, window2=26, window3=52)

    price = close.iloc[-1]

    macd_hist = macd_ind.macd_diff().iloc[-1]
    macd_hist_prev = macd_ind.macd_diff().iloc[-2]
    macd_trend = "bullish" if macd_hist > 0 else "bearish"
    macd_momentum = "strengthening" if abs(macd_hist) > abs(macd_hist_prev) else "weakening"

    span_a = ichimoku.ichimoku_a().iloc[-1]
    span_b = ichimoku.ichimoku_b().iloc[-1]
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    if price > cloud_top:
        ichimoku_signal = "above_cloud_bullish"
    elif price < cloud_bottom:
        ichimoku_signal = "below_cloud_bearish"
    else:
        ichimoku_signal = "inside_cloud_neutral"

    asset_cfg = config.ASSET_CONFIG.get(symbol, {})
    futures_sym = asset_cfg.get("futures_symbol", symbol.replace("/", ""))
    deriv = get_asset_derivatives(futures_sym)

    snap = {
        "symbol": symbol,
        "base": asset_cfg.get("base", symbol.split("/")[0]),
        "price": round(price, 2),
        "change_24h_pct": round((price / close.iloc[-24] - 1) * 100, 2),
        "change_7d_pct": round((price / close.iloc[0] - 1) * 100, 2),
        "volume_24h_btc": round(volume.iloc[-24:].sum(), 2),
        "rsi": round(rsi.iloc[-1], 1),
        "sma20": round(sma20.iloc[-1], 2),
        "sma50": round(sma50.iloc[-1], 2),
        "bb_upper": round(bb.bollinger_hband().iloc[-1], 2),
        "bb_lower": round(bb.bollinger_lband().iloc[-1], 2),
        "vs_sma20_pct": round((price / sma20.iloc[-1] - 1) * 100, 2),
        "vs_sma50_pct": round((price / sma50.iloc[-1] - 1) * 100, 2),
        "macd": round(macd_ind.macd().iloc[-1], 2),
        "macd_signal": round(macd_ind.macd_signal().iloc[-1], 2),
        "macd_histogram": round(macd_hist, 2),
        "macd_trend": macd_trend,
        "macd_momentum": macd_momentum,
        "stoch_rsi_k": round(stoch_rsi.stochrsi_k().iloc[-1], 1),
        "stoch_rsi_d": round(stoch_rsi.stochrsi_d().iloc[-1], 1),
        "atr": round(atr.average_true_range().iloc[-1], 2),
        "atr_pct": round(atr.average_true_range().iloc[-1] / price * 100, 2),
        "obv_slope": "rising" if obv.on_balance_volume().iloc[-1] > obv.on_balance_volume().iloc[-5] else "falling",
        "vwap": round(vwap.volume_weighted_average_price().iloc[-1], 2),
        "vs_vwap_pct": round((price / vwap.volume_weighted_average_price().iloc[-1] - 1) * 100, 2),
        "ichimoku_signal": ichimoku_signal,
        "funding_rate": deriv["funding_rate"],
        "funding_rate_annual": deriv["funding_rate_annual"],
        "open_interest_btc": deriv["open_interest_qty"],
        "open_interest_usd": round(deriv["open_interest_qty"] * price, 0) if deriv["open_interest_qty"] else None,
        "long_short_ratio": deriv["long_short_ratio"],
        "long_pct": deriv["long_pct"],
        "short_pct": deriv["short_pct"],
    }

    return snap


def get_asset_derivatives(futures_symbol: str) -> dict:
    """Fetch derivatives data for any futures symbol (BTCUSDT, ETHUSDT, etc.)."""
    result = {
        "funding_rate": None,
        "funding_rate_annual": None,
        "open_interest_qty": None,
        "long_short_ratio": None,
        "long_pct": None,
        "short_pct": None,
    }

    try:
        r = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex",
            params={"symbol": futures_symbol},
            timeout=5,
        )
        r.raise_for_status()
        d = r.json()
        rate = float(d.get("lastFundingRate", 0))
        result["funding_rate"] = round(rate * 100, 4)
        result["funding_rate_annual"] = round(rate * 3 * 365 * 100, 1)
    except Exception as exc:
        logger.debug("Funding rate for %s failed: %s", futures_symbol, exc)

    try:
        r = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest",
            params={"symbol": futures_symbol},
            timeout=5,
        )
        r.raise_for_status()
        d = r.json()
        result["open_interest_qty"] = round(float(d.get("openInterest", 0)), 2)
    except Exception as exc:
        logger.debug("Open interest for %s failed: %s", futures_symbol, exc)

    try:
        r = requests.get(
            f"{BINANCE_FUTURES_BASE}/futures/data/globalLongShortAccountRatio",
            params={"symbol": futures_symbol, "period": "4h", "limit": 1},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            d = data[0]
            result["long_short_ratio"] = round(float(d.get("longShortRatio", 1.0)), 3)
            result["long_pct"] = round(float(d.get("longAccount", 0.5)) * 100, 1)
            result["short_pct"] = round(float(d.get("shortAccount", 0.5)) * 100, 1)
    except Exception as exc:
        logger.debug("Long/short for %s failed: %s", futures_symbol, exc)

    return result


def get_asset_portfolio(exchange: ccxt.binance, symbol: str) -> dict:
    """Get portfolio holdings for a specific trading pair."""
    bal = exchange.fetch_balance()
    base = symbol.split("/")[0]
    quote = symbol.split("/")[1]
    base_free = bal.get(base, {}).get("free", 0.0)
    quote_free = bal.get(quote, {}).get("free", 0.0)
    return {
        "base": base,
        "quote": quote,
        "base_amount": round(base_free, 8),
        "quote_amount": round(quote_free, 2),
    }


def get_full_portfolio(exchange: ccxt.binance) -> dict:
    """
    Get complete portfolio state across all assets.
    Returns USDT balance + all crypto holdings with USD values.
    """
    bal = exchange.fetch_balance()
    usdt = bal.get("USDT", {}).get("free", 0.0)

    holdings = {}
    total_crypto_usd = 0.0

    for symbol in config.SYMBOLS:
        base = symbol.split("/")[0]
        amount = bal.get(base, {}).get("free", 0.0)
        if amount > 0:
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker.get("last", 0)
                usd_val = amount * price
            except Exception:
                price = 0
                usd_val = 0

            holdings[base] = {
                "amount": round(amount, 8),
                "price": round(price, 2),
                "usd_value": round(usd_val, 2),
            }
            total_crypto_usd += usd_val
        else:
            holdings[base] = {"amount": 0, "price": 0, "usd_value": 0}

    total = usdt + total_crypto_usd
    return {
        "usdt": round(usdt, 2),
        "holdings": holdings,
        "total_crypto_usd": round(total_crypto_usd, 2),
        "total_usd": round(total, 2),
        "crypto_alloc_pct": round(total_crypto_usd / total * 100, 1) if total > 0 else 0,
    }


def format_portfolio_context(portfolio: dict) -> str:
    """Format full portfolio for Claude's decision prompt."""
    lines = [f"  USDT:       ${portfolio['usdt']:.2f}"]
    for base, h in portfolio["holdings"].items():
        if h["amount"] > 0:
            lines.append(
                f"  {base}:        {h['amount']:.6f} "
                f"(${h['usd_value']:.2f} @ ${h['price']:,.2f})"
            )
        else:
            lines.append(f"  {base}:        0")
    lines.append(f"  Total:      ${portfolio['total_usd']:.2f}")
    lines.append(f"  Crypto:     {portfolio['crypto_alloc_pct']:.1f}% of portfolio")
    return "\n".join(lines)
