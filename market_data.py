import logging
import requests
import pandas as pd
import ccxt
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import config

logger = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"


def get_exchange() -> ccxt.binance:
    exchange = ccxt.binance({
        "apiKey": config.BINANCE_API_KEY,
        "secret": config.BINANCE_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    if config.TESTNET:
        exchange.set_sandbox_mode(True)
    return exchange


def get_fear_greed() -> dict:
    """
    Fetch Fear & Greed index with 30-day history from alternative.me (free, no key).
    Returns current value, label, 7-day trend direction, and 7-day average.
    Trend context is more useful to Claude than the point-in-time value alone.
    """
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=30", timeout=5)
        data = r.json()["data"]
        current = data[0]

        recent_7 = [int(d["value"]) for d in data[:7]]
        older_7  = [int(d["value"]) for d in data[7:14]] if len(data) >= 14 else recent_7
        avg_recent = sum(recent_7) / len(recent_7)
        avg_older  = sum(older_7)  / len(older_7)

        if avg_recent > avg_older + 3:
            trend = "rising"        # sentiment improving
        elif avg_recent < avg_older - 3:
            trend = "falling"       # sentiment deteriorating
        else:
            trend = "flat"

        return {
            "value":    int(current["value"]),
            "label":    current["value_classification"],
            "trend_7d": trend,
            "avg_7d":   round(avg_recent, 1),
        }
    except Exception:
        return {"value": 50, "label": "Neutral", "trend_7d": "flat", "avg_7d": 50}


def _regime_from_series(closes: pd.Series) -> str:
    """Classify a price series as bullish / bearish / neutral using RSI + SMA cross."""
    n = len(closes)
    if n < 21:
        return "neutral"
    w14 = min(14, n - 1)
    w20 = min(20, n - 1)
    w50 = min(50, n - 1)
    rsi_val = RSIIndicator(closes, window=w14).rsi().iloc[-1]
    sma20   = SMAIndicator(closes, window=w20).sma_indicator().iloc[-1]
    sma50   = SMAIndicator(closes, window=w50).sma_indicator().iloc[-1]
    price   = closes.iloc[-1]
    bull = sum([rsi_val > 52, price > sma20, sma20 > sma50])
    bear = sum([rsi_val < 48, price < sma20, sma20 < sma50])
    if bull >= 2:
        return "bullish"
    if bear >= 2:
        return "bearish"
    return "neutral"


def compute_timeframe_consensus(df: pd.DataFrame) -> dict:
    """
    Derive regime agreement across 1h, 4h, and 1d by resampling the 168-candle
    1h DataFrame already fetched in get_market_snapshot.

    Returns:
      tf_regime_1h/4h/1d  — individual regime labels
      tf_direction        — 'bullish' | 'bearish' | 'mixed'
      tf_agreement        — how many of the 3 timeframes agree (0-3)
    """
    df_idx = df.copy()
    df_idx.index = pd.to_datetime(df_idx["ts"], unit="ms")

    regime_1h = _regime_from_series(df_idx["close"])

    df_4h = df_idx.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    regime_4h = _regime_from_series(df_4h["close"])

    df_1d = df_idx.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    regime_1d = _regime_from_series(df_1d["close"])

    regimes = [regime_1h, regime_4h, regime_1d]
    bull_count = regimes.count("bullish")
    bear_count = regimes.count("bearish")

    if bull_count >= 2:
        direction, agreement = "bullish", bull_count
    elif bear_count >= 2:
        direction, agreement = "bearish", bear_count
    else:
        direction, agreement = "mixed", max(bull_count, bear_count)

    return {
        "tf_regime_1h":  regime_1h,
        "tf_regime_4h":  regime_4h,
        "tf_regime_1d":  regime_1d,
        "tf_direction":  direction,
        "tf_agreement":  agreement,
    }


def get_market_snapshot(exchange: ccxt.binance) -> dict:
    ohlcv = exchange.fetch_ohlcv(config.SYMBOL, "1h", limit=168)  # 7 days hourly
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # Core indicators (existing)
    rsi   = RSIIndicator(close, window=14).rsi()
    sma20 = SMAIndicator(close, window=20).sma_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    bb    = BollingerBands(close, window=20, window_dev=2)

    # New indicators
    macd_ind   = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    stoch_rsi  = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    atr        = AverageTrueRange(high, low, close, window=14)
    obv        = OnBalanceVolumeIndicator(close, volume)
    vwap       = VolumeWeightedAveragePrice(high, low, close, volume, window=24)
    ichimoku   = IchimokuIndicator(high, low, window1=9, window2=26, window3=52)

    price = close.iloc[-1]
    fg    = get_fear_greed()
    deriv = get_derivatives_data()

    # MACD histogram direction for trend signal
    macd_hist = macd_ind.macd_diff().iloc[-1]
    macd_hist_prev = macd_ind.macd_diff().iloc[-2]
    macd_trend = "bullish" if macd_hist > 0 else "bearish"
    macd_momentum = "strengthening" if abs(macd_hist) > abs(macd_hist_prev) else "weakening"

    # Ichimoku cloud position
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

    snap = {
        "symbol":         config.SYMBOL,
        "price":          round(price, 2),
        "change_24h_pct": round((price / close.iloc[-24] - 1) * 100, 2),
        "change_7d_pct":  round((price / close.iloc[0]   - 1) * 100, 2),
        "volume_24h_btc": round(volume.iloc[-24:].sum(), 2),
        "rsi":            round(rsi.iloc[-1], 1),
        "sma20":          round(sma20.iloc[-1], 2),
        "sma50":          round(sma50.iloc[-1], 2),
        "bb_upper":       round(bb.bollinger_hband().iloc[-1], 2),
        "bb_lower":       round(bb.bollinger_lband().iloc[-1], 2),
        "vs_sma20_pct":   round((price / sma20.iloc[-1] - 1) * 100, 2),
        "vs_sma50_pct":   round((price / sma50.iloc[-1] - 1) * 100, 2),
        "fear_greed":       fg["value"],
        "fear_greed_lbl":   fg["label"],
        "fear_greed_trend": fg["trend_7d"],
        "fear_greed_avg7d": fg["avg_7d"],
        # MACD
        "macd":           round(macd_ind.macd().iloc[-1], 2),
        "macd_signal":    round(macd_ind.macd_signal().iloc[-1], 2),
        "macd_histogram": round(macd_hist, 2),
        "macd_trend":     macd_trend,
        "macd_momentum":  macd_momentum,
        # Stochastic RSI
        "stoch_rsi_k":    round(stoch_rsi.stochrsi_k().iloc[-1], 1),
        "stoch_rsi_d":    round(stoch_rsi.stochrsi_d().iloc[-1], 1),
        # ATR (volatility)
        "atr":            round(atr.average_true_range().iloc[-1], 2),
        "atr_pct":        round(atr.average_true_range().iloc[-1] / price * 100, 2),
        # OBV trend
        "obv_slope":      "rising" if obv.on_balance_volume().iloc[-1] > obv.on_balance_volume().iloc[-5] else "falling",
        # VWAP
        "vwap":           round(vwap.volume_weighted_average_price().iloc[-1], 2),
        "vs_vwap_pct":    round((price / vwap.volume_weighted_average_price().iloc[-1] - 1) * 100, 2),
        # Ichimoku
        "ichimoku_signal": ichimoku_signal,
    }

    # Multi-timeframe regime consensus (uses the same df already in memory)
    snap.update(compute_timeframe_consensus(df))

    # Derivatives data (funding rate, open interest, long/short ratio)
    oi_btc = deriv["open_interest_btc"]
    oi_usd = round(oi_btc * price, 0) if oi_btc else None
    snap.update({
        "funding_rate":        deriv["funding_rate"],
        "funding_rate_annual": deriv["funding_rate_annual"],
        "open_interest_btc":   oi_btc,
        "open_interest_usd":   oi_usd,
        "long_short_ratio":    deriv["long_short_ratio"],
        "long_pct":            deriv["long_pct"],
        "short_pct":           deriv["short_pct"],
    })

    return snap


def get_derivatives_data() -> dict:
    """
    Fetch BTC/USDT perpetual derivatives data from Binance Futures public API.
    Returns funding rate, open interest, and long/short ratio.
    No auth required — these are public endpoints.
    """
    result = {
        "funding_rate": None,
        "funding_rate_annual": None,
        "open_interest_btc": None,
        "open_interest_usd": None,
        "long_short_ratio": None,
        "long_pct": None,
        "short_pct": None,
    }

    # 1. Current funding rate
    try:
        r = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex",
            params={"symbol": "BTCUSDT"},
            timeout=5,
        )
        r.raise_for_status()
        d = r.json()
        rate = float(d.get("lastFundingRate", 0))
        result["funding_rate"] = round(rate * 100, 4)  # as percentage
        result["funding_rate_annual"] = round(rate * 3 * 365 * 100, 1)  # 3 settlements/day
    except Exception as exc:
        logger.debug("Funding rate fetch failed: %s", exc)

    # 2. Open interest
    try:
        r = requests.get(
            f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest",
            params={"symbol": "BTCUSDT"},
            timeout=5,
        )
        r.raise_for_status()
        d = r.json()
        oi_btc = float(d.get("openInterest", 0))
        result["open_interest_btc"] = round(oi_btc, 2)
    except Exception as exc:
        logger.debug("Open interest fetch failed: %s", exc)

    # 3. Long/Short ratio (top trader positions)
    try:
        r = requests.get(
            f"{BINANCE_FUTURES_BASE}/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "4h", "limit": 1},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            d = data[0]
            ratio = float(d.get("longShortRatio", 1.0))
            long_pct = float(d.get("longAccount", 0.5)) * 100
            short_pct = float(d.get("shortAccount", 0.5)) * 100
            result["long_short_ratio"] = round(ratio, 3)
            result["long_pct"] = round(long_pct, 1)
            result["short_pct"] = round(short_pct, 1)
    except Exception as exc:
        logger.debug("Long/short ratio fetch failed: %s", exc)

    return result


def get_portfolio(exchange: ccxt.binance) -> dict:
    bal  = exchange.fetch_balance()
    usdt = bal.get("USDT", {}).get("free", 0.0)
    btc  = bal.get("BTC",  {}).get("free", 0.0)
    return {"usdt": round(usdt, 2), "btc": round(btc, 8)}
