"""
Whale transaction monitoring — detect large BTC movements.

Two data sources (both free, no API key):
  1. Blockchain.info — on-chain large transactions (>50 BTC)
  2. Binance aggTrades — exchange-level large orders (>$100K)

Trading signals:
  - Large exchange inflows (deposit to known exchange addresses) = sell pressure
  - Large exchange outflows = accumulation (bullish)
  - Spike in large Binance sells = whale dumping
  - Spike in large Binance buys = whale accumulating
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15
ONCHAIN_WHALE_THRESHOLD_BTC = 50
EXCHANGE_WHALE_THRESHOLD_USD = 100_000


def _get_recent_large_onchain_txs() -> dict:
    """Check latest block for large BTC transactions."""
    result = {"large_tx_count": 0, "total_btc_moved": 0, "largest_tx_btc": 0}

    try:
        r = requests.get("https://blockchain.info/latestblock", timeout=REQUEST_TIMEOUT)
        if not r.ok:
            return result
        block_hash = r.json().get("hash")
        if not block_hash:
            return result

        r2 = requests.get(
            f"https://blockchain.info/rawblock/{block_hash}?limit=50",
            timeout=REQUEST_TIMEOUT,
        )
        if not r2.ok:
            return result

        txs = r2.json().get("tx", [])
        large_count = 0
        total_moved = 0.0
        largest = 0.0

        for tx in txs:
            total_out = sum(o.get("value", 0) for o in tx.get("out", []))
            btc_val = total_out / 1e8
            if btc_val >= ONCHAIN_WHALE_THRESHOLD_BTC:
                large_count += 1
                total_moved += btc_val
                largest = max(largest, btc_val)

        result["large_tx_count"] = large_count
        result["total_btc_moved"] = round(total_moved, 2)
        result["largest_tx_btc"] = round(largest, 2)

    except Exception as exc:
        logger.debug("On-chain whale scan failed: %s", exc)

    return result


def _get_binance_large_trades() -> dict:
    """Check recent Binance BTC/USDT trades for whale orders."""
    result = {
        "large_buys": 0, "large_sells": 0,
        "buy_volume_usd": 0, "sell_volume_usd": 0,
        "net_flow": "neutral",
    }

    try:
        r = requests.get(
            "https://api.binance.com/api/v3/aggTrades",
            params={"symbol": "BTCUSDT", "limit": 1000},
            timeout=REQUEST_TIMEOUT,
        )
        if not r.ok:
            return result

        trades = r.json()
        buy_vol = 0.0
        sell_vol = 0.0
        buy_count = 0
        sell_count = 0

        for t in trades:
            usd_val = float(t["q"]) * float(t["p"])
            if usd_val >= EXCHANGE_WHALE_THRESHOLD_USD:
                if t["m"]:  # maker is buyer → trade is a sell
                    sell_vol += usd_val
                    sell_count += 1
                else:
                    buy_vol += usd_val
                    buy_count += 1

        result["large_buys"] = buy_count
        result["large_sells"] = sell_count
        result["buy_volume_usd"] = round(buy_vol, 0)
        result["sell_volume_usd"] = round(sell_vol, 0)

        if buy_vol > sell_vol * 1.5:
            result["net_flow"] = "whale_buying"
        elif sell_vol > buy_vol * 1.5:
            result["net_flow"] = "whale_selling"
        else:
            result["net_flow"] = "neutral"

    except Exception as exc:
        logger.debug("Binance whale trade scan failed: %s", exc)

    return result


def get_whale_data() -> dict:
    """Combined whale activity from on-chain + exchange."""
    onchain = _get_recent_large_onchain_txs()
    exchange = _get_binance_large_trades()

    # Overall whale signal
    signal = "neutral"
    if exchange["net_flow"] == "whale_buying":
        signal = "bullish"
    elif exchange["net_flow"] == "whale_selling":
        signal = "bearish"
    elif onchain["large_tx_count"] >= 5:
        signal = "high_activity"

    parts = []
    if onchain["large_tx_count"] > 0:
        parts.append(
            f"on-chain: {onchain['large_tx_count']} large txs "
            f"({onchain['total_btc_moved']:.0f} BTC, "
            f"largest {onchain['largest_tx_btc']:.0f} BTC)"
        )
    if exchange["large_buys"] + exchange["large_sells"] > 0:
        parts.append(
            f"exchange: {exchange['large_buys']} buys / "
            f"{exchange['large_sells']} sells ({exchange['net_flow']})"
        )

    return {
        "onchain": onchain,
        "exchange": exchange,
        "whale_signal": signal,
        "summary": ", ".join(parts) if parts else "no whale activity detected",
    }


def get_whale_context() -> str:
    """Format whale data for Claude's prompt."""
    data = get_whale_data()

    if data["summary"] == "no whale activity detected":
        return ""

    oc = data["onchain"]
    ex = data["exchange"]
    lines = ["WHALE ACTIVITY (real-time):"]

    if oc["large_tx_count"] > 0:
        lines.append(
            f"  On-chain:    {oc['large_tx_count']} large txs "
            f"(>{ONCHAIN_WHALE_THRESHOLD_BTC} BTC) — "
            f"{oc['total_btc_moved']:,.0f} BTC total, "
            f"largest {oc['largest_tx_btc']:,.0f} BTC"
        )

    total_whale_trades = ex["large_buys"] + ex["large_sells"]
    if total_whale_trades > 0:
        lines.append(
            f"  Binance:     {ex['large_buys']} whale buys "
            f"(${ex['buy_volume_usd']:,.0f}) / "
            f"{ex['large_sells']} whale sells "
            f"(${ex['sell_volume_usd']:,.0f})"
        )
        lines.append(f"  Net flow:    {ex['net_flow'].replace('_', ' ')}")

    lines.append(f"  Signal:      {data['whale_signal']}")

    return "\n".join(lines)
