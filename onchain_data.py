"""
On-chain Bitcoin data from free public APIs (no keys required).

Sources:
  blockchain.info  — network stats: hash rate, tx volume, BTC sent
  mempool.space    — mempool: fees, congestion, pending tx count, mining hashrate

Trading signals derived:
  - Hash rate trend: rising = miner confidence (bullish), dropping = potential stress
  - Mempool congestion: high fees = heavy usage (bull market), low fees = calm
  - Transaction volume: spikes can signal large movements or exchange activity
  - Fee pressure: rising fees often precede big price moves
"""
from __future__ import annotations

import logging
import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 12


def get_onchain_data() -> dict:
    """
    Fetch Bitcoin on-chain metrics from free APIs.
    Returns a dict with network stats — never raises.
    """
    result = {
        "hash_rate_eh": None,
        "mempool_tx_count": None,
        "mempool_vsize_mb": None,
        "fee_fastest_sat": None,
        "fee_economy_sat": None,
        "fee_pressure": None,
        "network_tx_24h": None,
        "btc_sent_24h": None,
    }

    # 1. blockchain.info — 24h network stats
    try:
        r = requests.get("https://api.blockchain.info/stats", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        d = r.json()
        hash_rate_gh = d.get("hash_rate", 0)
        result["hash_rate_eh"] = round(hash_rate_gh / 1e9, 1) if hash_rate_gh else None
        result["network_tx_24h"] = d.get("n_tx")
        est_sent = d.get("estimated_btc_sent", 0)
        if est_sent:
            result["btc_sent_24h"] = round(est_sent / 1e8, 0)
    except Exception as exc:
        logger.debug("blockchain.info stats failed: %s", exc)

    # 2. mempool.space — fees
    try:
        r = requests.get(
            "https://mempool.space/api/v1/fees/recommended",
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()
        fastest = d.get("fastestFee", 0)
        economy = d.get("economyFee", 0)
        result["fee_fastest_sat"] = fastest
        result["fee_economy_sat"] = economy

        if fastest >= 50:
            result["fee_pressure"] = "very_high"
        elif fastest >= 20:
            result["fee_pressure"] = "high"
        elif fastest >= 5:
            result["fee_pressure"] = "moderate"
        else:
            result["fee_pressure"] = "low"
    except Exception as exc:
        logger.debug("mempool.space fees failed: %s", exc)

    # 3. mempool.space — mempool state
    try:
        r = requests.get(
            "https://mempool.space/api/mempool",
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()
        result["mempool_tx_count"] = d.get("count")
        vsize = d.get("vsize", 0)
        result["mempool_vsize_mb"] = round(vsize / 1e6, 1) if vsize else None
    except Exception as exc:
        logger.debug("mempool.space mempool failed: %s", exc)

    return result


def get_onchain_context() -> str:
    """Format on-chain data for Claude's prompt."""
    data = get_onchain_data()

    if data["hash_rate_eh"] is None and data["mempool_tx_count"] is None:
        return ""

    lines = ["ON-CHAIN DATA (Bitcoin network — real-time):"]

    hr = data.get("hash_rate_eh")
    if hr:
        lines.append(f"  Hash Rate:     {hr} EH/s")

    tx = data.get("network_tx_24h")
    if tx:
        lines.append(f"  Transactions:  {tx:,} (24h)")

    sent = data.get("btc_sent_24h")
    if sent:
        lines.append(f"  BTC Sent:      {sent:,.0f} BTC (24h)")

    mp_count = data.get("mempool_tx_count")
    mp_size = data.get("mempool_vsize_mb")
    if mp_count:
        size_str = f" / {mp_size} MB" if mp_size else ""
        lines.append(f"  Mempool:       {mp_count:,} pending tx{size_str}")

    fast = data.get("fee_fastest_sat")
    econ = data.get("fee_economy_sat")
    pressure = data.get("fee_pressure", "unknown")
    if fast is not None:
        lines.append(f"  Fees:          {fast} sat/vB fast, {econ} sat/vB economy ({pressure} pressure)")

    return "\n".join(lines)
