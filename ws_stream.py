"""
Real-time Binance WebSocket data stream with anomaly detection.

Connects to Binance combined WebSocket streams for BTC/USDT:
  - kline@1m:     1-minute candlesticks for price monitoring
  - aggTrade:     aggregate trades for volume/whale detection
  - miniTicker:   24h mini-ticker for quick stats
  - forceOrder:   futures liquidation events (cascade detection)

Anomaly detection (runs continuously, can interrupt 4h sleep):
  - Flash crash:  price drops >CRASH_THRESHOLD_PCT in CRASH_WINDOW_SEC
  - Breakout:     price rises >BREAKOUT_THRESHOLD_PCT in same window
  - Volume spike: trade volume 5x above rolling average
  - Liquidation cascade: >$5M liquidated in 5 minutes

Designed to run as a background asyncio task alongside the trading loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
logger = logging.getLogger(__name__)

# ── Detection thresholds ────────────────────────────────────────────────────

CRASH_THRESHOLD_PCT = 2.0
BREAKOUT_THRESHOLD_PCT = 2.0
CRASH_WINDOW_SEC = 300          # 5-minute rolling window
VOLUME_SPIKE_MULTIPLIER = 5.0   # 5x above 1h average = spike
LIQUIDATION_THRESHOLD_USD = 5_000_000  # $5M in 5 min = cascade
PRICE_HISTORY_MAXLEN = 600      # 10 min of per-second prices

# ── WebSocket URLs ──────────────────────────────────────────────────────────

SPOT_WS = "wss://stream.binance.com:9443/stream"
FUTURES_WS = "wss://fstream.binance.com/stream"

SPOT_STREAMS = [
    "btcusdt@kline_1m",
    "btcusdt@aggTrade",
    "btcusdt@miniTicker",
    "ethusdt@kline_1m",
    "ethusdt@aggTrade",
    "ethusdt@miniTicker",
]
FUTURES_STREAMS = [
    "btcusdt@forceOrder",
    "ethusdt@forceOrder",
]


@dataclass
class AnomalyEvent:
    """Represents a detected market anomaly."""
    event_type: str       # flash_crash | breakout | volume_spike | liquidation_cascade
    severity: str         # warning | critical
    symbol: str           # btcusdt | ethusdt
    price: float
    change_pct: float
    detail: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SymbolState:
    """Per-symbol real-time state."""
    symbol: str = ""
    price: float = 0.0
    volume_24h: float = 0.0
    price_change_24h_pct: float = 0.0
    last_update: float = 0.0

    price_history: deque = field(default_factory=lambda: deque(maxlen=PRICE_HISTORY_MAXLEN))

    recent_buys_usd: float = 0.0
    recent_sells_usd: float = 0.0
    trade_count_1m: int = 0
    _trade_window: deque = field(default_factory=lambda: deque(maxlen=10000))

    _volume_1m_history: deque = field(default_factory=lambda: deque(maxlen=60))
    current_1m_volume_usd: float = 0.0
    _current_1m_start: float = 0.0

    _liq_window: deque = field(default_factory=lambda: deque(maxlen=1000))
    liq_total_5m_usd: float = 0.0
    liq_long_usd: float = 0.0
    liq_short_usd: float = 0.0


# ── Global state ────────────────────────────────────────────────────────────

_symbols: dict[str, SymbolState] = {}
_anomalies: deque = deque(maxlen=50)
_anomaly_cooldown: dict = {}
_connected: bool = False
_anomaly_callbacks: list = []
_stop_event: asyncio.Event | None = None

# Backward-compatible alias — points to BTC state
state: SymbolState = SymbolState(symbol="btcusdt")


def _get_sym(symbol: str) -> SymbolState:
    """Get or create per-symbol state."""
    if symbol not in _symbols:
        _symbols[symbol] = SymbolState(symbol=symbol)
    return _symbols[symbol]


def on_anomaly(callback):
    """Register an async callback for anomaly events: async def cb(event: AnomalyEvent)"""
    _anomaly_callbacks.append(callback)


def get_realtime_price(symbol: str = "btcusdt") -> float:
    """Get the latest WebSocket price (0.0 if not connected)."""
    return _get_sym(symbol).price


def get_realtime_state(symbol: str = "btcusdt") -> dict:
    """Snapshot of current real-time state for a symbol."""
    ss = _get_sym(symbol)

    # Compute rolling price change over crash window
    _prune_price_history(ss)
    price_5m_ago = _get_price_n_seconds_ago(ss, CRASH_WINDOW_SEC)
    change_5m = 0.0
    if price_5m_ago and ss.price:
        change_5m = (ss.price / price_5m_ago - 1) * 100

    # Trade flow imbalance
    _prune_trades(ss)
    total_flow = ss.recent_buys_usd + ss.recent_sells_usd
    buy_pct = (ss.recent_buys_usd / total_flow * 100) if total_flow > 0 else 50

    # Volume average
    avg_vol = _avg_1m_volume(ss)

    return {
        "symbol": symbol,
        "price": ss.price,
        "price_change_5m_pct": round(change_5m, 3),
        "volume_24h": ss.volume_24h,
        "price_change_24h_pct": ss.price_change_24h_pct,
        "buy_pressure_pct": round(buy_pct, 1),
        "trade_count_1m": ss.trade_count_1m,
        "avg_volume_1m_usd": round(avg_vol, 0),
        "current_volume_1m_usd": round(ss.current_1m_volume_usd, 0),
        "liq_total_5m_usd": round(ss.liq_total_5m_usd, 0),
        "liq_long_usd": round(ss.liq_long_usd, 0),
        "liq_short_usd": round(ss.liq_short_usd, 0),
        "connected": _connected,
        "last_update": ss.last_update,
        "recent_anomalies": [
            {
                "type": a.event_type, "severity": a.severity,
                "symbol": a.symbol, "change_pct": a.change_pct,
                "detail": a.detail, "timestamp": a.timestamp,
            }
            for a in list(_anomalies)[-5:]
            if a.symbol == symbol
        ],
    }


def get_ws_context(symbol: str = "btcusdt") -> str:
    """Format real-time WebSocket data for Claude's prompt."""
    ss = _get_sym(symbol)
    if not _connected or ss.price == 0:
        return ""

    label = symbol.replace("usdt", "").upper()
    s = get_realtime_state(symbol)
    lines = [f"REAL-TIME STREAM — {label} (WebSocket, sub-second):"]
    lines.append(f"  Live price:    ${s['price']:,.2f}")
    lines.append(f"  5m change:     {s['price_change_5m_pct']:+.2f}%")
    lines.append(f"  Buy pressure:  {s['buy_pressure_pct']:.0f}% (1m window)")

    avg_vol = s["avg_volume_1m_usd"]
    cur_vol = s["current_volume_1m_usd"]
    if avg_vol > 0:
        vol_ratio = cur_vol / avg_vol
        vol_label = "SPIKE" if vol_ratio > 3 else "elevated" if vol_ratio > 1.5 else "normal"
        lines.append(f"  Volume 1m:     ${cur_vol:,.0f} ({vol_ratio:.1f}x avg — {vol_label})")

    if s["liq_total_5m_usd"] > 0:
        lines.append(
            f"  Liquidations:  ${s['liq_total_5m_usd']:,.0f} (5m) — "
            f"longs ${s['liq_long_usd']:,.0f} / shorts ${s['liq_short_usd']:,.0f}"
        )

    anomalies = s["recent_anomalies"]
    if anomalies:
        latest = anomalies[-1]
        age = time.time() - latest["timestamp"]
        if age < 600:
            lines.append(
                f"  ALERT:         {latest['type'].replace('_', ' ').upper()} "
                f"({latest['change_pct']:+.1f}%) — {latest['detail']}"
            )

    return "\n".join(lines)


# ── Internal helpers ────────────────────────────────────────────────────────

def _prune_price_history(ss: SymbolState):
    """Remove entries older than crash window + buffer."""
    cutoff = time.time() - CRASH_WINDOW_SEC - 30
    while ss.price_history and ss.price_history[0][0] < cutoff:
        ss.price_history.popleft()


def _get_price_n_seconds_ago(ss: SymbolState, seconds: int) -> float | None:
    """Get the price approximately N seconds ago from the rolling buffer."""
    target_time = time.time() - seconds
    best = None
    best_diff = float("inf")
    for ts, px in ss.price_history:
        diff = abs(ts - target_time)
        if diff < best_diff:
            best_diff = diff
            best = px
    return best if best_diff < 30 else None


def _prune_trades(ss: SymbolState):
    """Remove trades older than 60 seconds from the rolling window."""
    cutoff = time.time() - 60
    while ss._trade_window and ss._trade_window[0][0] < cutoff:
        ts, usd, is_buy = ss._trade_window.popleft()
        if is_buy:
            ss.recent_buys_usd -= usd
        else:
            ss.recent_sells_usd -= usd
        ss.trade_count_1m -= 1

    ss.recent_buys_usd = max(0, ss.recent_buys_usd)
    ss.recent_sells_usd = max(0, ss.recent_sells_usd)
    ss.trade_count_1m = max(0, ss.trade_count_1m)


def _avg_1m_volume(ss: SymbolState) -> float:
    """Average 1-minute volume over the last hour."""
    vols = list(ss._volume_1m_history)
    return sum(vols) / len(vols) if vols else 0


def _rotate_volume_bucket(ss: SymbolState):
    """Rotate the current 1-minute volume bucket."""
    now = time.time()
    if now - ss._current_1m_start >= 60:
        ss._volume_1m_history.append(ss.current_1m_volume_usd)
        ss.current_1m_volume_usd = 0.0
        ss._current_1m_start = now


def _prune_liquidations(ss: SymbolState):
    """Remove liquidation events older than 5 minutes."""
    cutoff = time.time() - 300
    while ss._liq_window and ss._liq_window[0][0] < cutoff:
        ts, usd, side = ss._liq_window.popleft()
        ss.liq_total_5m_usd -= usd
        if side == "long":
            ss.liq_long_usd -= usd
        else:
            ss.liq_short_usd -= usd

    ss.liq_total_5m_usd = max(0, ss.liq_total_5m_usd)
    ss.liq_long_usd = max(0, ss.liq_long_usd)
    ss.liq_short_usd = max(0, ss.liq_short_usd)


async def _fire_anomaly(event: AnomalyEvent):
    """Record anomaly and fire callbacks (with cooldown to avoid spam)."""
    now = time.time()
    cooldown_key = f"{event.symbol}_{event.event_type}"
    last_fired = _anomaly_cooldown.get(cooldown_key, 0)

    # 5-minute cooldown per event type per symbol (2 min for critical)
    cooldown_sec = 120 if event.severity == "critical" else 300
    if now - last_fired < cooldown_sec:
        return

    _anomaly_cooldown[cooldown_key] = now
    _anomalies.append(event)

    logger.warning(
        "ANOMALY: %s [%s] price=$%.0f change=%+.2f%% — %s",
        event.event_type, event.severity, event.price,
        event.change_pct, event.detail,
    )

    for cb in _anomaly_callbacks:
        try:
            await cb(event)
        except Exception as exc:
            logger.error("Anomaly callback error: %s", exc)


# ── Anomaly detection engine ────────────────────────────────────────────────

async def _check_anomalies():
    """Run all anomaly detectors against all tracked symbols."""
    for sym, ss in _symbols.items():
        if ss.price == 0:
            continue

        label = sym.replace("usdt", "").upper()

        # 1. Flash crash detection
        _prune_price_history(ss)
        price_5m_ago = _get_price_n_seconds_ago(ss, CRASH_WINDOW_SEC)
        if price_5m_ago and price_5m_ago > 0:
            change_pct = (ss.price / price_5m_ago - 1) * 100

            if change_pct <= -CRASH_THRESHOLD_PCT:
                await _fire_anomaly(AnomalyEvent(
                    event_type="flash_crash",
                    severity="critical",
                    symbol=sym,
                    price=ss.price,
                    change_pct=change_pct,
                    detail=f"{label} dropped {change_pct:.1f}% in {CRASH_WINDOW_SEC}s "
                           f"(${price_5m_ago:,.0f} → ${ss.price:,.0f})",
                ))

            elif change_pct >= BREAKOUT_THRESHOLD_PCT:
                await _fire_anomaly(AnomalyEvent(
                    event_type="breakout",
                    severity="warning",
                    symbol=sym,
                    price=ss.price,
                    change_pct=change_pct,
                    detail=f"{label} surged {change_pct:+.1f}% in {CRASH_WINDOW_SEC}s "
                           f"(${price_5m_ago:,.0f} → ${ss.price:,.0f})",
                ))

        # 3. Volume spike detection
        avg_vol = _avg_1m_volume(ss)
        if avg_vol > 0 and ss.current_1m_volume_usd > avg_vol * VOLUME_SPIKE_MULTIPLIER:
            ratio = ss.current_1m_volume_usd / avg_vol
            await _fire_anomaly(AnomalyEvent(
                event_type="volume_spike",
                severity="warning",
                symbol=sym,
                price=ss.price,
                change_pct=0,
                detail=f"{label} volume {ratio:.1f}x above average "
                       f"(${ss.current_1m_volume_usd:,.0f} vs avg ${avg_vol:,.0f})",
            ))

        # 4. Liquidation cascade detection
        _prune_liquidations(ss)
        if ss.liq_total_5m_usd >= LIQUIDATION_THRESHOLD_USD:
            dominant = "longs" if ss.liq_long_usd > ss.liq_short_usd else "shorts"
            await _fire_anomaly(AnomalyEvent(
                event_type="liquidation_cascade",
                severity="critical",
                symbol=sym,
                price=ss.price,
                change_pct=0,
                detail=f"{label} ${ss.liq_total_5m_usd:,.0f} liquidated in 5m "
                       f"(mostly {dominant}: L=${ss.liq_long_usd:,.0f} / S=${ss.liq_short_usd:,.0f})",
            ))


# ── Message handlers ────────────────────────────────────────────────────────

def _handle_kline(ss: SymbolState, data: dict):
    """Process 1-minute kline update."""
    k = data.get("k", {})
    if not k:
        return
    price = float(k.get("c", 0))
    if price > 0:
        ss.price = price
        ss.last_update = time.time()
        ss.price_history.append((time.time(), price))


def _handle_agg_trade(ss: SymbolState, data: dict):
    """Process aggregate trade — track volume and trade flow."""
    price = float(data.get("p", 0))
    qty = float(data.get("q", 0))
    is_buyer_maker = data.get("m", False)

    if price <= 0 or qty <= 0:
        return

    usd_val = price * qty
    is_buy = not is_buyer_maker
    now = time.time()

    ss.price = price
    ss.last_update = now
    ss.price_history.append((now, price))

    ss._trade_window.append((now, usd_val, is_buy))
    if is_buy:
        ss.recent_buys_usd += usd_val
    else:
        ss.recent_sells_usd += usd_val
    ss.trade_count_1m += 1

    ss.current_1m_volume_usd += usd_val
    _rotate_volume_bucket(ss)


def _handle_mini_ticker(ss: SymbolState, data: dict):
    """Process 24h mini-ticker stats."""
    ss.volume_24h = float(data.get("v", 0))
    price = float(data.get("c", 0))
    open_price = float(data.get("o", 0))
    if price > 0:
        ss.price = price
        ss.last_update = time.time()
    if open_price > 0 and price > 0:
        ss.price_change_24h_pct = round((price / open_price - 1) * 100, 2)


def _handle_force_order(ss: SymbolState, data: dict):
    """Process futures liquidation event."""
    order = data.get("o", {})
    if not order:
        return

    price = float(order.get("p", 0))
    qty = float(order.get("q", 0))
    side = order.get("S", "").upper()

    if price <= 0 or qty <= 0:
        return

    usd_val = price * qty
    now = time.time()

    liq_side = "long" if side == "SELL" else "short"

    ss._liq_window.append((now, usd_val, liq_side))
    ss.liq_total_5m_usd += usd_val
    if liq_side == "long":
        ss.liq_long_usd += usd_val
    else:
        ss.liq_short_usd += usd_val


# ── Stream dispatcher ───────────────────────────────────────────────────────

_EVENT_HANDLERS = {
    "kline": _handle_kline,
    "aggTrade": _handle_agg_trade,
    "24hrMiniTicker": _handle_mini_ticker,
    "forceOrder": _handle_force_order,
}


def _dispatch(msg: dict):
    """Route a WebSocket message to the correct per-symbol handler."""
    # Combined stream format: {"stream": "btcusdt@kline_1m", "data": {...}}
    stream_name = msg.get("stream", "")
    data = msg.get("data", msg)
    event_type = data.get("e", "")

    handler = _EVENT_HANDLERS.get(event_type)
    if not handler:
        return

    # Extract symbol from stream name (e.g., "btcusdt@kline_1m" → "btcusdt")
    symbol = data.get("s", "").lower()
    if not symbol and "@" in stream_name:
        symbol = stream_name.split("@")[0]
    if not symbol:
        return

    ss = _get_sym(symbol)
    handler(ss, data)

    # Keep backward-compatible `state` pointing at BTC
    global state
    if symbol == "btcusdt":
        state = ss


# ── WebSocket connection manager ────────────────────────────────────────────

async def _connect_stream(url: str, streams: list[str], name: str):
    """Connect to a Binance WebSocket combined stream with auto-reconnect."""
    global _connected
    try:
        import websockets
    except ImportError:
        logger.warning("websockets not installed — %s stream disabled", name)
        return

    full_url = f"{url}?streams={'/'.join(streams)}"

    while not _stop_event.is_set():
        try:
            async with websockets.connect(
                full_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                _connected = True
                logger.info("WebSocket %s connected: %s", name, ", ".join(streams))

                async for raw in ws:
                    if _stop_event.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                        _dispatch(msg)
                    except json.JSONDecodeError:
                        continue

        except asyncio.CancelledError:
            break
        except Exception as exc:
            _connected = False
            logger.warning("WebSocket %s disconnected: %s — reconnecting in 5s", name, exc)
            try:
                await asyncio.wait_for(_stop_event.wait(), timeout=5)
                break
            except asyncio.TimeoutError:
                pass

    _connected = False
    logger.info("WebSocket %s stopped", name)


async def _anomaly_loop():
    """Run anomaly detection every 2 seconds across all symbols."""
    while not _stop_event.is_set():
        try:
            for ss in _symbols.values():
                _prune_trades(ss)
            await _check_anomalies()
        except Exception as exc:
            logger.debug("Anomaly check error: %s", exc)
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=2)
            break
        except asyncio.TimeoutError:
            pass


# ── Public lifecycle ────────────────────────────────────────────────────────

async def start():
    """Start all WebSocket streams and the anomaly detection loop."""
    global _stop_event, state
    _stop_event = asyncio.Event()

    # Initialize per-symbol state
    for sym in ("btcusdt", "ethusdt"):
        ss = _get_sym(sym)
        ss._current_1m_start = time.time()
    state = _get_sym("btcusdt")

    tasks = [
        asyncio.create_task(_connect_stream(SPOT_WS, SPOT_STREAMS, "spot")),
        asyncio.create_task(_connect_stream(FUTURES_WS, FUTURES_STREAMS, "futures")),
        asyncio.create_task(_anomaly_loop()),
    ]

    logger.info("WebSocket streams starting (spot + futures + anomaly detector)")
    return tasks


async def stop():
    """Gracefully stop all WebSocket streams."""
    global _connected
    if _stop_event:
        _stop_event.set()
    _connected = False
    logger.info("WebSocket streams stopping")
