"""
Claude-Powered BTC Trading Bot
================================
Claude (Haiku) analyses market data + world news every 4h and decides:
  buy / hold / sell  →  Python executes with hard safety limits.

Features:
  - MySQL persistent storage (trades, snapshots, lessons, reviews, research)
  - World news context: crypto, macro, gold headlines injected every cycle
  - Self-correction: evaluates past trades, generates lessons for future cycles
  - Historical context: last 5 decisions injected into Claude's prompt
  - Live-trade confirmation: LIVE mode requires Telegram ✅/❌ approval first
  - Weekly deep review: Claude Fable generates lessons from 7-day performance
  - NEW COIN RESEARCH: Claude Fable scores newly listed coins 0-100 for investment

Telegram commands (BTC trading):
  /start    — begin the trading loop
  /stop     — pause the bot
  /status   — portfolio snapshot
  /analyze  — trigger immediate analysis
  /history  — last 5 trades with outcomes
  /lessons  — lessons Claude has self-learned
  /review   — trigger a weekly deep-review now (Fable)

Telegram commands (coin research):
  /newcoins          — scan CoinGecko for new/trending coins, score top 3
  /research <symbol> — deep-dive research on any specific coin (Fable)
  /watchlist         — view your saved investment watchlist

  /help     — full command list
"""
import asyncio
import datetime
import json
import logging
import sys
import uuid as uuid_mod
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

import config
import market_data as md
import claude_analyzer
import executor
import database as db
import attribution
import self_correction
import weekly_review
import coin_researcher
import ml_signal
import ws_stream
import multi_asset
import analytics
import grid_dca
import rl_position
import coin_screener
import report_generator
import thesis_generator

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Global state ───────────────────────────────────────────────────────────
bot_active    = False
analysis_loop = None
exchange      = None
tg_app        = None
baseline_usd  = None
_last_ml_prediction = None   # (direction, price_at_prediction) for drift tracking
_last_ml_price      = None
_ws_tasks: list     = []     # WebSocket background tasks
_emergency_event: asyncio.Event | None = None  # set by anomaly detector to wake the loop

# Pending live-trade confirmations: conf_id → (asyncio.Event, {"approved": bool})
_pending: dict[str, tuple[asyncio.Event, dict]] = {}

# ── Circuit-breaker state ──────────────────────────────────────────────────
_session_peak_usd: float | None = None   # highest portfolio value since /start
_daily_start_usd:  float | None = None   # portfolio at start of current calendar day
_daily_date:       str          = ""     # ISO date when _daily_start_usd was recorded
_sizing_scale:     float        = 1.0    # position-size multiplier (halved at 10% drawdown)
_weekly_start_usd: float | None = None   # portfolio at start of current ISO week
_weekly_key:       str          = ""     # ISO year-week when _weekly_start_usd was recorded
_cycle_failures:   int          = 0      # consecutive failed analysis cycles

# Execution-health window: True/False for whether each recently SUBMITTED
# order actually filled. Gate-skips (Low USDT, allocation cap, R:R, dust)
# don't count — only orders that reached the exchange. A run of failures
# here means our sizing/precision is producing invalid orders, which no
# portfolio-value circuit breaker would ever catch (rejected orders don't
# move the portfolio). Reset on restart; re-accumulates within a few cycles.
_exec_window: list[bool] = []
EXEC_WINDOW_SIZE   = 6   # look back this many submitted orders
EXEC_FAIL_ALERT    = 3   # alert when this many of the window failed
_exec_alerted:     bool  = False  # de-dupe: alert once per failure streak


def _persist_halt(reason: str):
    """Record a durable halt so auto-start cannot resurrect trading past it."""
    try:
        db.set_state("halted", reason)
    except Exception as exc:
        logger.error("Failed to persist halt state: %s", exc)


async def _check_execution_health(result: dict, symbol: str):
    """
    Track whether SUBMITTED orders are actually filling, and alert on a run
    of exchange rejections. Only orders that reached the exchange count —
    gate-skips (insufficient balance, allocation cap, R:R, dust) are normal
    and ignored. This is the monitor that would have caught the two-week
    $0-notional failure within hours: rejected orders don't move the
    portfolio, so no value-based circuit breaker ever fires on them.
    """
    global _exec_window, _exec_alerted

    if not result.get("submitted"):
        return  # order never reached the exchange — not an execution outcome

    _exec_window.append(bool(result.get("success")))
    if len(_exec_window) > EXEC_WINDOW_SIZE:
        _exec_window = _exec_window[-EXEC_WINDOW_SIZE:]

    fails = _exec_window.count(False)

    if fails >= EXEC_FAIL_ALERT and not _exec_alerted:
        _exec_alerted = True
        err = result.get("error", "unknown")
        logger.error("Execution health: %d/%d recent orders REJECTED — %s",
                     fails, len(_exec_window), err)
        db.log_event("execution_failure",
                     f"{fails}/{len(_exec_window)} recent orders rejected",
                     "error", {"last_error": err, "symbol": symbol})
        await notify(
            f"🚨 *Execution health warning*\n"
            f"{fails} of the last {len(_exec_window)} submitted orders were "
            f"*rejected by the exchange* \\(not risk\\-gated skips\\)\\.\n"
            f"Latest: `{_esc(str(err)[:120])}`\n"
            f"Trades are being decided but not filling — check sizing/precision\\."
        )
    elif fails == 0:
        _exec_alerted = False  # clean streak — re-arm the alert


async def _warn_if_unprotected(result: dict, symbol: str):
    """
    Alert when a filled buy/sell left its position WITHOUT a stop-loss resting
    on the exchange. The executor tries to place a protective bracket but the
    exchange can reject it (e.g. PERCENT_PRICE_BY_SIDE, MAX_NUM_ALGO_ORDERS);
    that used to be swallowed as "non-fatal", silently leaving live positions
    unprotected. Surface it instead.
    """
    if not (result.get("success") and result.get("submitted")):
        return
    if result.get("action") not in ("buy", "sell"):
        return
    if result.get("stop_protected"):
        return
    sym = symbol.split("/")[0]
    logger.warning("%s %s filled but stop-loss NOT placed on exchange — position unprotected",
                   sym, result.get("action"))
    db.log_event("unprotected_position",
                 f"{sym} {result.get('action')} filled without an exchange stop-loss",
                 "warning", {"symbol": symbol})
    await notify(
        f"⚠️ *Position unprotected*\n"
        f"{sym} {result.get('action', '').upper()} filled, but the stop\\-loss "
        f"order was *rejected by the exchange*\\. The position has no resting "
        f"stop right now — check open orders\\."
    )


# ── Circuit-breaker helpers ────────────────────────────────────────────────

def _count_consecutive_losses() -> int:
    """Count trailing consecutive 'wrong' outcomes from the most recent trades."""
    trades = db.get_recent_trades(10)
    count = 0
    for t in trades:
        outcome = t.get("outcome")
        if outcome == "wrong":
            count += 1
        elif outcome == "correct":
            break
        # trades with no outcome yet (unevaluated) are skipped
    return count


async def _check_circuit_breakers(total: float) -> bool:
    """
    Evaluate all five circuit-breaker conditions against the current portfolio value.

    Returns True  → trading may proceed (sizing scale may be adjusted).
    Returns False → bot paused; run_cycle should return immediately.

    Mutates: bot_active, _sizing_scale, _session_peak_usd,
             _daily_start_usd, _daily_date
    """
    global bot_active, _session_peak_usd, _daily_start_usd, _daily_date, _sizing_scale
    global _weekly_start_usd, _weekly_key

    today = datetime.date.today().isoformat()

    # Reset daily baseline at the start of each calendar day
    if _daily_date != today:
        _daily_start_usd = total
        _daily_date = today
        logger.info("Circuit breakers: new day — daily baseline $%.2f", total)

    # Reset weekly baseline at the start of each ISO week
    iso = datetime.date.today().isocalendar()
    week_key = f"{iso[0]}-W{iso[1]:02d}"
    if _weekly_key != week_key:
        _weekly_start_usd = total
        _weekly_key = week_key
        logger.info("Circuit breakers: new week %s — weekly baseline $%.2f", week_key, total)

    # Track session peak for drawdown calculation
    if _session_peak_usd is None or total > _session_peak_usd:
        _session_peak_usd = total

    # 1. Daily loss gate — 3% intraday drop pauses trading for the rest of the day
    if _daily_start_usd and total < _daily_start_usd * (1 - config.DAILY_LOSS_HALT_PCT):
        daily_loss_pct = (_daily_start_usd - total) / _daily_start_usd * 100
        logger.warning("Circuit breaker: daily loss %.1f%% — pausing bot", daily_loss_pct)
        db.log_event("circuit_breaker", f"Daily loss {daily_loss_pct:.1f}%", "warning",
                     {"daily_start": _daily_start_usd, "current": total})
        bot_active = False
        _persist_halt(f"daily loss gate: -{daily_loss_pct:.1f}% intraday")
        await notify(
            f"⚠️ *Daily loss gate triggered*\n"
            f"Down {daily_loss_pct:.1f}% today "
            f"\\(${_daily_start_usd:.2f} → ${total:.2f}\\)\\.\n"
            f"Trading paused for the rest of today\\. Use /start tomorrow\\."
        )
        return False

    # 1b. Weekly loss gate — 6% drop from the week's start halts until review
    if _weekly_start_usd and total < _weekly_start_usd * (1 - config.WEEKLY_LOSS_HALT_PCT):
        weekly_loss_pct = (_weekly_start_usd - total) / _weekly_start_usd * 100
        logger.warning("Circuit breaker: weekly loss %.1f%% — halting bot", weekly_loss_pct)
        db.log_event("circuit_breaker", f"Weekly loss {weekly_loss_pct:.1f}%", "warning",
                     {"weekly_start": _weekly_start_usd, "current": total})
        bot_active = False
        _persist_halt(f"weekly loss gate: -{weekly_loss_pct:.1f}% this week")
        await notify(
            f"⛔ *Weekly loss gate triggered*\n"
            f"Down {weekly_loss_pct:.1f}% this week "
            f"\\(${_weekly_start_usd:.2f} → ${total:.2f}\\)\\.\n"
            f"Bot halted — review the strategy before /start\\."
        )
        return False

    # 2. Consecutive loss gate — 5 straight losses means strategy needs review
    consec = _count_consecutive_losses()
    if consec >= config.CONSECUTIVE_LOSS_HALT:
        logger.warning("Circuit breaker: %d consecutive losses — pausing bot", consec)
        db.log_event("circuit_breaker", f"Consecutive losses: {consec}", "warning")
        bot_active = False
        _persist_halt(f"consecutive loss gate: {consec} losses in a row")
        await notify(
            f"⚠️ *Consecutive loss gate triggered*\n"
            f"{consec} losses in a row\\.\n"
            f"Bot paused — review strategy, then /start to resume\\."
        )
        return False

    # 3 & 4. Drawdown gates — measured from session peak
    drawdown = (_session_peak_usd - total) / _session_peak_usd if _session_peak_usd else 0.0

    if drawdown >= config.DRAWDOWN_HALT_PCT:
        logger.warning("Circuit breaker: critical drawdown %.1f%% — halting bot", drawdown * 100)
        db.log_event("circuit_breaker", f"Critical drawdown {drawdown:.1%}", "warning",
                     {"peak": _session_peak_usd, "current": total})
        bot_active = False
        _persist_halt(f"critical drawdown halt: {drawdown:.1%} from session peak")
        await notify(
            f"⛔ *Critical drawdown halt*\n"
            f"{drawdown:.1%} drawdown from session peak "
            f"\\(${_session_peak_usd:.2f} → ${total:.2f}\\)\\.\n"
            f"Bot halted\\. Use /start to resume after reviewing\\."
        )
        return False

    if drawdown >= config.DRAWDOWN_REDUCE_PCT:
        if _sizing_scale != 0.5:
            _sizing_scale = 0.5
            logger.warning("Circuit breaker: %.1f%% drawdown — position sizing halved", drawdown * 100)
            db.log_event("circuit_breaker", f"Sizing halved — {drawdown:.1%} drawdown", "warning")
            await notify(
                f"⚠️ *Position sizing reduced 50%*\n"
                f"{drawdown:.1%} drawdown from session peak\\.\n"
                f"Trading continues at half\\-size until portfolio recovers\\."
            )
    else:
        # Recovered above the reduce threshold — restore full sizing
        if _sizing_scale != 1.0:
            _sizing_scale = 1.0
            logger.info("Circuit breakers: drawdown recovered — full sizing restored")

    # 5. Equity MA gate — below 20-snapshot rolling average means equity is trending down
    try:
        snaps = db.get_snapshots(limit=20)
        if len(snaps) >= 20:
            equity_ma = sum(s["total_usd"] for s in snaps) / len(snaps)
            if total < equity_ma * 0.99 and _sizing_scale == 1.0:
                _sizing_scale = 0.5
                logger.info(
                    "Circuit breaker: equity $%.2f below 20-snapshot MA $%.2f — sizing at 50%%",
                    total, equity_ma,
                )
                db.log_event("circuit_breaker",
                             f"Equity below MA: ${total:.2f} < ${equity_ma:.2f}", "info")
    except Exception:
        pass  # non-fatal — snapshots may not exist yet

    # Persist a risk snapshot so the dashboard can show live safety state
    try:
        db.set_state("risk_status", json.dumps({
            "total_usd":        round(total, 2),
            "session_peak_usd": round(_session_peak_usd or 0, 2),
            "daily_start_usd":  round(_daily_start_usd or 0, 2),
            "weekly_start_usd": round(_weekly_start_usd or 0, 2),
            "sizing_scale":     _sizing_scale,
            "drawdown_pct":     round(drawdown * 100, 2),
            "checked_at":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }))
    except Exception:
        pass

    return True


# ── WebSocket anomaly handler ──────────────────────────────────────────────

async def _on_anomaly(event: ws_stream.AnomalyEvent):
    """Called by WebSocket anomaly detector — alert via Telegram + wake the loop."""
    emoji = {
        "flash_crash": "🔻",
        "breakout": "🚀",
        "volume_spike": "📊",
        "liquidation_cascade": "💀",
    }.get(event.event_type, "⚠️")

    sym_label = event.symbol.replace("usdt", "").upper()
    abs_change = abs(event.change_pct)

    # Build context-aware explanation for each alert type
    if event.event_type == "volume_spike":
        if abs_change < 0.5:
            context = (
                "What this means: Unusually high trading volume detected but price isn't moving much. "
                "This is often normal — large institutional orders, OTC trades clearing, or market-maker activity. "
                "No action needed. The bot will factor this into its next analysis."
            )
            severity_label = "LOW PRIORITY"
        else:
            context = (
                "What this means: High volume WITH price movement — smart money may be positioning. "
                "The bot is monitoring this closely. If the move continues, the next analysis cycle will respond."
            )
            severity_label = "MONITOR"

    elif event.event_type == "flash_crash":
        context = (
            "What this means: Price dropped sharply in a short time. This could be a liquidation cascade, "
            "a whale dump, or panic selling. The bot is automatically running an emergency analysis right now "
            "to decide whether to act. Watch for the follow-up analysis message. "
            "DO NOT panic sell manually — let the bot analyze first."
        )
        severity_label = "CRITICAL — Bot analyzing now"

    elif event.event_type == "breakout":
        context = (
            "What this means: Price surged upward quickly. This could be a genuine breakout or a fake-out "
            "that reverses. The bot will analyze whether this momentum is sustainable in the next cycle. "
            "DO NOT FOMO buy manually — let the bot decide if the breakout is real."
        )
        severity_label = "WATCH — Don't chase"

    elif event.event_type == "liquidation_cascade":
        context = (
            "What this means: Large number of leveraged traders are being forced to sell/buy by exchanges. "
            "This creates a cascade effect — price typically keeps moving in the same direction. "
            "The bot is running an emergency analysis right now. This is the most significant alert type. "
            "Pay close attention to the follow-up analysis."
        )
        severity_label = "CRITICAL — Emergency analysis triggered"

    else:
        context = "Unusual market activity detected. The bot is monitoring the situation."
        severity_label = "INFO"

    message = (
        f"{emoji} {sym_label} — {event.event_type.replace('_', ' ').upper()}\n"
        f"Price: ${event.price:,.0f} | Change: {event.change_pct:+.1f}%\n"
        f"{event.detail}\n\n"
        f"[{severity_label}]\n"
        f"{context}"
    )

    await notify(message)

    # Critical events interrupt the 4h sleep → trigger immediate analysis
    if event.severity == "critical" and _emergency_event and bot_active:
        logger.info("Emergency wake: %s — triggering immediate analysis", event.event_type)
        _emergency_event.set()


# ── Live-trade confirmation flow ────────────────────────────────────────────

def _confirmation_message(decision: dict, snap: dict, port: dict,
                          preview: dict | None = None) -> str:
    price   = snap["price"]
    base    = snap.get("symbol", config.SYMBOL).split("/")[0]
    # Prefer the executor's dry-run size (what will actually fill) over
    # Claude's advisory trade_usd, which the executor overrides.
    if preview and preview.get("planned_usd"):
        exec_usd = preview["planned_usd"]
        exec_qty = preview["planned_qty"]
    else:
        exec_usd = decision["trade_usd"]
        exec_qty = exec_usd / price
    signals = ", ".join(decision.get("signals", [])) or "—"
    return (
        f"⏳ *LIVE TRADE — CONFIRM BEFORE EXECUTION*\n\n"
        f"Action:     *{decision['action'].upper()}*\n"
        f"Amount:     *${exec_usd:.2f}*  ({exec_qty:.6f} {base})\n"
        f"{base} price:  *${price:,}*\n"
        f"Confidence: {decision['confidence']:.0%}  \\|  Risk: {decision['risk']}\n\n"
        f"📊 Signals: `{signals}`\n"
        f"💬 _{decision['reason']}_\n\n"
        f"RSI {snap['rsi']}  \\|  F\\&G {snap['fear_greed']}/100 ({snap['fear_greed_lbl']})\n"
        f"Portfolio: ${port['usdt']:.2f} USDT  +  {port['btc']:.6f} BTC\n\n"
        f"⏰ Auto\\-expires in 5 minutes"
    )


async def request_confirmation(decision: dict, snap: dict, port: dict,
                               timeout: int = 300, preview: dict | None = None) -> bool:
    """
    Send a Telegram inline-button confirmation request.
    Waits up to `timeout` seconds. Returns True if approved, False otherwise.
    Only called when TESTNET=false. `preview` is the executor dry-run result,
    used to show the true executed size.
    """
    conf_id = str(uuid_mod.uuid4())
    event   = asyncio.Event()
    verdict = {"approved": False}
    _pending[conf_id] = (event, verdict)

    db.save_pending_confirmation(conf_id, decision, snap, port)

    msg_text = _confirmation_message(decision, snap, port, preview=preview)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Execute",  callback_data=f"approve_{conf_id}"),
        InlineKeyboardButton("❌ Cancel",   callback_data=f"reject_{conf_id}"),
    ]])

    sent = await tg_app.bot.send_message(
        chat_id=config.TELEGRAM_CHAT_ID,
        text=msg_text,
        parse_mode="MarkdownV2",
        reply_markup=keyboard,
    )

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        approved = verdict["approved"]
        suffix   = "\n\n✅ *Executing order\\.\\.\\.*" if approved else "\n\n❌ *Order cancelled*"
        await tg_app.bot.edit_message_text(
            chat_id=config.TELEGRAM_CHAT_ID,
            message_id=sent.message_id,
            text=msg_text + suffix,
            parse_mode="MarkdownV2",
        )
        return approved

    except asyncio.TimeoutError:
        db.update_confirmation_status(conf_id, "expired")
        await tg_app.bot.edit_message_text(
            chat_id=config.TELEGRAM_CHAT_ID,
            message_id=sent.message_id,
            text=msg_text + "\n\n⏰ *Expired — trade skipped*",
            parse_mode="MarkdownV2",
        )
        return False

    finally:
        _pending.pop(conf_id, None)


async def handle_callback(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """
    Handle inline button callbacks:
      approve_<uuid>  / reject_<uuid>   — live trade confirmations
      wl_<SYM>_<db_id>_<price>          — add coin to watchlist
    """
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data:
        return

    # ── Watchlist button ──────────────────────────────────────────────────
    if data.startswith("wl_"):
        parts = data.split("_", 3)   # wl, symbol, db_id, price
        if len(parts) >= 3:
            _, symbol, db_id, *price_parts = parts
            price = float(price_parts[0]) if price_parts else 0.0
            # Look up coin details from the research record
            try:
                row = db.get_research_by_id(int(db_id)) or {}
                coin_researcher.add_to_watchlist(
                    row.get("coin_id", symbol.lower()),
                    symbol,
                    row.get("name", symbol),
                    price,
                    row.get("suggested_usd", 0),
                    db_id,
                )
                await query.edit_message_reply_markup(None)
                await query.message.reply_text(
                    f"📌 *{symbol}* added to watchlist\\!  Use /watchlist to view\\.",
                    parse_mode="MarkdownV2",
                )
            except Exception as exc:
                logger.error("Watchlist callback error: %s", exc)
        return

    # ── Trade confirmation buttons ────────────────────────────────────────
    if "_" not in data:
        return

    action, conf_id = data.split("_", 1)
    if conf_id not in _pending:
        try:
            await query.edit_message_reply_markup(None)
        except Exception:
            pass
        return

    event, verdict = _pending[conf_id]
    verdict["approved"] = (action == "approve")
    db.update_confirmation_status(
        conf_id, "approved" if verdict["approved"] else "rejected"
    )
    event.set()


# ── Core cycle ─────────────────────────────────────────────────────────────

async def run_cycle():
    """One full analysis → decision → (confirm) → trade cycle."""
    global baseline_usd, bot_active, _last_ml_prediction, _last_ml_price

    logger.info("── analysis cycle start ──")
    db.log_event("cycle_start", "Analysis cycle started")
    try:
        snap = md.get_market_snapshot(exchange)
        port = md.get_portfolio(exchange)

        # Evaluate previous ML prediction for drift detection
        if _last_ml_prediction and _last_ml_price:
            ml_signal.log_prediction_outcome(
                _last_ml_prediction, _last_ml_price, snap["price"]
            )

        # Evaluate outcomes of unevaluated decisions incl. holds (self-correction + RL)
        lessons = self_correction.evaluate_and_learn(current_price=snap["price"])
        attribution.persist_scoreboard()  # refresh Phase-2 scoreboard for API/dashboard
        for lesson in lessons:
            db.log_event("lesson", lesson, data={"source": "self_correction"})
        if lessons:
            joined = "\n".join(f"_{_esc(lesson)}_" for lesson in lessons[:3])
            await notify(
                f"🧠 *Lesson(s) learned from recent decisions:*\n{joined}"
            )
            # Feed RL agent with trade outcome reward
            try:
                last_trade = db.get_recent_trades(1)
                if last_trade and last_trade[0].get("outcome"):
                    t = last_trade[0]
                    entry_px = t.get("price", snap["price"])
                    pnl_pct = (snap["price"] / entry_px - 1) * 100 if entry_px > 0 else 0
                    reward = rl_position.compute_reward("hold", pnl_pct)
                    rl_position.learn(reward, snap, port, entry_px)
            except Exception:
                pass

        total = db.save_snapshot(port["usdt"], port["btc"], snap["price"])
        logger.info(
            "BTC $%,.0f | RSI %.0f | F&G %s | TF=%s(%d/3) | Portfolio $%.2f",
            snap["price"], snap["rsi"], snap["fear_greed"],
            snap.get("tf_direction", "?"), snap.get("tf_agreement", 0), total,
        )

        # Circuit-breaker checks (daily loss, consecutive losses, drawdown, equity MA)
        if not await _check_circuit_breakers(total):
            return

        # Stop-loss guard (legacy — kept for backward compatibility)
        if baseline_usd and total < baseline_usd * (1 - config.STOP_LOSS_PCT):
            msg = (
                f"⛔ *Stop\\-loss triggered\\!*\n"
                f"Started: ${baseline_usd:.2f} → Now: ${total:.2f} "
                f"\\({(total / baseline_usd - 1) * 100:.1f}%\\)\nBot paused\\."
            )
            logger.warning("Stop-loss hit. Pausing bot.")
            db.log_event("stop_loss", f"Portfolio dropped to ${total:.2f}", "warning",
                         {"baseline": baseline_usd, "current": total})
            bot_active = False
            await notify(msg)
            return

        # Grid/DCA management for sideways regime
        market_trend = snap.get("macd_trend", "")
        bb_width = (snap["bb_upper"] - snap["bb_lower"]) / snap["price"] * 100 if snap["price"] > 0 else 99
        is_sideways = (
            snap["rsi"] > 35 and snap["rsi"] < 65
            and bb_width < 6
            and snap.get("ichimoku_signal", "") == "inside_cloud_neutral"
        )

        if is_sideways and not grid_dca.is_grid_active():
            grid_plan = grid_dca.compute_grid_plan(snap, port)
            if grid_plan.strategy != "none":
                grid_result = grid_dca.execute_grid(exchange, grid_plan)
                logger.info("Grid activated: %s", grid_result)
                await notify(
                    f"📐 *{_esc(grid_plan.strategy.upper())} activated*\n"
                    f"_{_esc(grid_plan.rationale)}_"
                )
        elif not is_sideways and grid_dca.is_grid_active():
            cancelled = grid_dca.cancel_grid(exchange)
            logger.info("Grid cancelled — regime change (%d orders)", cancelled)
            await notify(f"📐 Grid cancelled — market no longer sideways \\({cancelled} orders\\)")

        # Ask Claude multi-agent pipeline (market + sentiment + ML → decision)
        decision = claude_analyzer.analyze(snap, port, exchange=exchange)

        # Timeframe consensus gate — block directional trades when < 2/3 timeframes agree
        tf_direction = snap.get("tf_direction", "mixed")
        tf_agreement = snap.get("tf_agreement", 0)
        if decision["action"] in ("buy", "sell") and tf_agreement < 2:
            # Only block if the agreed direction contradicts or doesn't support the trade
            action_dir = "bullish" if decision["action"] == "buy" else "bearish"
            if tf_direction != action_dir:
                original_action = decision["action"]
                decision["action"] = "hold"
                decision["reason"] = (
                    f"Timeframe gate: only {tf_agreement}/3 timeframes agree ({tf_direction}) "
                    f"— {original_action} blocked. Original: {decision['reason']}"
                )
                logger.info(
                    "Timeframe gate: blocked %s — %d/3 TFs agree, direction=%s",
                    original_action, tf_agreement, tf_direction,
                )
                db.log_event("timeframe_gate",
                             f"Blocked {original_action}: {tf_agreement}/3 agree, {tf_direction}",
                             data={"tf_1h": snap.get("tf_regime_1h"),
                                   "tf_4h": snap.get("tf_regime_4h"),
                                   "tf_1d": snap.get("tf_regime_1d")})

        # Apply circuit-breaker sizing scale (0.5 when in drawdown / equity below MA)
        if _sizing_scale < 1.0 and decision["action"] in ("buy", "sell"):
            original_usd = decision.get("trade_usd", 0)
            decision["trade_usd"] = max(
                config.MIN_TRADE_USD, round(original_usd * _sizing_scale, 2)
            )
            logger.info(
                "Sizing scale %.0f%% applied: $%.2f → $%.2f",
                _sizing_scale * 100, original_usd, decision["trade_usd"],
            )

        # Mirror-ready alert the moment a validated gate fires (independent of
        # whether caps/approval let the bot itself act on it)
        await _maybe_gate_alert(decision, snap)

        # Block buy if total crypto (BTC+ETH combined) is at the cap — the
        # executor's per-symbol cap cannot see the other asset's allocation
        if decision["action"] == "buy":
            combined = multi_asset.get_full_portfolio(exchange)
            if combined["crypto_alloc_pct"] >= config.MAX_TOTAL_CRYPTO_PCT * 100:
                decision["action"] = "hold"
                decision["reason"] = (
                    f"Total crypto at {combined['crypto_alloc_pct']:.0f}% "
                    f"(max {config.MAX_TOTAL_CRYPTO_PCT:.0%}) — was: {decision['reason']}"
                )
                logger.info("Total-crypto cap: BTC buy blocked at %.1f%%",
                            combined["crypto_alloc_pct"])

        # In LIVE mode, get human approval before any buy/sell — show the
        # ACTUAL executed size (risk-managed + min-notional bump), not
        # Claude's advisory trade_usd, so the number you approve matches
        # what fills.
        if not config.TESTNET and decision["action"] in ("buy", "sell"):
            preview = executor.execute(exchange, decision, snap, port,
                                       size_scale=_sizing_scale, dry_run=True)
            confirmed = await request_confirmation(decision, snap, port, preview=preview)
            if not confirmed:
                logger.info("Trade rejected/expired by user — skipping.")
                db.log_trade(
                    {"action": decision["action"], "amount_usd": 0,
                     "btc_amount": 0, "success": False, "error": "user_rejected"},
                    decision, snap,
                )
                return

        # Execute (executor has its own code-level safety checks; the
        # circuit-breaker sizing scale must reach the EXECUTED amount)
        result = executor.execute(exchange, decision, snap, port,
                                  size_scale=_sizing_scale)
        if result.get("risk_data"):
            decision["risk_data"] = result["risk_data"]
        await _check_execution_health(result, snap.get("symbol", config.SYMBOL))
        await _warn_if_unprotected(result, snap.get("symbol", config.SYMBOL))
        db.log_trade(result, decision, snap)
        db.log_event("trade", f"{decision['action'].upper()} ${decision['trade_usd']:.2f}",
                     data={"action": decision["action"], "amount": decision["trade_usd"],
                           "price": snap["price"], "success": result["success"],
                           "confidence": decision["confidence"],
                           "news_sentiment": decision.get("news_sentiment", "")})

        # Track ML prediction for drift detection next cycle
        _last_ml_prediction = decision.get("ml_prediction", None)
        _last_ml_price = snap["price"]

        # Notify result
        emoji = {"buy": "🟢", "hold": "⚪", "sell": "🔴"}.get(decision["action"], "⚪")
        mode  = "" if config.TESTNET else " 🔴 LIVE"
        news_sent = decision.get("news_sentiment", "")
        social_sent = decision.get("social_sentiment", "")
        market_assess = decision.get("market_assessment", "")
        ml_pred = decision.get("ml_prediction", "")
        ml_prob = decision.get("ml_probability")

        lines = [
            f"{emoji} *{decision['action'].upper()}*{_esc(mode)}  \\|  BTC ${snap['price']:,}",
            f"RSI {snap['rsi']}  \\|  F\\&G {snap['fear_greed']}/100 \\({_esc(snap['fear_greed_lbl'])}\\)",
            f"Portfolio: ${total:.2f}  \\|  Confidence {decision['confidence']:.0%}",
            f"💬 _{_esc(decision['reason'])}_",
        ]
        if market_assess:
            lines.append(f"📊 Market: {_esc(market_assess)}")
        if news_sent:
            lines.append(f"📰 News: {_esc(news_sent)}")
        if social_sent and social_sent != "no social data":
            lines.append(f"💬 Social: {_esc(social_sent)}")
        if ml_pred and ml_prob is not None:
            lines.append(f"🤖 ML: {_esc(ml_pred.upper())} \\({ml_prob:.0%} up prob\\)")
        fr = snap.get("funding_rate")
        if fr is not None:
            ls = snap.get("long_short_ratio", "?")
            lines.append(f"📈 Funding: {fr}%  \\|  L/S: {ls}")
        if result.get("error"):
            lines.append(f"⚠️ Skipped: {_esc(result['error'])}")
        try:
            perf = analytics.format_compact_report()
            lines.append(f"📊 {_esc(perf)}")
        except Exception:
            pass
        await notify("\n".join(lines))

    except Exception as exc:
        logger.exception("Cycle error")
        db.log_event("error", str(exc), "error")
        await notify("⚠️ Cycle error — check bot\\.log")


async def run_eth_cycle():
    """ETH analysis cycle — runs after BTC cycle each interval."""
    global bot_active

    # A circuit breaker or stop-loss may have halted the bot during the BTC
    # cycle — never trade ETH after a halt.
    if not bot_active:
        logger.info("ETH cycle skipped — bot halted during BTC cycle")
        return

    logger.info("── ETH analysis cycle start ──")
    try:
        eth_sym = "ETH/USDT"
        eth_cfg = config.ASSET_CONFIG[eth_sym]

        snap = multi_asset.get_asset_snapshot(exchange, eth_sym)
        full_port = multi_asset.get_full_portfolio(exchange)
        port_ctx = multi_asset.format_portfolio_context(full_port)

        # Check total crypto allocation before allowing ETH buy
        eth_holding = full_port["holdings"].get("ETH", {})
        eth_usd = eth_holding.get("usd_value", 0)
        eth_alloc = eth_usd / full_port["total_usd"] if full_port["total_usd"] > 0 else 0

        # Fear & Greed (shared with BTC — market-wide sentiment)
        fg = md.get_fear_greed()
        snap["fear_greed"] = fg["value"]
        snap["fear_greed_lbl"] = fg["label"]

        # Run Claude analysis for ETH (reuses the same multi-agent pipeline)
        eth_port = {
            "usdt": full_port["usdt"],
            "btc": eth_holding.get("amount", 0),  # mapped to "btc" key for compatibility
        }
        decision = claude_analyzer.analyze(snap, eth_port, exchange=exchange)

        # Timeframe consensus gate — same rule as the BTC cycle
        tf_direction = snap.get("tf_direction", "mixed")
        tf_agreement = snap.get("tf_agreement", 0)
        if decision["action"] in ("buy", "sell") and tf_agreement < 2:
            action_dir = "bullish" if decision["action"] == "buy" else "bearish"
            if tf_direction != action_dir:
                original_action = decision["action"]
                decision["action"] = "hold"
                decision["reason"] = (
                    f"Timeframe gate: only {tf_agreement}/3 timeframes agree ({tf_direction}) "
                    f"— {original_action} blocked. Original: {decision['reason']}"
                )
                logger.info(
                    "ETH timeframe gate: blocked %s — %d/3 TFs agree, direction=%s",
                    original_action, tf_agreement, tf_direction,
                )
                db.log_event("timeframe_gate",
                             f"ETH blocked {original_action}: {tf_agreement}/3 agree, {tf_direction}",
                             data={"symbol": eth_sym,
                                   "tf_1h": snap.get("tf_regime_1h"),
                                   "tf_4h": snap.get("tf_regime_4h"),
                                   "tf_1d": snap.get("tf_regime_1d")})

        # Override trade limits with ETH-specific config
        decision["trade_usd"] = max(
            eth_cfg["min_trade_usd"],
            min(eth_cfg["max_trade_usd"], float(decision.get("trade_usd", eth_cfg["base_trade_usd"]))),
        )

        # Apply circuit-breaker sizing scale (halved in drawdown — same as BTC)
        if _sizing_scale < 1.0 and decision["action"] in ("buy", "sell"):
            original_usd = decision["trade_usd"]
            decision["trade_usd"] = max(
                eth_cfg["min_trade_usd"], round(original_usd * _sizing_scale, 2)
            )
            logger.info(
                "ETH sizing scale %.0f%% applied: $%.2f → $%.2f",
                _sizing_scale * 100, original_usd, decision["trade_usd"],
            )

        # Mirror-ready alert the moment a validated gate fires
        await _maybe_gate_alert(decision, snap)

        # Block buy if ETH allocation already at max
        if decision["action"] == "buy" and eth_alloc >= eth_cfg["max_alloc_pct"]:
            decision["action"] = "hold"
            decision["reason"] = f"ETH already at {eth_alloc:.0%} (max {eth_cfg['max_alloc_pct']:.0%})"

        # Block buy if total crypto too high
        if decision["action"] == "buy" and full_port["crypto_alloc_pct"] >= config.MAX_TOTAL_CRYPTO_PCT * 100:
            decision["action"] = "hold"
            decision["reason"] = f"Total crypto at {full_port['crypto_alloc_pct']:.0f}% (max {config.MAX_TOTAL_CRYPTO_PCT:.0%})"

        # In LIVE mode, get approval — show the true executed size
        if not config.TESTNET and decision["action"] in ("buy", "sell"):
            preview = executor.execute(exchange, decision, snap, eth_port,
                                       size_scale=_sizing_scale, dry_run=True)
            confirmed = await request_confirmation(decision, snap, eth_port, preview=preview)
            if not confirmed:
                logger.info("ETH trade rejected/expired by user — skipping.")
                return

        # Execute using the ETH symbol (circuit-breaker scale applies here too)
        result = executor.execute(exchange, decision, snap, eth_port,
                                  size_scale=_sizing_scale)
        if result.get("risk_data"):
            decision["risk_data"] = result["risk_data"]   # was missing — ETH trades stored no stop/target
        await _check_execution_health(result, eth_sym)
        await _warn_if_unprotected(result, eth_sym)
        db.log_trade(result, decision, snap)
        db.log_event(
            "trade_eth",
            f"ETH {decision['action'].upper()} ${decision['trade_usd']:.2f}",
            data={"symbol": eth_sym, "action": decision["action"],
                  "amount": decision["trade_usd"], "price": snap["price"]},
        )

        # Notify
        emoji = {"buy": "🟢", "hold": "⚪", "sell": "🔴"}.get(decision["action"], "⚪")
        lines = [
            f"{emoji} *ETH {decision['action'].upper()}*  \\|  ETH ${snap['price']:,}",
            f"RSI {snap['rsi']}  \\|  Confidence {decision['confidence']:.0%}",
            f"💬 _{_esc(decision['reason'])}_",
        ]
        market_assess = decision.get("market_assessment", "")
        if market_assess:
            lines.append(f"📊 {_esc(market_assess)}")
        if result.get("error"):
            lines.append(f"⚠️ Skipped: {_esc(result['error'])}")
        await notify("\n".join(lines))

    except Exception as exc:
        logger.exception("ETH cycle error")
        await notify(f"⚠️ ETH cycle error — {_esc(str(exc)[:100])}")


async def _send_daily_briefing():
    """Send a plain-English morning briefing via Telegram."""
    try:
        snap = md.get_market_snapshot(exchange)
        port = md.get_portfolio(exchange)
        price = snap["price"]
        total = port["usdt"] + port["btc"] * price
        fg = snap.get("fear_greed", "?")
        fg_lbl = snap.get("fear_greed_lbl", "")
        rsi = snap.get("rsi", "?")

        # Yesterday's trades
        trades = db.get_recent_trades(20)
        yesterday_buys = len([t for t in trades if t["action"] == "buy" and t.get("success")])
        yesterday_sells = len([t for t in trades if t["action"] == "sell" and t.get("success")])
        yesterday_holds = len([t for t in trades if t["action"] == "hold"])

        # Build briefing
        lines = [
            f"Good morning! Here's your daily crypto briefing:\n",
            f"BTC: ${price:,.0f} | RSI: {rsi} | Fear & Greed: {fg}/100 ({fg_lbl})",
            f"Portfolio: ${total:,.2f}",
            f"",
            f"Yesterday: {yesterday_holds} holds, {yesterday_buys} buys, {yesterday_sells} sells",
        ]

        if yesterday_buys == 0 and yesterday_sells == 0:
            lines.append("The bot held cash all day — conditions weren't favorable for trading. This is disciplined behavior.")

        # Market mood context
        if fg != "?" and int(fg) <= 20:
            lines.append("\nMarket mood: Extreme Fear. Historically, extreme fear periods often precede recoveries — but timing is uncertain.")
        elif fg != "?" and int(fg) >= 75:
            lines.append("\nMarket mood: Greed. Prices may be overextended. The bot will be cautious about new entries.")

        lines.append("\nNothing requires your action. The bot is monitoring 24/7.")

        await notify("\n".join(lines))
        logger.info("Daily briefing sent")
    except Exception as exc:
        logger.debug("Daily briefing error: %s", exc)


async def _loop():
    """Periodic loop: run cycle → sleep → check weekly review → repeat."""
    global bot_active, _emergency_event, _cycle_failures
    _emergency_event = asyncio.Event()

    _last_briefing_date = ""

    while bot_active:
        _emergency_event.clear()

        # Daily morning briefing — once per day
        today_str = datetime.date.today().isoformat()
        if today_str != _last_briefing_date:
            _last_briefing_date = today_str
            try:
                await _send_daily_briefing()
            except Exception:
                logger.debug("Daily briefing failed")

        # Exchange-error circuit breaker: a cycle that raises must not kill
        # the loop silently (the process would look healthy while trading is
        # dead), and repeated failures mean the exchange/API is unhealthy —
        # halt durably rather than retry forever.
        try:
            await run_cycle()         # BTC analysis
            await run_eth_cycle()     # ETH analysis
            _cycle_failures = 0
        except Exception as exc:
            _cycle_failures += 1
            logger.error("Analysis cycle failed (%d/%d): %s",
                         _cycle_failures, config.CYCLE_FAILURE_HALT, exc, exc_info=True)
            db.log_event("cycle_error", str(exc)[:200], "error",
                         {"consecutive": _cycle_failures})
            if _cycle_failures >= config.CYCLE_FAILURE_HALT:
                bot_active = False
                _persist_halt(
                    f"cycle failure gate: {_cycle_failures} consecutive failures "
                    f"({str(exc)[:80]})"
                )
                await notify(
                    f"⛔ *Cycle failure halt*\n"
                    f"{_cycle_failures} consecutive analysis cycles failed\\.\n"
                    f"Last error: {_esc(str(exc)[:100])}\n"
                    f"Bot halted — investigate, then /start to resume\\."
                )
                break

        # Weekly deep-review check (Fable — runs ~once per week)
        if weekly_review.should_run():
            await notify("📊 Running weekly deep review with Claude Fable…")
            try:
                summary = weekly_review.run()
                await notify(
                    f"📊 *Weekly Review Complete*\n\n_{_esc(summary)}_"
                )
            except Exception:
                logger.exception("Weekly review error")
            try:
                await notify(_esc(attribution.report_text()))
            except Exception:
                logger.exception("Attribution report error")

        # ML model training/retraining (weekly, runs locally on VPS)
        if ml_signal.should_retrain():
            logger.info("ML model retraining triggered...")
            try:
                meta = ml_signal.train_model(exchange)
                if "error" not in meta:
                    await notify(
                        f"🤖 *ML Model Retrained*\n"
                        f"Samples: {meta.get('samples', 0):,}\n"
                        f"Features: {meta.get('n_features', 0)}\n"
                        f"CV Accuracy: {meta.get('cv_accuracy', 0):.1%}\n"
                        f"F1 Score: {meta.get('cv_f1', 0):.3f}"
                    )
                else:
                    logger.warning("ML training skipped: %s", meta["error"])
            except Exception:
                logger.exception("ML training error")

        # Interruptible sleep — wakes early on flash crash / liquidation cascade
        sleep_secs = config.ANALYSIS_INTERVAL_HOURS * 3600
        for _ in range(sleep_secs):
            if not bot_active:
                return
            if _emergency_event.is_set():
                logger.info("Emergency event — sleep interrupted for immediate analysis")
                await notify("⚡ *Emergency analysis triggered by anomaly detector*")
                break
            await asyncio.sleep(1)


# ── Telegram helpers ────────────────────────────────────────────────────────

def _gate_alert_text(decision: dict, snap: dict) -> str | None:
    """
    Mirror-ready gate alert: everything a human needs to place the same trade
    on a personal account within minutes. Returns None when no gate fired.

    Entry/target/stop use the identical ±PROFIT_TARGET_PCT barriers and
    LOOKAHEAD_HOURS time exit that produced the OOS validation record —
    the mirrored trade must be the trade that was validated, not a variant.
    """
    if decision.get("ml_buy_signal"):
        side = "BUY"
    elif decision.get("ml_sell_signal"):
        side = "SELL"
    else:
        return None

    price = float(snap["price"])
    symbol = snap.get("symbol") or config.SYMBOL
    prob = decision.get("ml_probability")
    pt = ml_signal.PROFIT_TARGET_PCT / 100
    sl = ml_signal.STOP_LOSS_PCT / 100
    target = price * (1 + pt) if side == "BUY" else price * (1 - pt)
    stop = price * (1 - sl) if side == "BUY" else price * (1 + sl)
    prob_txt = f"p={prob:.3f} cleared the validated gate" if prob is not None else "validated gate fired"

    return (
        f"🎯 GATE SIGNAL — {symbol} {side}\n"
        f"{prob_txt}\n\n"
        f"To mirror on a personal account (act within ~15 min, 1h-bar signal):\n"
        f"• Entry: market ≈ ${price:,.2f}\n"
        f"• Target: ${target:,.2f} ({'+' if side == 'BUY' else '-'}{ml_signal.PROFIT_TARGET_PCT}%)\n"
        f"• Stop: ${stop:,.2f} ({'-' if side == 'BUY' else '+'}{ml_signal.STOP_LOSS_PCT}%)\n"
        f"• Time exit: close after {ml_signal.LOOKAHEAD_HOURS}h if neither hits\n"
        f"• Suggested clip while verifying: $20–50\n\n"
        f"Basis: 123 OOS trades, 113W/10L. This signal extends the forward record."
    )


async def _maybe_gate_alert(decision: dict, snap: dict):
    """Send the mirror alert whenever a gate fires — even if the bot itself
    ends up capped, halted, or rejected: a human mirroring manually still
    wants the signal, and the event must enter the forward record."""
    text = _gate_alert_text(decision, snap)
    if not text:
        return
    symbol = snap.get("symbol") or config.SYMBOL
    side = "BUY" if decision.get("ml_buy_signal") else "SELL"
    db.log_event(
        "gate_signal",
        f"{symbol} {side} gate @ ${float(snap['price']):,.2f}",
        data={"symbol": symbol, "side": side, "price": snap["price"],
              "ml_probability": decision.get("ml_probability")},
    )
    await notify(_esc(text))


def _esc(text: str) -> str:
    """Escape MarkdownV2 special characters."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


async def notify(text: str):
    if tg_app and config.TELEGRAM_CHAT_ID:
        try:
            await tg_app.bot.send_message(
                chat_id=config.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="MarkdownV2",
            )
        except Exception:
            # MarkdownV2 failed — retry as plain text (strip formatting)
            try:
                plain = text.replace("\\", "")
                await tg_app.bot.send_message(
                    chat_id=config.TELEGRAM_CHAT_ID,
                    text=plain,
                )
            except Exception as e2:
                logger.error("Telegram notify error: %s", e2)


def _auth(update: Update) -> bool:
    return str(update.effective_chat.id) == config.TELEGRAM_CHAT_ID


# ── Telegram command handlers ───────────────────────────────────────────────

async def _start_trading() -> str:
    """
    Initialise circuit-breaker baselines, websocket streams and the analysis
    loop. Shared by the /start command and process-boot auto-start.
    Caller must ensure the loop is not already active.
    Returns the human-readable status message.
    """
    global bot_active, analysis_loop, baseline_usd, _ws_tasks
    global _session_peak_usd, _daily_start_usd, _daily_date, _sizing_scale

    port = md.get_portfolio(exchange)
    snap = md.get_market_snapshot(exchange)
    baseline_usd = port["usdt"] + port["btc"] * snap["price"]

    # Initialise circuit-breaker baselines for this session
    _session_peak_usd = baseline_usd
    _daily_start_usd  = baseline_usd
    _daily_date       = datetime.date.today().isoformat()
    _sizing_scale     = 1.0

    # Start WebSocket real-time streams + anomaly detection
    ws_stream.on_anomaly(_on_anomaly)
    _ws_tasks = await ws_stream.start()

    bot_active    = True
    analysis_loop = asyncio.create_task(_loop())

    logger.info("Bot started. Baseline: $%.2f", baseline_usd)
    db.log_event("bot_start", f"Bot started — baseline ${baseline_usd:.2f}",
                 data={"mode": "TESTNET" if config.TESTNET else "LIVE",
                       "baseline": baseline_usd})

    mode = "TESTNET" if config.TESTNET else "🔴 LIVE — trades need your approval"
    return (
        f"✅ Bot started ({mode})\n"
        f"Portfolio: ${baseline_usd:.2f}\n"
        f"Interval: {config.ANALYSIS_INTERVAL_HOURS}h | "
        f"Stop-loss: -{config.STOP_LOSS_PCT:.0%}\n"
        f"WebSocket: real-time price + anomaly detection active"
    )


async def _post_init(app: Application):
    """
    Auto-start the trading loop when the process boots.

    Without this, every systemd restart left the bot ALIVE BUT IDLE until a
    human sent /start in Telegram (observed live: 20h of zero cycles after a
    restart while telegram polling kept the process looking healthy).
    Opt out with BOT_AUTO_START=false.
    """
    import os
    if os.getenv("BOT_AUTO_START", "true").lower() not in ("1", "true", "yes"):
        logger.info("Auto-start disabled (BOT_AUTO_START) — waiting for /start")
        return
    if bot_active:
        return

    # A durable halt (circuit breaker or manual /stop) outranks auto-start —
    # a systemd restart must never resurrect trading past a risk halt.
    halt_reason = None
    try:
        halt_reason = db.get_state("halted")
    except Exception as exc:
        logger.warning("Could not read halt state: %s", exc)
    if halt_reason:
        logger.warning("Auto-start SUPPRESSED — durable halt active: %s", halt_reason)
        try:
            await app.bot.send_message(
                chat_id=config.TELEGRAM_CHAT_ID,
                text=(f"♻️ Bot process restarted, but trading stays HALTED:\n"
                      f"{halt_reason}\n\nSend /start to clear the halt and resume."),
            )
        except Exception:
            pass
        return

    text = await _start_trading()
    logger.info("Auto-started trading loop on process boot")
    try:
        await app.bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text="♻️ Bot process restarted — trading loop auto-started.\n\n" + text,
        )
    except Exception as exc:
        logger.warning("Auto-start Telegram notify failed: %s", exc)


async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    global _cycle_failures
    if not _auth(update):
        return
    if bot_active:
        await update.message.reply_text("Bot is already running.")
        return

    # /start is the explicit human override that clears any durable halt
    try:
        if db.get_state("halted"):
            db.clear_state("halted")
            logger.info("Durable halt cleared by /start")
    except Exception as exc:
        logger.warning("Could not clear halt state: %s", exc)
    _cycle_failures = 0

    text = await _start_trading()
    await update.message.reply_text(text)


async def cmd_stop(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    global bot_active, analysis_loop
    if not _auth(update):
        return
    bot_active = False
    _persist_halt("manual stop via /stop")  # survives systemd restarts
    if analysis_loop:
        analysis_loop.cancel()
    await ws_stream.stop()
    for t in _ws_tasks:
        t.cancel()
    await update.message.reply_text("⛔ Bot stopped (WebSocket streams closed).")
    logger.info("Bot stopped by user.")
    db.log_event("bot_stop", "Bot stopped by user")


async def cmd_status(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    try:
        port    = md.get_portfolio(exchange)
        snap    = md.get_market_snapshot(exchange)
        btc_val = port["btc"] * snap["price"]
        total   = port["usdt"] + btc_val
        alloc   = btc_val / total * 100 if total else 0
        bot_state = "🟢 Running" if bot_active else "⛔ Stopped"
        mode    = "TESTNET" if config.TESTNET else "🔴 LIVE"

        # Real-time WebSocket data
        rt_btc = ws_stream.get_realtime_state("btcusdt")
        rt_eth = ws_stream.get_realtime_state("ethusdt")
        ws_status = "🟢 WS" if rt_btc["connected"] else "⚪ WS off"
        ws_lines = [ws_status]
        if rt_btc["connected"] and rt_btc["price"] > 0:
            ws_lines.append(f"BTC ${rt_btc['price']:,.0f} ({rt_btc['price_change_5m_pct']:+.2f}% 5m)")
        if rt_eth["connected"] and rt_eth["price"] > 0:
            ws_lines.append(f"ETH ${rt_eth['price']:,.0f} ({rt_eth['price_change_5m_pct']:+.2f}% 5m)")

        await update.message.reply_text(
            f"📊 *Status: {bot_state} ({mode})*\n"
            f"BTC: ${snap['price']:,}  |  RSI {snap['rsi']}  |  F&G {snap['fear_greed']}\n"
            f"USDT: ${port['usdt']:.2f}\n"
            f"BTC:  {port['btc']:.6f} (${btc_val:.2f})\n"
            f"Total: ${total:.2f}  |  BTC alloc {alloc:.0f}%\n"
            f"{' | '.join(ws_lines)}",
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_analyze(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    await update.message.reply_text("🔍 Running one-off analysis…")
    await run_cycle()


async def cmd_history(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    rows = db.get_recent_trades(5)
    if not rows:
        await update.message.reply_text("No trades yet.")
        return
    lines = ["*Last 5 trades:*"]
    for row in rows:
        e = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}.get(row["action"], "⚪")
        outcome  = f" [{row['outcome']}]" if row.get("outcome") else ""
        status   = "✓" if row["success"] else f"✗ {row.get('error') or ''}"
        lines.append(
            f"{e} {row['created_at'][:10]}  "
            f"{row['action'].upper()}  "
            f"${row['amount_usd']:.2f}  "
            f"@${row.get('price', 0):,.0f}  "
            f"{status}{outcome}"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_lessons(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    lessons = db.get_active_lessons(10)
    if not lessons:
        await update.message.reply_text("No lessons learned yet. Keep trading!")
        return
    lines = ["*Active lessons Claude has learned:*"]
    for i, l in enumerate(lessons, 1):
        lines.append(f"{i}. {l}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_review(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    await update.message.reply_text("📊 Running deep review with Claude Fable (may take a few minutes)…")
    try:
        summary = weekly_review.run()
        await update.message.reply_text(
            f"📊 *Weekly Review*\n\n{summary}",
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(f"Review error: {e}")


async def cmd_newcoins(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """Scan CoinGecko for newly listed + trending coins; score top 3 with Fable."""
    if not _auth(update):
        return
    await update.message.reply_text(
        "🔍 Scanning new coins\\.\\.\\. this takes 60\\-90 seconds while Claude reads "
        "GitHub stats, market data, and community metrics for each candidate\\.",
        parse_mode="MarkdownV2",
    )

    loop = asyncio.get_event_loop()
    try:
        reports = await loop.run_in_executor(None, coin_researcher.scan_new_coins)
    except Exception as e:
        logger.exception("New coin scan error")
        await update.message.reply_text(f"Scan error: {e}")
        return

    summary = coin_researcher.format_scan_summary(reports)
    await update.message.reply_text(summary, parse_mode="MarkdownV2")

    # Send full report + watchlist button for each 'buy' or 'watch' verdict
    for r in reports:
        if r.get("verdict") in ("buy", "watch"):
            full = coin_researcher.format_deep_report(r)
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    "📌 Add to Watchlist",
                    callback_data=f"wl_{r['symbol']}_{r.get('_db_id', '')}_{r.get('price_usd', 0)}",
                )
            ]])
            try:
                await update.message.reply_text(
                    full, parse_mode="MarkdownV2", reply_markup=keyboard
                )
            except Exception:
                # Telegram has a 4096-char limit; send without markup on failure
                await update.message.reply_text(
                    full[:4000], parse_mode="MarkdownV2"
                )


async def cmd_research(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """Deep research on any coin by symbol or name: /research RNDR"""
    if not _auth(update):
        return
    args = (_ctx.args or [])
    if not args:
        await update.message.reply_text(
            "Usage: /research \\<symbol or name\\>\nExample: /research RNDR",
            parse_mode="MarkdownV2",
        )
        return

    query = " ".join(args).strip()
    await update.message.reply_text(
        f"🔬 Researching *{_esc(query)}* with Claude Fable \\(may take a few minutes\\)\\.\\.\\.",
        parse_mode="MarkdownV2",
    )

    loop = asyncio.get_event_loop()
    try:
        report = await loop.run_in_executor(
            None, coin_researcher.research_by_query, query
        )
    except Exception as e:
        logger.exception("Research error for %s", query)
        await update.message.reply_text(f"Research error: {e}")
        return

    if not report:
        await update.message.reply_text(
            f"❌ Coin '{_esc(query)}' not found on CoinGecko\\. "
            f"Try the full name or CoinGecko ID\\.",
            parse_mode="MarkdownV2",
        )
        return

    full = coin_researcher.format_deep_report(report)
    keyboard = None
    if report.get("verdict") in ("buy", "watch"):
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "📌 Add to Watchlist",
                callback_data=f"wl_{report['symbol']}_{report.get('_db_id', '')}_{report.get('price_usd', 0)}",
            )
        ]])

    try:
        await update.message.reply_text(
            full, parse_mode="MarkdownV2", reply_markup=keyboard
        )
    except Exception:
        await update.message.reply_text(full[:4000], parse_mode="MarkdownV2")


async def cmd_watchlist(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """Show saved investment watchlist."""
    if not _auth(update):
        return
    items = coin_researcher.get_watchlist()
    if not items:
        await update.message.reply_text(
            "Watchlist is empty\\. Use /newcoins or /research then tap *📌 Add to Watchlist*\\.",
            parse_mode="MarkdownV2",
        )
        return

    lines = ["📌 *Your Investment Watchlist*\n"]
    for item in items:
        lines.append(
            f"• *{_esc(item['symbol'])}* — {_esc(item['name'])}\n"
            f"  Entry price: ${_esc(str(item.get('entry_price', '?')))}\n"
            f"  Target invest: ${_esc(str(item.get('target_usd', '?')))}"
        )

    recent = coin_researcher.get_recent_research(5)
    if recent:
        lines.append("\n*Recent research:*")
        for r in recent:
            v = {"buy": "✅", "watch": "👀", "avoid": "❌"}.get(r.get("verdict", ""), "❓")
            wl = " 📌" if r.get("on_watchlist") else ""
            lines.append(
                f"  {v} {_esc(r['symbol'])} {r.get('investment_score', 0)}/100{_esc(wl)} "
                f"— {_esc(r.get('summary', '')[:60])}"
            )

    await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2")


async def cmd_performance(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """Show performance analytics — /performance or /performance 7 or /performance 30."""
    if not _auth(update):
        return
    args = _ctx.args or []
    days = None
    if args:
        try:
            days = int(args[0])
        except ValueError:
            pass

    try:
        report = analytics.format_report(days)
        await update.message.reply_text(report, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Analytics error: {e}")


async def cmd_screen(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """Scan top 50 coins and rank by momentum score."""
    if not _auth(update):
        return
    await update.message.reply_text("📊 Scanning top 50 coins by momentum...")
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, coin_screener.run_screening, 50)
        msg = coin_screener.format_screening_telegram(results)
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"Screening error: {e}")


async def cmd_report(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """Generate weekly market report with Claude Fable."""
    if not _auth(update):
        return
    args = _ctx.args or []
    days = 7
    if args:
        try:
            days = int(args[0])
        except ValueError:
            pass

    await update.message.reply_text(f"📊 Generating {days}-day market report with Claude Fable (may take a few minutes)...")
    loop = asyncio.get_event_loop()
    try:
        report = await loop.run_in_executor(None, report_generator.generate_report, days, "weekly" if days <= 7 else "monthly")
        if report.get("content"):
            content = report["content"]
            # Split if too long for Telegram
            if len(content) > 4000:
                await update.message.reply_text(content[:4000])
                await update.message.reply_text(content[4000:8000])
            else:
                await update.message.reply_text(content)
        else:
            await update.message.reply_text("Report generation failed — check logs.")
    except Exception as e:
        await update.message.reply_text(f"Report error: {e}")


async def cmd_thesis(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """Generate investment thesis for any coin: /thesis SOL or /thesis SOL 5000"""
    if not _auth(update):
        return
    args = _ctx.args or []
    if not args:
        await update.message.reply_text("Usage: /thesis <symbol> [portfolio_size]\nExample: /thesis SOL 10000")
        return

    query = args[0].strip()
    portfolio_size = 10000
    if len(args) > 1:
        try:
            portfolio_size = float(args[1])
        except ValueError:
            pass

    await update.message.reply_text(f"🔬 Generating investment thesis for *{query.upper()}* (portfolio: ${portfolio_size:,.0f})...\nThis uses Claude Fable and may take a few minutes.", parse_mode="Markdown")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, thesis_generator.generate_thesis, query, portfolio_size)
        if not result:
            await update.message.reply_text(f"Coin '{query}' not found on CoinGecko.")
            return
        if result.get("error"):
            await update.message.reply_text(f"Thesis error: {result['error']}")
            return

        thesis = result["thesis"]
        header = f"📋 *Investment Thesis — {result['symbol']}*\n{result['name']} | ${result.get('price', 0):,.4f} | Cap: ${(result.get('market_cap') or 0)/1e9:.1f}B\n\n"

        # Split if too long
        full = header + thesis
        if len(full) > 4000:
            await update.message.reply_text(full[:4000], parse_mode="Markdown")
            if len(thesis) > 3800:
                await update.message.reply_text(thesis[3800:7600])
        else:
            await update.message.reply_text(full, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Thesis error: {e}")


async def cmd_help(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    await update.message.reply_text(
        "📈 *BTC + ETH Trading*\n"
        "/start    — start the trading loop\n"
        "/stop     — pause the bot\n"
        "/status   — portfolio & market snapshot\n"
        "/analyze  — run an immediate analysis\n"
        "/history  — last 5 trades with outcomes\n"
        "/lessons  — lessons Claude has self-learned\n"
        "/review   — run weekly deep review now\n"
        "/performance — analytics (or /performance 7)\n\n"
        "🔍 *Coin Research*\n"
        "/newcoins          — scan & score new/trending coins\n"
        "/research <symbol> — deep-dive any specific coin\n"
        "/watchlist         — view saved investment watchlist\n\n"
        "📊 *Advisory Tools*\n"
        "/screen            — scan top 50 coins by momentum\n"
        "/report            — generate weekly market report\n"
        "/report 30         — generate 30-day report\n"
        "/thesis <symbol>   — full investment thesis\n"
        "/thesis SOL 5000   — thesis for $5K portfolio\n\n"
        "/help     — this message",
        parse_mode="Markdown",
    )


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    global exchange, tg_app

    missing = [
        k for k in (
            "ANTHROPIC_API_KEY", "BINANCE_API_KEY", "BINANCE_SECRET",
            "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID",
            "MYSQL_PASSWORD",
        )
        if not getattr(config, k, None)
    ]
    if missing:
        logger.error("Missing env vars: %s", missing)
        sys.exit(1)

    db.init()
    exchange = md.get_exchange()

    logger.info(
        "Exchange: Binance %s | Symbol: %s | Interval: %dh",
        "TESTNET" if config.TESTNET else "LIVE",
        config.SYMBOL, config.ANALYSIS_INTERVAL_HOURS,
    )
    if not config.TESTNET:
        logger.warning("LIVE MODE — every buy/sell will require Telegram confirmation")

    tg_app = Application.builder().token(config.TELEGRAM_TOKEN).post_init(_post_init).build()
    tg_app.add_handler(CommandHandler("start",   cmd_start))
    tg_app.add_handler(CommandHandler("stop",    cmd_stop))
    tg_app.add_handler(CommandHandler("status",  cmd_status))
    tg_app.add_handler(CommandHandler("analyze", cmd_analyze))
    tg_app.add_handler(CommandHandler("history", cmd_history))
    tg_app.add_handler(CommandHandler("lessons",   cmd_lessons))
    tg_app.add_handler(CommandHandler("review",    cmd_review))
    tg_app.add_handler(CommandHandler("performance", cmd_performance))
    # Coin research commands
    tg_app.add_handler(CommandHandler("newcoins",  cmd_newcoins))
    tg_app.add_handler(CommandHandler("research",  cmd_research))
    tg_app.add_handler(CommandHandler("watchlist", cmd_watchlist))
    # Advisory tools
    tg_app.add_handler(CommandHandler("screen",    cmd_screen))
    tg_app.add_handler(CommandHandler("report",    cmd_report))
    tg_app.add_handler(CommandHandler("thesis",    cmd_thesis))
    tg_app.add_handler(CommandHandler("help",      cmd_help))
    tg_app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("Telegram bot ready — trading loop auto-starts on boot "
                "(BOT_AUTO_START=false to require manual /start)")
    tg_app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
