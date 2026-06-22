"""
Claude-Powered BTC Trading Bot
================================
Claude (Haiku) analyses market data + world news every 4h and decides:
  buy / hold / sell  →  Python executes with hard safety limits.

Features:
  - Supabase persistent storage (trades, snapshots, lessons, reviews, research)
  - World news context: crypto, macro, gold headlines injected every cycle
  - Self-correction: evaluates past trades, generates lessons for future cycles
  - Historical context: last 5 decisions injected into Claude's prompt
  - Live-trade confirmation: LIVE mode requires Telegram ✅/❌ approval first
  - Weekly deep review: Claude Opus generates lessons from 7-day performance
  - NEW COIN RESEARCH: Claude Opus scores newly listed coins 0-100 for investment

Telegram commands (BTC trading):
  /start    — begin the trading loop
  /stop     — pause the bot
  /status   — portfolio snapshot
  /analyze  — trigger immediate analysis
  /history  — last 5 trades with outcomes
  /lessons  — lessons Claude has self-learned
  /review   — trigger a weekly deep-review now (Opus)

Telegram commands (coin research):
  /newcoins          — scan CoinGecko for new/trending coins, score top 3
  /research <symbol> — deep-dive research on any specific coin (Opus)
  /watchlist         — view your saved investment watchlist

  /help     — full command list
"""
import asyncio
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
import self_correction
import weekly_review
import coin_researcher
import ml_signal
import ws_stream
import multi_asset
import analytics
import grid_dca
import rl_position

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


# ── WebSocket anomaly handler ──────────────────────────────────────────────

async def _on_anomaly(event: ws_stream.AnomalyEvent):
    """Called by WebSocket anomaly detector — alert via Telegram + wake the loop."""
    emoji = {
        "flash_crash": "🔻",
        "breakout": "🚀",
        "volume_spike": "📊",
        "liquidation_cascade": "💀",
    }.get(event.event_type, "⚠️")

    sev = "🔴 CRITICAL" if event.severity == "critical" else "🟡 WARNING"
    sym_label = event.symbol.replace("usdt", "").upper()

    await notify(
        f"{emoji} *{sev}: {sym_label} {_esc(event.event_type.replace('_', ' ').upper())}*\n"
        f"Price: ${event.price:,.0f}  |  Change: {event.change_pct:+.1f}%\n"
        f"_{_esc(event.detail)}_"
    )

    # Critical events interrupt the 4h sleep → trigger immediate analysis
    if event.severity == "critical" and _emergency_event and bot_active:
        logger.info("Emergency wake: %s — triggering immediate analysis", event.event_type)
        _emergency_event.set()


# ── Live-trade confirmation flow ────────────────────────────────────────────

def _confirmation_message(decision: dict, snap: dict, port: dict) -> str:
    price   = snap["price"]
    btc_qty = decision["trade_usd"] / price
    signals = ", ".join(decision.get("signals", [])) or "—"
    return (
        f"⏳ *LIVE TRADE — CONFIRM BEFORE EXECUTION*\n\n"
        f"Action:     *{decision['action'].upper()}*\n"
        f"Amount:     *${decision['trade_usd']:.2f}*  ({btc_qty:.6f} BTC)\n"
        f"BTC price:  *${price:,}*\n"
        f"Confidence: {decision['confidence']:.0%}  |  Risk: {decision['risk']}\n\n"
        f"📊 Signals: `{signals}`\n"
        f"💬 _{decision['reason']}_\n\n"
        f"RSI {snap['rsi']}  |  F\\&G {snap['fear_greed']}/100 ({snap['fear_greed_lbl']})\n"
        f"Portfolio: ${port['usdt']:.2f} USDT  +  {port['btc']:.6f} BTC\n\n"
        f"⏰ Auto\\-expires in 5 minutes"
    )


async def request_confirmation(decision: dict, snap: dict, port: dict,
                               timeout: int = 300) -> bool:
    """
    Send a Telegram inline-button confirmation request.
    Waits up to `timeout` seconds. Returns True if approved, False otherwise.
    Only called when TESTNET=false.
    """
    conf_id = str(uuid_mod.uuid4())
    event   = asyncio.Event()
    verdict = {"approved": False}
    _pending[conf_id] = (event, verdict)

    db.save_pending_confirmation(conf_id, decision, snap, port)

    msg_text = _confirmation_message(decision, snap, port)
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

        # Evaluate outcome of the previous actionable trade (self-correction + RL)
        lesson = self_correction.evaluate_and_learn(current_price=snap["price"])
        if lesson:
            db.log_event("lesson", lesson, data={"source": "self_correction"})
            await notify(
                f"🧠 *Lesson learned from last trade:*\n_{_esc(lesson)}_"
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
            "BTC $%,.0f | RSI %.0f | F&G %s | Portfolio $%.2f",
            snap["price"], snap["rsi"], snap["fear_greed"], total,
        )

        # Stop-loss guard
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

        # In LIVE mode, get human approval before any buy/sell
        if not config.TESTNET and decision["action"] in ("buy", "sell"):
            confirmed = await request_confirmation(decision, snap, port)
            if not confirmed:
                logger.info("Trade rejected/expired by user — skipping.")
                db.log_trade(
                    {"action": decision["action"], "amount_usd": 0,
                     "btc_amount": 0, "success": False, "error": "user_rejected"},
                    decision, snap,
                )
                return

        # Execute (executor has its own code-level safety checks)
        result = executor.execute(exchange, decision, snap, port)
        if result.get("risk_data"):
            decision["risk_data"] = result["risk_data"]
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
            f"{emoji} *{decision['action'].upper()}*{_esc(mode)}  |  BTC ${snap['price']:,}",
            f"RSI {snap['rsi']}  |  F\\&G {snap['fear_greed']}/100 \\({_esc(snap['fear_greed_lbl'])}\\)",
            f"Portfolio: ${total:.2f}  |  Confidence {decision['confidence']:.0%}",
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
            lines.append(f"📈 Funding: {fr}%  |  L/S: {ls}")
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

        # Override trade limits with ETH-specific config
        decision["trade_usd"] = max(
            eth_cfg["min_trade_usd"],
            min(eth_cfg["max_trade_usd"], float(decision.get("trade_usd", eth_cfg["base_trade_usd"]))),
        )

        # Block buy if ETH allocation already at max
        if decision["action"] == "buy" and eth_alloc >= eth_cfg["max_alloc_pct"]:
            decision["action"] = "hold"
            decision["reason"] = f"ETH already at {eth_alloc:.0%} (max {eth_cfg['max_alloc_pct']:.0%})"

        # Block buy if total crypto too high
        if decision["action"] == "buy" and full_port["crypto_alloc_pct"] >= config.MAX_TOTAL_CRYPTO_PCT * 100:
            decision["action"] = "hold"
            decision["reason"] = f"Total crypto at {full_port['crypto_alloc_pct']:.0f}% (max {config.MAX_TOTAL_CRYPTO_PCT:.0%})"

        # In LIVE mode, get approval
        if not config.TESTNET and decision["action"] in ("buy", "sell"):
            confirmed = await request_confirmation(decision, snap, eth_port)
            if not confirmed:
                logger.info("ETH trade rejected/expired by user — skipping.")
                return

        # Execute using the ETH symbol
        result = executor.execute(exchange, decision, snap, eth_port)
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
            f"{emoji} *ETH {decision['action'].upper()}*  |  ETH ${snap['price']:,}",
            f"RSI {snap['rsi']}  |  Confidence {decision['confidence']:.0%}",
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


async def _loop():
    """Periodic loop: run cycle → sleep → check weekly review → repeat."""
    global bot_active, _emergency_event
    _emergency_event = asyncio.Event()

    while bot_active:
        _emergency_event.clear()
        await run_cycle()         # BTC analysis
        await run_eth_cycle()     # ETH analysis

        # Weekly deep-review check (Opus — runs ~once per week)
        if weekly_review.should_run():
            await notify("📊 Running weekly deep review with Claude Opus…")
            try:
                summary = weekly_review.run()
                await notify(
                    f"📊 *Weekly Review Complete*\n\n_{_esc(summary)}_"
                )
            except Exception:
                logger.exception("Weekly review error")

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
        except Exception as e:
            logger.error("Telegram notify error: %s", e)


def _auth(update: Update) -> bool:
    return str(update.effective_chat.id) == config.TELEGRAM_CHAT_ID


# ── Telegram command handlers ───────────────────────────────────────────────

async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    global bot_active, analysis_loop, baseline_usd, _ws_tasks
    if not _auth(update):
        return
    if bot_active:
        await update.message.reply_text("Bot is already running.")
        return

    port = md.get_portfolio(exchange)
    snap = md.get_market_snapshot(exchange)
    baseline_usd = port["usdt"] + port["btc"] * snap["price"]

    # Start WebSocket real-time streams + anomaly detection
    ws_stream.on_anomaly(_on_anomaly)
    _ws_tasks = await ws_stream.start()

    bot_active    = True
    analysis_loop = asyncio.create_task(_loop())

    mode = "TESTNET" if config.TESTNET else "🔴 LIVE — trades need your approval"
    await update.message.reply_text(
        f"✅ Bot started ({mode})\n"
        f"Portfolio: ${baseline_usd:.2f}\n"
        f"Interval: {config.ANALYSIS_INTERVAL_HOURS}h | "
        f"Stop-loss: -{config.STOP_LOSS_PCT:.0%}\n"
        f"WebSocket: real-time price + anomaly detection active"
    )
    logger.info("Bot started. Baseline: $%.2f", baseline_usd)
    db.log_event("bot_start", f"Bot started — baseline ${baseline_usd:.2f}",
                 data={"mode": "TESTNET" if config.TESTNET else "LIVE",
                       "baseline": baseline_usd})


async def cmd_stop(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    global bot_active, analysis_loop
    if not _auth(update):
        return
    bot_active = False
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
    await update.message.reply_text("📊 Running deep review with Claude Opus (may take ~30s)…")
    try:
        summary = weekly_review.run()
        await update.message.reply_text(
            f"📊 *Weekly Review*\n\n{summary}",
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(f"Review error: {e}")


async def cmd_newcoins(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    """Scan CoinGecko for newly listed + trending coins; score top 3 with Opus."""
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
        f"🔬 Researching *{_esc(query)}* with Claude Opus \\(~30 seconds\\)\\.\\.\\.",
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

    tg_app = Application.builder().token(config.TELEGRAM_TOKEN).build()
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
    tg_app.add_handler(CommandHandler("help",      cmd_help))
    tg_app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("Telegram bot ready — send /start to begin")
    tg_app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
