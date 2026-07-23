"""
Trade executor with integrated risk management.

Flow:
  1. Receives Claude's decision (action + confidence)
  2. Risk manager calculates optimal size (Kelly + ATR + confidence)
  3. Executor validates safety checks (allocation cap, minimum balance)
  4. Places market order + OCO stop-loss/take-profit on Binance
  5. Returns detailed result with risk metadata

Code is the last line of defence — never blindly trust Claude's output.
"""
import logging
import ccxt
import config
import risk_manager

logger = logging.getLogger(__name__)


def _place_exit_orders(exchange: ccxt.binance, symbol: str, action: str,
                       qty: float, stop_loss: float, take_profit: float) -> dict:
    """
    Place stop-loss and take-profit exit orders after a market fill.
    After BUY → place SELL stop-loss + SELL take-profit limit.
    After SELL → place BUY stop-loss + BUY take-profit limit.
    Returns dict with order references (either may be None if it fails).
    """
    result = {"stop_loss_order": None, "take_profit_order": None}
    exit_side = "sell" if action == "buy" else "buy"
    base = symbol.split("/")[0]

    # Stop-loss
    try:
        sl_limit = _price_to_precision(exchange, symbol, stop_loss * (0.999 if action == "buy" else 1.001))
        result["stop_loss_order"] = exchange.create_order(
            symbol=symbol,
            type="STOP_LOSS_LIMIT",
            side=exit_side,
            amount=qty,
            price=sl_limit,
            params={"stopPrice": _price_to_precision(exchange, symbol, stop_loss), "timeInForce": "GTC"},
        )
        logger.info(
            "Stop-loss placed: %s %.6f %s trigger=$%,.0f limit=$%,.0f",
            exit_side.upper(), qty, base, stop_loss, sl_limit,
        )
    except Exception as exc:
        logger.warning("Stop-loss order failed (non-fatal): %s", exc)

    # Take-profit
    try:
        tp_price = _price_to_precision(exchange, symbol, take_profit)
        result["take_profit_order"] = exchange.create_order(
            symbol=symbol,
            type="TAKE_PROFIT_LIMIT",
            side=exit_side,
            amount=qty,
            price=tp_price,
            params={"stopPrice": tp_price, "timeInForce": "GTC"},
        )
        logger.info(
            "Take-profit placed: %s %.6f %s @ $%,.0f",
            exit_side.upper(), qty, base, take_profit,
        )
    except Exception as exc:
        logger.warning("Take-profit order failed (non-fatal): %s", exc)

    return result


def _amount_to_precision(exchange: ccxt.binance, symbol: str, qty: float) -> float:
    """
    Round an order quantity to the exchange's LOT_SIZE step for this symbol.
    ccxt normalizes the step per market; hardcoded rounding breaks the moment
    a new asset with a different step is added. Falls back to the raw qty if
    the market isn't loaded.
    """
    try:
        return float(exchange.amount_to_precision(symbol, qty))
    except Exception:
        return qty


def _price_to_precision(exchange: ccxt.binance, symbol: str, price: float) -> float:
    """Round a price to the exchange's PRICE_FILTER tick for this symbol."""
    try:
        return float(exchange.price_to_precision(symbol, price))
    except Exception:
        return round(price, 2)


def _min_notional(exchange: ccxt.binance, symbol: str) -> float:
    """
    The exchange's real minimum order value (price * qty) for this symbol,
    read from ccxt's normalized market limits. Binance rejects any order
    below this with a NOTIONAL filter error — risk-managed sizing (Kelly
    /ATR/confidence, or a circuit-breaker halving) can shrink well below
    it, so this must be checked at order time, not assumed from config.
    Falls back to a conservative constant if markets aren't loaded yet.
    """
    try:
        cost_min = exchange.markets.get(symbol, {}).get("limits", {}).get("cost", {}).get("min")
        if cost_min:
            return float(cost_min)
    except Exception:
        pass
    return config.EXCHANGE_MIN_NOTIONAL_FALLBACK


def _sized_amount(action: str, decision: dict, recommended_usd: float,
                  size_scale: float = 1.0) -> tuple[float, bool]:
    """
    Final executed size from the risk-managed recommendation.

    - size_scale: circuit-breaker multiplier (0.5 in drawdowns). Previously
      main.py only scaled the advisory decision["trade_usd"], which execute()
      ignores — so drawdown halving never affected real orders.
    - Validated ML gate buys earn GATE_TRADE_MULT up to MAX_GATE_TRADE_USD
      (alpha-sleeve tier; evidence basis in ROADMAP.md Phase 0 exit).
    Returns (amount_usd, gate_trade).
    """
    amount = recommended_usd * size_scale
    gate = False
    if action == "buy" and decision.get("ml_buy_signal"):
        amount = min(amount * config.GATE_TRADE_MULT, config.MAX_GATE_TRADE_USD)
        gate = True
    return max(config.MIN_TRADE_USD, round(amount, 2)), gate


def execute(exchange: ccxt.binance, decision: dict, snapshot: dict,
            portfolio: dict, size_scale: float = 1.0, dry_run: bool = False) -> dict:
    """
    Size and (unless dry_run) place a market order + bracket exits.

    dry_run=True runs the full sizing + gate pipeline and returns the exact
    order that WOULD be submitted (planned_usd / planned_qty), without
    touching the exchange. Used to show the true executed size in the LIVE
    confirmation prompt, so the number you approve matches what fills.
    """
    action = decision["action"]
    confidence = decision.get("confidence", 0.5)
    price = snapshot["price"]
    # Trade the analysed symbol — the ETH cycle passes ETH/USDT snapshots.
    # (portfolio["btc"] holds the base-asset amount for whichever symbol this is)
    symbol = snapshot.get("symbol") or config.SYMBOL
    base = symbol.split("/")[0]

    # Risk manager determines optimal position size and stop levels
    risk = risk_manager.assess_trade(action, confidence, snapshot, portfolio)

    # Risk-managed size, then circuit-breaker scale + alpha-sleeve gate tier
    amount, gate_trade = _sized_amount(action, decision, risk.recommended_usd, size_scale)
    if gate_trade:
        logger.info(
            "Alpha-sleeve: validated ML gate — size $%.2f (base $%.2f x%.1f, cap $%.0f)",
            amount, risk.recommended_usd, config.GATE_TRADE_MULT, config.MAX_GATE_TRADE_USD,
        )

    result = {
        "action": action,
        "amount_usd": 0.0,
        "btc_amount": 0.0,
        "order": None,
        "stop_order": None,
        "success": False,
        "error": None,
        "submitted": False,   # True once an order was actually sent to the exchange
        "planned_usd": 0.0,   # dry_run: the size that would be ordered
        "planned_qty": 0.0,   # dry_run: the base qty that would be ordered
        "risk_data": {
            "recommended_usd": risk.recommended_usd,
            "stop_loss": risk.stop_loss_price,
            "take_profit": risk.take_profit_price,
            "trailing_distance": risk.trailing_stop_distance,
            "risk_reward": risk.risk_reward_ratio,
            "kelly_fraction": risk.kelly_fraction,
            "atr_multiplier": risk.atr_multiplier,
            "rationale": risk.position_rationale,
        },
    }

    if action == "buy":
        # Bump to the exchange's real minimum notional BEFORE the balance
        # and allocation checks below, so those checks validate the size
        # that will actually be submitted — not the pre-bump one.
        min_notional = _min_notional(exchange, symbol)
        if amount < min_notional:
            bumped = round(min_notional * 1.01, 2)  # 1% buffer for price drift before fill
            logger.info(
                "BUY %s: sizing bumped $%.2f → $%.2f to clear exchange minimum notional ($%.2f)",
                base, amount, bumped, min_notional,
            )
            amount = bumped

        if portfolio["usdt"] < amount + 1.0:
            result["error"] = f"Low USDT: have ${portfolio['usdt']:.2f}, need ${amount:.2f}+fee"
            logger.warning(result["error"])
            return result

        btc_val = portfolio["btc"] * price
        total = portfolio["usdt"] + btc_val
        new_alloc = (btc_val + amount) / total if total > 0 else 1.0
        if new_alloc > config.MAX_BTC_ALLOC_PCT:
            result["error"] = f"Would exceed {config.MAX_BTC_ALLOC_PCT:.0%} BTC cap ({new_alloc:.1%})"
            logger.warning(result["error"])
            return result

        # Reject if risk/reward ratio is too low
        if risk.risk_reward_ratio > 0 and risk.risk_reward_ratio < 1.0:
            result["error"] = f"R:R too low ({risk.risk_reward_ratio:.1f}), need ≥1.0"
            logger.warning(result["error"])
            return result

        btc_qty = _amount_to_precision(exchange, symbol, amount / price)

        if dry_run:
            result.update({"planned_usd": amount, "planned_qty": btc_qty, "success": True})
            return result

        try:
            result["submitted"] = True
            order = exchange.create_market_buy_order(symbol, btc_qty)
            # Prefer the exchange's actual fill over our pre-order estimate
            filled_qty = float(order.get("filled") or btc_qty)
            filled_usd = float(order.get("cost") or amount)
            result.update({
                "success": True, "amount_usd": filled_usd,
                "btc_amount": filled_qty, "order": order,
            })
            logger.info(
                "BUY %.6f %s for $%.2f @ $%,.0f | SL=$%,.0f TP=$%,.0f R:R=%.1f",
                filled_qty, base, filled_usd, price,
                risk.stop_loss_price, risk.take_profit_price, risk.risk_reward_ratio,
            )

            # Place stop-loss order on exchange (non-blocking)
            if risk.stop_loss_price > 0:
                exit_orders = _place_exit_orders(
                    exchange, symbol, action, filled_qty,
                    risk.stop_loss_price, risk.take_profit_price,
                )
                result["stop_order"] = exit_orders.get("stop_loss_order")
                result["tp_order"] = exit_orders.get("take_profit_order")

        except Exception as e:
            result["error"] = str(e)
            logger.error("BUY failed: %s", e)

    elif action == "sell":
        if portfolio["btc"] <= 0:
            result["error"] = f"No {base} to sell"
            return result

        # Risk-managed sell: use recommended amount or 10% of holdings, whichever is less
        holding_usd = portfolio["btc"] * price
        sell_usd = min(amount, holding_usd * 0.10)

        # A 10% trim can fall below the exchange's minimum notional even
        # when the recommended amount wouldn't — bump up to the minimum
        # (safe: it only means trimming a slightly larger slice), unless
        # the whole position is smaller than the minimum (dust — can't be
        # sold via a market order at all).
        min_notional = _min_notional(exchange, symbol)
        if sell_usd < min_notional:
            if holding_usd >= min_notional:
                # Cap at the full holding — never propose selling more
                # than is actually owned.
                bumped = min(round(min_notional * 1.01, 2), round(holding_usd, 2))
                logger.info(
                    "SELL %s: sizing bumped $%.2f → $%.2f (10%% trim was below exchange minimum $%.2f)",
                    base, sell_usd, bumped, min_notional,
                )
                sell_usd = bumped
            else:
                result["error"] = (
                    f"Position (${holding_usd:.2f}) below exchange minimum "
                    f"(${min_notional:.2f}) — too small to sell"
                )
                logger.warning(result["error"])
                return result

        btc_qty = _amount_to_precision(exchange, symbol, sell_usd / price)

        min_qty = (exchange.markets.get(symbol, {})
                   .get("limits", {}).get("amount", {}).get("min", 0.00001))
        if btc_qty < min_qty:
            result["error"] = f"Sell qty {btc_qty:.8f} below minimum {min_qty}"
            logger.warning(result["error"])
            return result

        if dry_run:
            result.update({"planned_usd": sell_usd, "planned_qty": btc_qty, "success": True})
            return result

        try:
            result["submitted"] = True
            order = exchange.create_market_sell_order(symbol, btc_qty)
            filled_qty = float(order.get("filled") or btc_qty)
            filled_usd = float(order.get("cost") or sell_usd)
            result.update({
                "success": True, "amount_usd": filled_usd,
                "btc_amount": filled_qty, "order": order,
            })
            logger.info(
                "SELL %.6f %s for $%.2f @ $%,.0f | SL=$%,.0f",
                filled_qty, base, filled_usd, price, risk.stop_loss_price,
            )

            if risk.stop_loss_price > 0:
                exit_orders = _place_exit_orders(
                    exchange, symbol, action, filled_qty,
                    risk.stop_loss_price, risk.take_profit_price,
                )
                result["stop_order"] = exit_orders.get("stop_loss_order")
                result["tp_order"] = exit_orders.get("take_profit_order")

        except Exception as e:
            result["error"] = str(e)
            logger.error("SELL failed: %s", e)

    else:  # hold
        result["success"] = True
        logger.info("HOLD — no trade | Risk assessment: %s", risk.position_rationale)

    return result
