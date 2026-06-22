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


def _place_exit_orders(exchange: ccxt.binance, action: str, btc_qty: float,
                       stop_loss: float, take_profit: float) -> dict:
    """
    Place stop-loss and take-profit exit orders after a market fill.
    After BUY → place SELL stop-loss + SELL take-profit limit.
    After SELL → place BUY stop-loss + BUY take-profit limit.
    Returns dict with order references (either may be None if it fails).
    """
    result = {"stop_loss_order": None, "take_profit_order": None}
    exit_side = "sell" if action == "buy" else "buy"

    # Stop-loss
    try:
        sl_limit = round(stop_loss * (0.999 if action == "buy" else 1.001), 2)
        result["stop_loss_order"] = exchange.create_order(
            symbol=config.SYMBOL,
            type="STOP_LOSS_LIMIT",
            side=exit_side,
            amount=btc_qty,
            price=sl_limit,
            params={"stopPrice": round(stop_loss, 2), "timeInForce": "GTC"},
        )
        logger.info(
            "Stop-loss placed: %s %.6f BTC trigger=$%,.0f limit=$%,.0f",
            exit_side.upper(), btc_qty, stop_loss, sl_limit,
        )
    except Exception as exc:
        logger.warning("Stop-loss order failed (non-fatal): %s", exc)

    # Take-profit
    try:
        result["take_profit_order"] = exchange.create_order(
            symbol=config.SYMBOL,
            type="TAKE_PROFIT_LIMIT",
            side=exit_side,
            amount=btc_qty,
            price=round(take_profit, 2),
            params={"stopPrice": round(take_profit, 2), "timeInForce": "GTC"},
        )
        logger.info(
            "Take-profit placed: %s %.6f BTC @ $%,.0f",
            exit_side.upper(), btc_qty, take_profit,
        )
    except Exception as exc:
        logger.warning("Take-profit order failed (non-fatal): %s", exc)

    return result


def execute(exchange: ccxt.binance, decision: dict, snapshot: dict,
            portfolio: dict) -> dict:
    action = decision["action"]
    confidence = decision.get("confidence", 0.5)
    price = snapshot["price"]

    # Risk manager determines optimal position size and stop levels
    risk = risk_manager.assess_trade(action, confidence, snapshot, portfolio)

    # Use risk-managed size instead of Claude's raw suggestion
    amount = risk.recommended_usd

    result = {
        "action": action,
        "amount_usd": 0.0,
        "btc_amount": 0.0,
        "order": None,
        "stop_order": None,
        "success": False,
        "error": None,
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

        btc_qty = amount / price
        try:
            order = exchange.create_market_buy_order(config.SYMBOL, btc_qty)
            result.update({
                "success": True, "amount_usd": amount,
                "btc_amount": btc_qty, "order": order,
            })
            logger.info(
                "BUY %.6f BTC for $%.2f @ $%,.0f | SL=$%,.0f TP=$%,.0f R:R=%.1f",
                btc_qty, amount, price,
                risk.stop_loss_price, risk.take_profit_price, risk.risk_reward_ratio,
            )

            # Place stop-loss order on exchange (non-blocking)
            if risk.stop_loss_price > 0:
                exit_orders = _place_exit_orders(
                    exchange, action, btc_qty,
                    risk.stop_loss_price, risk.take_profit_price,
                )
                result["stop_order"] = exit_orders.get("stop_loss_order")
                result["tp_order"] = exit_orders.get("take_profit_order")

        except Exception as e:
            result["error"] = str(e)
            logger.error("BUY failed: %s", e)

    elif action == "sell":
        if portfolio["btc"] <= 0:
            result["error"] = "No BTC to sell"
            return result

        # Risk-managed sell: use recommended amount or 10% of holdings, whichever is less
        sell_usd = min(amount, portfolio["btc"] * price * 0.10)
        btc_qty = sell_usd / price

        min_qty = (exchange.markets.get(config.SYMBOL, {})
                   .get("limits", {}).get("amount", {}).get("min", 0.00001))
        if btc_qty < min_qty:
            result["error"] = f"Sell qty {btc_qty:.8f} below minimum {min_qty}"
            logger.warning(result["error"])
            return result

        try:
            order = exchange.create_market_sell_order(config.SYMBOL, btc_qty)
            result.update({
                "success": True, "amount_usd": sell_usd,
                "btc_amount": btc_qty, "order": order,
            })
            logger.info(
                "SELL %.6f BTC for $%.2f @ $%,.0f | SL=$%,.0f",
                btc_qty, sell_usd, price, risk.stop_loss_price,
            )

            if risk.stop_loss_price > 0:
                exit_orders = _place_exit_orders(
                    exchange, action, btc_qty,
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
