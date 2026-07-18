"""
Executed-size verification: the circuit-breaker scale and the alpha-sleeve
gate multiplier must shape the REAL order amount (they previously only
touched the advisory decision["trade_usd"], which execute() ignores).

Run on the VPS:
    cd /var/www/crypto_bot && venv/bin/python -m unittest tests.test_sizing -v
"""
import unittest

import config
from executor import _sized_amount


class SizedAmount(unittest.TestCase):
    def test_plain_buy_uses_risk_recommendation(self):
        amount, gate = _sized_amount("buy", {}, 10.0)
        self.assertEqual(amount, 10.0)
        self.assertFalse(gate)

    def test_drawdown_scale_halves_the_executed_amount(self):
        amount, _ = _sized_amount("buy", {}, 10.0, size_scale=0.5)
        self.assertEqual(amount, 5.0)

    def test_scale_never_drops_below_the_floor(self):
        amount, _ = _sized_amount("buy", {}, 3.0, size_scale=0.5)
        self.assertEqual(amount, config.MIN_TRADE_USD)

    def test_gate_buy_earns_the_multiplier(self):
        amount, gate = _sized_amount("buy", {"ml_buy_signal": True}, 10.0)
        self.assertEqual(amount, min(10.0 * config.GATE_TRADE_MULT,
                                     config.MAX_GATE_TRADE_USD))
        self.assertTrue(gate)

    def test_gate_buy_respects_the_ceiling(self):
        amount, _ = _sized_amount("buy", {"ml_buy_signal": True}, 100.0)
        self.assertEqual(amount, config.MAX_GATE_TRADE_USD)

    def test_gate_multiplier_stacks_on_drawdown_scale(self):
        # In a drawdown, even gate trades run at half size first
        amount, _ = _sized_amount("buy", {"ml_buy_signal": True}, 10.0, size_scale=0.5)
        self.assertEqual(amount, min(5.0 * config.GATE_TRADE_MULT,
                                     config.MAX_GATE_TRADE_USD))

    def test_sell_ignores_the_buy_gate(self):
        amount, gate = _sized_amount("sell", {"ml_buy_signal": True}, 10.0)
        self.assertEqual(amount, 10.0)
        self.assertFalse(gate)

    def test_hold_style_decisions_untouched(self):
        amount, gate = _sized_amount("buy", {"ml_buy_signal": False}, 12.5)
        self.assertEqual(amount, 12.5)
        self.assertFalse(gate)


if __name__ == "__main__":
    unittest.main()
