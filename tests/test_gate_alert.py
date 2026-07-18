"""
Mirror-ready gate alert: fires only on gate signals, and the numbers a human
would copy onto a personal account must match the validated barriers exactly.

Run on the VPS:
    cd /var/www/crypto_bot && venv/bin/python -m unittest tests.test_gate_alert -v
"""
import asyncio
import unittest

import main
import ml_signal


SNAP = {"price": 64000.0, "symbol": "BTC/USDT"}


class AlertText(unittest.TestCase):
    def test_no_gate_no_alert(self):
        self.assertIsNone(main._gate_alert_text({"action": "buy"}, SNAP))

    def test_buy_gate_numbers_match_validated_barriers(self):
        text = main._gate_alert_text(
            {"ml_buy_signal": True, "ml_probability": 0.923}, SNAP)
        self.assertIn("BTC/USDT BUY", text)
        self.assertIn("p=0.923", text)
        target = 64000 * (1 + ml_signal.PROFIT_TARGET_PCT / 100)
        stop = 64000 * (1 - ml_signal.STOP_LOSS_PCT / 100)
        self.assertIn(f"${target:,.2f}", text)
        self.assertIn(f"${stop:,.2f}", text)
        self.assertIn(f"close after {ml_signal.LOOKAHEAD_HOURS}h", text)

    def test_sell_gate_flips_direction(self):
        text = main._gate_alert_text({"ml_sell_signal": True}, SNAP)
        self.assertIn("SELL", text)
        # For a sell, the target sits BELOW entry and the stop above
        target = 64000 * (1 - ml_signal.PROFIT_TARGET_PCT / 100)
        stop = 64000 * (1 + ml_signal.STOP_LOSS_PCT / 100)
        self.assertIn(f"${target:,.2f}", text)
        self.assertIn(f"${stop:,.2f}", text)


class AlertSending(unittest.TestCase):
    def setUp(self):
        self._orig_notify = main.notify
        self._orig_log = main.db.log_event
        self.sent, self.events = [], []

        async def fake_notify(text):
            self.sent.append(text)
        main.notify = fake_notify
        main.db.log_event = lambda *a, **k: self.events.append((a, k))

    def tearDown(self):
        main.notify = self._orig_notify
        main.db.log_event = self._orig_log

    def test_gate_sends_alert_and_logs_event(self):
        asyncio.run(main._maybe_gate_alert(
            {"ml_buy_signal": True, "ml_probability": 0.91}, SNAP))
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0][0][0], "gate_signal")

    def test_hold_cycle_stays_silent(self):
        asyncio.run(main._maybe_gate_alert({"action": "hold"}, SNAP))
        self.assertEqual(self.sent, [])
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
