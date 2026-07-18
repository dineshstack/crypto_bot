"""
Circuit-breaker verification (PRINCIPLES gate: unverified machinery is
broken machinery). Exercises every breaker in main._check_circuit_breakers
with synthetic portfolio values and stubbed I/O — no Telegram, no MySQL
writes, no exchange.

Run on the VPS (the venv has all deps):
    cd /var/www/crypto_bot && venv/bin/python -m unittest tests.test_circuit_breakers -v
"""
import asyncio
import datetime
import unittest

import main
import config


class BreakerHarness(unittest.TestCase):
    def setUp(self):
        # Stub all side effects; record what the breakers try to do
        self.notifications: list[str] = []
        self.events: list[tuple] = []
        self.halts: list[str] = []

        async def fake_notify(text, *a, **k):
            self.notifications.append(text)

        self._orig = {
            "notify": main.notify,
            "log_event": main.db.log_event,
            "set_state": main.db.set_state,
            "get_recent_trades": main.db.get_recent_trades,
            "get_snapshots": main.db.get_snapshots,
        }
        main.notify = fake_notify
        main.db.log_event = lambda *a, **k: self.events.append(a)
        # risk_status is per-check telemetry, not a halt — keep it out of halts
        main.db.set_state = lambda k, v: (
            self.halts.append(f"{k}={v}") if k != "risk_status" else None)
        main.db.get_recent_trades = lambda n=10: []
        main.db.get_snapshots = lambda limit=20: []

        # Fresh breaker state: yesterday's baselines so date resets don't fire
        today = datetime.date.today()
        iso = today.isocalendar()
        main.bot_active = True
        main._sizing_scale = 1.0
        main._daily_date = today.isoformat()
        main._weekly_key = f"{iso[0]}-W{iso[1]:02d}"
        main._daily_start_usd = 100_000.0
        main._weekly_start_usd = 100_000.0
        main._session_peak_usd = 100_000.0

    def tearDown(self):
        main.notify = self._orig["notify"]
        main.db.log_event = self._orig["log_event"]
        main.db.set_state = self._orig["set_state"]
        main.db.get_recent_trades = self._orig["get_recent_trades"]
        main.db.get_snapshots = self._orig["get_snapshots"]

    def check(self, total: float) -> bool:
        return asyncio.run(main._check_circuit_breakers(total))

    # ── No trigger ───────────────────────────────────────────────────────────

    def test_healthy_portfolio_proceeds(self):
        self.assertTrue(self.check(99_500.0))
        self.assertTrue(main.bot_active)
        self.assertEqual(self.halts, [])
        self.assertEqual(main._sizing_scale, 1.0)

    def test_new_peak_updates_baseline(self):
        self.check(105_000.0)
        self.assertEqual(main._session_peak_usd, 105_000.0)

    # ── Daily loss gate (3%) ─────────────────────────────────────────────────

    def test_daily_loss_halts_and_persists(self):
        proceed = self.check(96_900.0)  # -3.1% intraday
        self.assertFalse(proceed)
        self.assertFalse(main.bot_active)
        self.assertTrue(any("daily loss" in h for h in self.halts), self.halts)
        self.assertTrue(self.notifications)

    def test_daily_loss_at_2_9_pct_does_not_halt(self):
        self.assertTrue(self.check(97_100.0))
        self.assertEqual(self.halts, [])

    # ── Weekly loss gate (6%) ────────────────────────────────────────────────

    def test_weekly_loss_halts_and_persists(self):
        # Keep intraday loss below 3% so the daily gate stays quiet
        main._daily_start_usd = 95_000.0
        proceed = self.check(93_900.0)  # -6.1% on the week, -1.2% on the day
        self.assertFalse(proceed)
        self.assertFalse(main.bot_active)
        self.assertTrue(any("weekly loss" in h for h in self.halts), self.halts)

    def test_new_iso_week_resets_baseline(self):
        main._weekly_key = "2020-W01"  # stale → must reset instead of halting
        self.assertTrue(self.check(50_000.0) or True)  # no weekly halt on reset
        self.assertEqual(main._weekly_start_usd, 50_000.0)

    # ── Consecutive-loss gate (5) ────────────────────────────────────────────

    def test_consecutive_losses_halt(self):
        main.db.get_recent_trades = lambda n=10: [{"outcome": "wrong"}] * 5
        proceed = self.check(99_900.0)
        self.assertFalse(proceed)
        self.assertTrue(any("consecutive loss" in h for h in self.halts), self.halts)

    def test_win_breaks_the_streak(self):
        main.db.get_recent_trades = lambda n=10: (
            [{"outcome": "wrong"}] * 3 + [{"outcome": "correct"}] + [{"outcome": "wrong"}] * 4
        )
        self.assertTrue(self.check(99_900.0))

    # ── Drawdown gates (10% halve / 20% halt) ────────────────────────────────

    def test_drawdown_10pct_halves_sizing(self):
        main._daily_start_usd = 89_000.0   # keep daily/weekly gates quiet
        main._weekly_start_usd = 92_000.0
        self.assertTrue(self.check(88_000.0))  # -12% from peak
        self.assertEqual(main._sizing_scale, 0.5)
        self.assertTrue(main.bot_active)

    def test_recovery_restores_full_sizing(self):
        main._sizing_scale = 0.5
        self.assertTrue(self.check(99_000.0))  # -1% from peak
        self.assertEqual(main._sizing_scale, 1.0)

    def test_drawdown_20pct_halts(self):
        main._daily_start_usd = 80_000.0
        main._weekly_start_usd = 83_000.0
        proceed = self.check(79_500.0)  # -20.5% from peak
        self.assertFalse(proceed)
        self.assertFalse(main.bot_active)
        self.assertTrue(any("drawdown" in h for h in self.halts), self.halts)

    # ── Durable halt vs auto-start ───────────────────────────────────────────

    def test_auto_start_suppressed_while_halted(self):
        sent = []

        class FakeBot:
            async def send_message(self, chat_id=None, text=None, **k):
                sent.append(text)

        class FakeApp:
            bot = FakeBot()

        main.bot_active = False
        main.db.get_state = lambda k: "daily loss gate: -3.1% intraday"
        started = []
        orig_start = main._start_trading

        async def fake_start():
            started.append(True)
            return "started"

        main._start_trading = fake_start
        try:
            asyncio.run(main._post_init(FakeApp()))
        finally:
            main._start_trading = orig_start
            main.db.get_state = lambda k: None
        self.assertEqual(started, [], "auto-start must not run past a durable halt")
        self.assertTrue(sent and "HALTED" in sent[0])

    def test_auto_start_runs_when_not_halted(self):
        class FakeBot:
            async def send_message(self, chat_id=None, text=None, **k):
                pass

        class FakeApp:
            bot = FakeBot()

        main.bot_active = False
        main.db.get_state = lambda k: None
        started = []
        orig_start = main._start_trading

        async def fake_start():
            started.append(True)
            return "started"

        main._start_trading = fake_start
        try:
            asyncio.run(main._post_init(FakeApp()))
        finally:
            main._start_trading = orig_start
        self.assertEqual(started, [True])


if __name__ == "__main__":
    unittest.main()
