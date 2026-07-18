"""
Attribution scoreboard verification: direction parsing of free-text agent
assessments, per-source spread math, decisive hit-rates, ML calibration
buckets, and the PRINCIPLES §6 verdict rule.

Run on the VPS:
    cd /var/www/crypto_bot && venv/bin/python -m unittest tests.test_attribution -v
"""
import json
import unittest

import attribution


def _row(rid, price, price_after, decision, chg_7d=0.0, created="2026-07-18 12:00:00"):
    return {
        "id": rid, "created_at": created, "action": decision.get("action", "hold"),
        "price": price, "price_after_4h": price_after,
        "decision": json.dumps(decision),
        "market": json.dumps({"change_7d_pct": chg_7d}),
    }


class DirectionParsing(unittest.TestCase):
    def test_bullish_leading_text(self):
        self.assertEqual(attribution._direction(
            "BTC shows bullish momentum (2/3 timeframes, OBV rising)"), "bullish")

    def test_bearish_headwinds(self):
        self.assertEqual(attribution._direction(
            "BTC faces near-term headwinds from negative price narrative"), "bearish")

    def test_mixed_and_empty_are_neutral(self):
        self.assertEqual(attribution._direction("neutral (1↑ 1↓ signals)"), "neutral")
        self.assertEqual(attribution._direction(None), "neutral")
        self.assertEqual(attribution._direction("no social data"), "neutral")

    def test_bull_and_bear_mentions_tally(self):
        # "bullish macro setup despite bearish social sentiment" → tie → neutral
        self.assertEqual(attribution._direction(
            "Bitcoin shows bullish macro setup despite bearish social sentiment"),
            "neutral")


class SourceCalls(unittest.TestCase):
    def test_action_maps_to_decision_agent(self):
        calls = attribution._source_calls({"action": "buy"})
        self.assertEqual(calls["decision_agent"], "bullish")
        self.assertEqual(calls["ml_gate"], "neutral")

    def test_ml_gate_flags(self):
        calls = attribution._source_calls({"action": "hold", "ml_buy_signal": True})
        self.assertEqual(calls["ml_gate"], "bullish")


class Verdicts(unittest.TestCase):
    def test_small_sample_is_insufficient(self):
        self.assertEqual(attribution._verdict(10, 1.0), "insufficient data")

    def test_under_target_is_accumulating(self):
        self.assertEqual(attribution._verdict(120, 1.0), "accumulating")

    def test_zero_spread_at_target_flags_review(self):
        self.assertIn("REVIEW", attribution._verdict(250, -0.01))

    def test_positive_spread_at_target_contributes(self):
        self.assertEqual(attribution._verdict(250, 0.15), "contributing")


class ScoreboardMath(unittest.TestCase):
    def setUp(self):
        self._orig = attribution.db._execute
        # Two bullish decision-agent calls (+3%, -1%) and one bearish (-3%)
        self.rows = [
            _row(1, 100.0, 103.0, {"action": "buy", "ml_probability": 0.95,
                                   "ml_buy_signal": True}, chg_7d=5.0),
            _row(2, 100.0, 99.0, {"action": "buy", "ml_probability": 0.42}),
            _row(3, 100.0, 97.0, {"action": "sell", "ml_probability": 0.41}, chg_7d=-5.0),
        ]
        attribution.db._execute = lambda *a, **k: self.rows

    def tearDown(self):
        attribution.db._execute = self._orig

    def test_spread_is_bull_avg_minus_bear_avg(self):
        s = attribution.scoreboard()["sources"]["decision_agent"]
        self.assertEqual(s["avg_move_when_bullish_pct"], 1.0)   # (3 + -1)/2
        self.assertEqual(s["avg_move_when_bearish_pct"], -3.0)
        self.assertEqual(s["bull_bear_spread_pct"], 4.0)

    def test_decisive_hits_use_2pct_band(self):
        s = attribution.scoreboard()["sources"]["decision_agent"]
        # +3% (bull, hit) and -3% (bear, hit) are decisive; -1% is not
        self.assertEqual(s["decisive_calls"], 2)
        self.assertEqual(s["decisive_hit_rate"], 1.0)

    def test_ml_gate_scores_only_gate_cycles(self):
        s = attribution.scoreboard()["sources"]["ml_gate"]
        self.assertEqual(s["directional_calls"], 1)
        self.assertEqual(s["avg_move_when_bullish_pct"], 3.0)

    def test_regime_split_records_aligned_move(self):
        s = attribution.scoreboard()["sources"]["decision_agent"]
        self.assertEqual(s["by_regime"]["bull"]["n"], 1)
        # bear-regime row was a sell into a -3% move → aligned +3.0
        self.assertEqual(s["by_regime"]["bear"]["avg_aligned_move_pct"], 3.0)

    def test_calibration_buckets(self):
        calib = attribution.scoreboard()["ml_calibration"]
        top = next(c for c in calib if c["bucket"] == "0.9-1.0")
        self.assertEqual(top["n"], 1)
        self.assertEqual(top["up_rate_4h"], 1.0)
        mid = next(c for c in calib if c["bucket"] == "0.4-0.5")
        self.assertEqual(mid["n"], 2)
        self.assertEqual(mid["up_rate_4h"], 0.0)

    def test_corrupt_cross_symbol_rows_excluded(self):
        # An ETH row "evaluated" at BTC price → absurd move → excluded
        self.rows.append(_row(4, 1746.11, 62759.28, {"action": "buy"}))
        board = attribution.scoreboard()
        self.assertEqual(board["n_scored"], 3)
        self.assertEqual(board["n_excluded_corrupt"], 1)

    def test_report_text_renders(self):
        text = attribution.report_text()
        self.assertIn("Attribution scoreboard", text)
        self.assertIn("decision_agent", text)


if __name__ == "__main__":
    unittest.main()
