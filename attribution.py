"""
Phase 2 attribution — the minimally viable "Meta AI" (ROADMAP Phase 2).

Every evaluated cycle row already stores each source's opinion (decision JSON)
and the realized 4h forward move (price_after_4h). This module joins the two
into a scoreboard: per source, the average forward move conditioned on its
call — bullish-call average minus bearish-call average is the source's edge —
plus a hit-rate on decisive moves (|move| >= 2%, same band self_correction
uses for wrong/correct).

Decision rule (PRINCIPLES §6): after 200 scored calls, a source whose
bull-bear spread is indistinguishable from zero gets its influence reduced.

The scoreboard is persisted to bot_state["attribution"] so the Laravel API
and dashboard read the exact numbers the bot computed — no second scoring
implementation.
"""
import json
import logging
from datetime import datetime, timezone

import database as db

logger = logging.getLogger(__name__)

DECISIVE_PCT = 2.0    # matches self_correction WRONG_THRESHOLD_PCT
SCORE_TARGET = 200    # PRINCIPLES §6 review bar
REGIME_7D_PCT = 3.0   # |7d change| beyond this = trending regime

_BULL_WORDS = ("bullish", "bull", "positive", "risk-on", "accumulat")
_BEAR_WORDS = ("bearish", "bear", "negative", "risk-off", "capitulat", "headwind")


def _direction(text) -> str:
    """Directional lean of a free-text agent assessment (leading words win)."""
    if not text:
        return "neutral"
    t = str(text)[:120].lower()
    bull = sum(t.count(w) for w in _BULL_WORDS)
    bear = sum(t.count(w) for w in _BEAR_WORDS)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


def _regime(market: dict) -> str:
    try:
        chg = float(market.get("change_7d_pct") or 0)
    except (TypeError, ValueError):
        return "chop"
    if chg >= REGIME_7D_PCT:
        return "bull"
    if chg <= -REGIME_7D_PCT:
        return "bear"
    return "chop"


def _source_calls(decision: dict) -> dict:
    """Each source's directional call for one cycle ('neutral' = abstain)."""
    action = decision.get("action", "hold")
    ml_call = "neutral"
    if decision.get("ml_buy_signal"):
        ml_call = "bullish"
    elif decision.get("ml_sell_signal"):
        ml_call = "bearish"
    return {
        "decision_agent": {"buy": "bullish", "sell": "bearish"}.get(action, "neutral"),
        "ml_gate": ml_call,
        "market_agent": _direction(decision.get("market_assessment")),
        "sentiment_agent": _direction(decision.get("sentiment_assessment")),
        "news_feed": _direction(decision.get("news_sentiment")),
        "social_feed": _direction(decision.get("social_sentiment")),
    }


def _new_bucket() -> dict:
    return {"calls": 0, "bull": 0, "bear": 0, "abstain": 0,
            "bull_move_sum": 0.0, "bear_move_sum": 0.0,
            "decisive": 0, "decisive_hits": 0,
            "regimes": {}}


def scoreboard(limit: int = 2000) -> dict:
    """Compute the attribution scoreboard over evaluated cycle rows."""
    rows = db._execute(
        """SELECT id, created_at, action, price, price_after_4h, decision, market
           FROM trades
           WHERE outcome IS NOT NULL AND price_after_4h IS NOT NULL
             AND price > 0
           ORDER BY id DESC LIMIT %s""",
        (limit,),
        fetch="all",
    )

    sources: dict[str, dict] = {}
    calib: dict[int, dict] = {}   # ml_probability decile -> {n, prob_sum, up}
    n_scored = 0
    first_at = last_at = None

    for r in rows:
        try:
            decision = r["decision"]
            if isinstance(decision, str):
                decision = json.loads(decision)
            market = r["market"]
            if isinstance(market, str):
                market = json.loads(market)
            move = (float(r["price_after_4h"]) - float(r["price"])) / float(r["price"]) * 100
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        n_scored += 1
        ts = str(r["created_at"])
        first_at = ts if first_at is None or ts < first_at else first_at
        last_at = ts if last_at is None or ts > last_at else last_at
        regime = _regime(market)
        decisive = abs(move) >= DECISIVE_PCT

        for name, call in _source_calls(decision).items():
            s = sources.setdefault(name, _new_bucket())
            s["calls"] += 1
            if call == "neutral":
                s["abstain"] += 1
                continue
            side = "bull" if call == "bullish" else "bear"
            s[side] += 1
            s[f"{side}_move_sum"] += move
            reg = s["regimes"].setdefault(regime, {"n": 0, "move_sum": 0.0})
            reg["n"] += 1
            reg["move_sum"] += move if side == "bull" else -move
            if decisive:
                s["decisive"] += 1
                if (call == "bullish") == (move > 0):
                    s["decisive_hits"] += 1

        prob = decision.get("ml_probability")
        if prob is not None:
            try:
                prob = float(prob)
                b = min(int(prob * 10), 9)
                c = calib.setdefault(b, {"n": 0, "prob_sum": 0.0, "up": 0})
                c["n"] += 1
                c["prob_sum"] += prob
                c["up"] += 1 if move > 0 else 0
            except (TypeError, ValueError):
                pass

    out_sources = {}
    for name, s in sources.items():
        avg_bull = s["bull_move_sum"] / s["bull"] if s["bull"] else None
        avg_bear = s["bear_move_sum"] / s["bear"] if s["bear"] else None
        spread = (avg_bull - avg_bear) if (avg_bull is not None and avg_bear is not None) else None
        directional = s["bull"] + s["bear"]
        out_sources[name] = {
            "cycles": s["calls"],
            "directional_calls": directional,
            "abstains": s["abstain"],
            "avg_move_when_bullish_pct": round(avg_bull, 3) if avg_bull is not None else None,
            "avg_move_when_bearish_pct": round(avg_bear, 3) if avg_bear is not None else None,
            "bull_bear_spread_pct": round(spread, 3) if spread is not None else None,
            "decisive_calls": s["decisive"],
            "decisive_hit_rate": round(s["decisive_hits"] / s["decisive"], 3) if s["decisive"] else None,
            "by_regime": {
                k: {"n": v["n"], "avg_aligned_move_pct": round(v["move_sum"] / v["n"], 3)}
                for k, v in sorted(s["regimes"].items())
            },
            "verdict": _verdict(directional, spread),
        }

    calibration = [
        {"bucket": f"{b/10:.1f}-{(b+1)/10:.1f}",
         "n": c["n"],
         "avg_prob": round(c["prob_sum"] / c["n"], 3),
         "up_rate_4h": round(c["up"] / c["n"], 3)}
        for b, c in sorted(calib.items())
    ]

    return {
        "n_scored": n_scored,
        "score_target": SCORE_TARGET,
        "decisive_threshold_pct": DECISIVE_PCT,
        "window": {"from": first_at, "to": last_at},
        "note": ("Forward move is the 4h horizon self_correction evaluates; "
                 "ML calibration uses 4h up-rate as a proxy for the "
                 "triple-barrier label, so treat buckets directionally."),
        "sources": out_sources,
        "ml_calibration": calibration,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _verdict(directional: int, spread) -> str:
    if directional < 50:
        return "insufficient data"
    if directional < SCORE_TARGET:
        return "accumulating"
    if spread is None:
        return "accumulating"
    if spread <= 0:
        return "REVIEW: no measurable edge (PRINCIPLES §6)"
    return "contributing"


def persist_scoreboard() -> dict | None:
    """Recompute and store the scoreboard in bot_state for the API/dashboard."""
    try:
        board = scoreboard()
        db.set_state("attribution", json.dumps(board))
        return board
    except Exception as exc:
        logger.warning("Attribution scoreboard failed (non-fatal): %s", exc)
        return None


def report_text(board: dict | None = None) -> str:
    """Compact weekly Telegram report."""
    board = board or scoreboard()
    lines = [
        "📊 Attribution scoreboard",
        f"{board['n_scored']} cycles scored (target {board['score_target']} "
        f"directional calls per source before PRINCIPLES §6 review)",
        "",
    ]
    for name, s in sorted(board["sources"].items()):
        spread = s["bull_bear_spread_pct"]
        spread_txt = f"{spread:+.2f}%" if spread is not None else "n/a"
        hit = s["decisive_hit_rate"]
        hit_txt = f", decisive hit {hit:.0%} ({s['decisive_calls']})" if hit is not None else ""
        lines.append(
            f"• {name}: {s['directional_calls']} calls, "
            f"bull-bear spread {spread_txt}/4h{hit_txt} — {s['verdict']}"
        )
    if board["ml_calibration"]:
        lines.append("")
        lines.append("ML calibration (prob → 4h up-rate):")
        for c in board["ml_calibration"]:
            lines.append(f"  {c['bucket']}: {c['up_rate_4h']:.0%} up over {c['n']}")
    return "\n".join(lines)
