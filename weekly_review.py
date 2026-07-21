"""
Weekly deep-review module using Claude Fable 5 (falls back to Opus 4.8).

Runs once per week (triggered from main.py's loop check).
Looks at the last 7 days of trades, evaluates performance, and produces:
  - A plain-English performance summary sent to Telegram
  - 3 new lessons stored in MySQL for future prompt injection

Uses adaptive thinking so Opus can reason deeply about patterns.
"""
import logging
from datetime import datetime, timedelta, timezone
import anthropic
import claude_deep
import config
import database as db

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

REVIEW_INTERVAL_DAYS = 7


def should_run() -> bool:
    """
    Return True if ≥ 7 days have passed since the last review.
    On a fresh database (no review yet), fall back to the first trade date so
    the review loop bootstraps itself once a week of history exists.
    """
    last = db.get_last_weekly_review_date()
    if last is None:
        first_trade = db.get_first_trade_date()
        if first_trade is None:
            return False  # no trades yet — nothing to review
        return (datetime.now(timezone.utc) - first_trade).days >= REVIEW_INTERVAL_DAYS
    return (datetime.now(timezone.utc) - last).days >= REVIEW_INTERVAL_DAYS


def run() -> str:
    """
    Perform the weekly review. Returns a short summary string for Telegram.
    Saves full review + new lessons to MySQL.
    """
    now   = datetime.now(timezone.utc)
    start = now - timedelta(days=REVIEW_INTERVAL_DAYS)

    trades     = db.get_trades_in_period(start, now)
    actionable = [t for t in trades if t["action"] != "hold"]
    correct    = [t for t in actionable if t.get("outcome") == "correct"]
    wrong      = [t for t in actionable if t.get("outcome") == "wrong"]
    holds      = [t for t in trades if t["action"] == "hold"]

    if not trades:
        db.save_weekly_review(start, now, 0, 0, 0, 0.0, "No trades this week.")
        return "No trades in the last 7 days — nothing to review."

    # Build human-readable trade history
    lines = []
    for t in trades:
        m = t.get("market") or {}
        d = t.get("decision") or {}
        pct_str = ""
        if t.get("price_after_4h") and t.get("price"):
            pct = (t["price_after_4h"] - t["price"]) / t["price"] * 100
            pct_str = f" → {pct:+.1f}% [{t.get('outcome','?')}]"
        lines.append(
            f"  {t['created_at'][:16]}Z  "
            f"{t['action'].upper():4}  "
            f"${t.get('amount_usd',0):.0f}  "
            f"@${t.get('price',0):,}  "
            f"RSI {m.get('rsi','?')}  "
            f"F&G {m.get('fear_greed','?')}"
            f"{pct_str}"
        )

    existing_lessons = db.get_active_lessons(10)
    lessons_text = (
        "\n".join(f"  - {l}" for l in existing_lessons)
        if existing_lessons else "  (none yet)"
    )

    win_rate = len(correct) / len(actionable) * 100 if actionable else 0

    prompt = f"""You are reviewing a BTC trading bot's performance for the past 7 days.

TRADE HISTORY ({len(trades)} total: {len(actionable)} actionable, {len(holds)} holds):
{chr(10).join(lines)}

CURRENT METRICS:
  Actionable trades : {len(actionable)}
  Correct outcomes  : {len(correct)}
  Wrong outcomes    : {len(wrong)}
  Win rate          : {win_rate:.0f}%
  Holds             : {len(holds)}

ACTIVE LESSONS (already in use — do not repeat these):
{lessons_text}

Your task:
1. Write a SUMMARY paragraph (3–4 sentences) covering what went well, what didn't, and the market context.
2. Provide exactly 3 NEW lessons not already listed above, each starting with "Avoid", "Do not", or "Only". Be specific about RSI ranges, price relationships, or Fear & Greed levels where relevant.

Use this exact format (no extra text):
SUMMARY: <paragraph>
LESSON 1: <lesson sentence>
LESSON 2: <lesson sentence>
LESSON 3: <lesson sentence>"""

    response = claude_deep.call_deep_model(
        _client, max_tokens=1024, thinking=True,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        logger.warning("Weekly review declined by safety filters (whole fallback chain)")
        return "Weekly review was declined by the model's safety filters — try /review again later."

    # Extract text block (thinking blocks are separate)
    review_text = next(
        (b.text for b in response.content if getattr(b, "type", None) == "text"),
        "",
    )

    # Parse and store new lessons
    for line in review_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("LESSON"):
            lesson_body = stripped.split(":", 1)[-1].strip()
            if lesson_body:
                db.save_lesson(lesson_body, "weekly_review")

    # Store review record
    db.save_weekly_review(start, now, len(actionable), len(correct), len(wrong), 0.0, review_text)

    # Return just the summary for Telegram
    for line in review_text.splitlines():
        if line.strip().startswith("SUMMARY:"):
            return line.split(":", 1)[-1].strip()

    return "Weekly review complete — lessons updated."
