"""
Auto-generated market reports for crypto advisors.

Generates weekly/monthly market reports using Claude Opus that advisors
can share with their clients. Each report includes:
  - Market overview (BTC, ETH, dominance, Fear & Greed)
  - Portfolio performance summary
  - Top signals detected this period
  - Sector/narrative rotation data
  - AI-generated market outlook and recommendations

Reports are stored in MySQL and viewable on the dashboard.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import anthropic


class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)


def _json(obj) -> str:
    return json.dumps(obj, cls=_SafeEncoder)

import config
import database as db
import analytics
import cross_asset
import options_data

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _gather_report_data(days: int = 7) -> dict:
    """Collect all data needed for the report."""
    # Portfolio performance
    metrics = analytics.compute_metrics(days)

    # Recent trades
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    trades = db.get_trades_in_period(start, end)
    actionable = [t for t in trades if t.get("action") != "hold" and t.get("success")]

    # Bot events (signals, anomalies)
    events = db.get_events(limit=200)
    signals = [e for e in events if e.get("event") in ("trade", "trade_eth", "circuit_breaker", "lesson")]
    recent_signals = signals[:20]

    # Cross-asset context
    try:
        macro = cross_asset.get_cross_asset_data()
    except Exception:
        macro = {"assets": {}, "summary": "unavailable"}

    # Options data
    try:
        opts = options_data.get_options_data()
    except Exception:
        opts = {"summary": "unavailable"}

    # Latest screening
    try:
        screening = db._execute(
            """SELECT symbol, momentum_score, change_7d_pct, risk_tier, market_cap
               FROM coin_screenings
               WHERE scan_date = (SELECT MAX(scan_date) FROM coin_screenings)
               ORDER BY momentum_score DESC LIMIT 10""",
            fetch="all",
        )
    except Exception:
        screening = []

    # Fear & Greed
    try:
        import requests
        fg_r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=5)
        fg_data = fg_r.json().get("data", []) if fg_r.ok else []
    except Exception:
        fg_data = []

    return {
        "period_days": days,
        "metrics": metrics,
        "total_decisions": len(trades),
        "actionable_trades": len(actionable),
        "holds": len(trades) - len(actionable),
        "signals": recent_signals,
        "macro": macro,
        "options": opts,
        "screening_top10": screening,
        "fear_greed_7d": fg_data,
    }


def generate_report(days: int = 7, report_type: str = "weekly") -> dict:
    """
    Generate a complete market report using Claude Opus.
    Returns the report dict and stores it in MySQL.
    """
    data = _gather_report_data(days)
    metrics = data["metrics"]
    # Ensure all metric values are float (MySQL returns Decimal)
    for k, v in metrics.items():
        if hasattr(v, '__float__') and not isinstance(v, (int, float, str)):
            metrics[k] = float(v)

    # Build the structured prompt
    fg_str = ""
    if data["fear_greed_7d"]:
        fg_values = [f"{d.get('value', '?')} ({d.get('value_classification', '?')})" for d in data["fear_greed_7d"][:7]]
        fg_str = f"Fear & Greed (last 7 days, newest first): {', '.join(fg_values)}"

    screening_str = ""
    if data["screening_top10"]:
        lines = []
        for s in data["screening_top10"]:
            cap = float(s.get("market_cap", 0))
            cap_str = f"${cap/1e9:.1f}B" if cap >= 1e9 else f"${cap/1e6:.0f}M"
            lines.append(f"  {s['symbol']}: score {s['momentum_score']}, 7d {s.get('change_7d_pct', 0):+.1f}%, {s['risk_tier']}, {cap_str}")
        screening_str = "Top 10 Coins by Momentum:\n" + "\n".join(lines)

    macro_str = data["macro"].get("summary", "unavailable")
    options_str = data["options"].get("summary", "unavailable")

    signal_lines = []
    for s in data["signals"][:15]:
        signal_lines.append(f"  [{s.get('event', '')}] {s.get('message', '')}")
    signals_str = "\n".join(signal_lines) if signal_lines else "No significant signals"

    prompt = f"""Generate a professional {report_type} cryptocurrency market report for an investment advisor to share with clients.

PERIOD: Last {days} days (ending {date.today().isoformat()})

PORTFOLIO PERFORMANCE:
  PnL: {metrics['pnl_pct']:+.1f}% (${metrics['pnl_usd']:+.2f})
  Win Rate: {metrics['win_rate']:.0%} ({metrics['wins']}W / {metrics['losses']}L)
  Sharpe Ratio: {metrics['sharpe_ratio']}
  Max Drawdown: {metrics['max_drawdown_pct']:.1f}%
  Total Decisions: {data['total_decisions']} ({data['actionable_trades']} trades, {data['holds']} holds)

MARKET CONTEXT:
  {fg_str}
  Macro: {macro_str}
  Options: {options_str}

{screening_str}

NOTABLE SIGNALS THIS PERIOD:
{signals_str}

Write the report with these sections:
1. EXECUTIVE SUMMARY (2-3 sentences — the key takeaway for clients)
2. MARKET OVERVIEW (BTC trend, ETH trend, overall market conditions, dominance)
3. KEY SIGNALS (what the AI trading system detected — translate into plain language)
4. SECTOR OUTLOOK (which crypto sectors/narratives look strong vs weak)
5. RISK ASSESSMENT (current market risks, what could go wrong)
6. RECOMMENDATION (general positioning advice — conservative/balanced/aggressive)

Write in professional but accessible language. Clients are not traders — explain terms simply.
Do not use markdown. Use plain text with section headers in CAPS."""

    try:
        resp = _client.beta.messages.create(
            model=config.CLAUDE_DEEP_MODEL,
            max_tokens=2000,
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": config.CLAUDE_DEEP_FALLBACK}],
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("report request declined by safety filters")
        # Fable always emits thinking blocks first — take the first text block
        report_text = next(
            (b.text for b in resp.content if getattr(b, "type", None) == "text"),
            "",
        ).strip()

        # Log the Claude call
        db.log_claude_call(
            cycle_id=f"report_{date.today().isoformat()}",
            agent="report_generator",
            model=resp.model,
            prompt=prompt[:5000],
            response=report_text[:5000],
            tokens_in=resp.usage.input_tokens if resp.usage else 0,
            tokens_out=resp.usage.output_tokens if resp.usage else 0,
        )
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        report_text = f"Report generation failed: {exc}"

    # Parse sections from the report
    today = date.today()
    period_start = today - timedelta(days=days)

    # Extract executive summary (first paragraph after header)
    summary = report_text[:500].split("\n\n")[0] if report_text else ""

    # Store in MySQL
    report_id = db._execute(
        """INSERT INTO market_reports
           (report_type, period_start, period_end, title, summary,
            market_overview, top_signals, sector_data, outlook, portfolio_perf, raw_data)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            report_type,
            period_start.isoformat(),
            today.isoformat(),
            f"{report_type.title()} Market Report — {today.strftime('%b %d, %Y')}",
            summary,
            report_text,
            _json([s.get("message", "") for s in data["signals"][:10]]),
            _json({"screening": [dict(s) for s in data["screening_top10"]] if data["screening_top10"] else []}),
            "",
            _json({"pnl_pct": metrics["pnl_pct"], "win_rate": metrics["win_rate"],
                        "sharpe": metrics["sharpe_ratio"], "max_dd": metrics["max_drawdown_pct"]}),
            _json(data.get("fear_greed_7d", [])),
        ),
    )

    logger.info("Market report generated: %s (#%s)", report_type, report_id)

    return {
        "id": report_id,
        "type": report_type,
        "title": f"{report_type.title()} Market Report — {today.strftime('%b %d, %Y')}",
        "content": report_text,
        "metrics": metrics,
        "period_start": period_start.isoformat(),
        "period_end": today.isoformat(),
    }


def get_latest_reports(limit: int = 10) -> list[dict]:
    """Get recent reports from MySQL."""
    rows = db._execute(
        "SELECT * FROM market_reports ORDER BY created_at DESC LIMIT %s",
        (limit,),
        fetch="all",
    )
    for r in rows:
        r["created_at"] = str(r["created_at"])
        r["period_start"] = str(r.get("period_start", ""))
        r["period_end"] = str(r.get("period_end", ""))
        for key in ("top_signals", "sector_data", "portfolio_perf", "raw_data"):
            if isinstance(r.get(key), str):
                try:
                    r[key] = json.loads(r[key])
                except Exception:
                    pass
    return rows


def format_report_telegram(report: dict) -> str:
    """Format report for Telegram (truncated summary)."""
    content = report.get("content", "")
    # Truncate to ~3000 chars for Telegram
    if len(content) > 3000:
        content = content[:2950] + "\n\n... [Full report available on dashboard]"

    metrics = report.get("metrics", {})
    header = (
        f"📊 *{report['type'].title()} Market Report*\n"
        f"_{report['period_start']} — {report['period_end']}_\n\n"
        f"PnL: {metrics.get('pnl_pct', 0):+.1f}% \\| "
        f"WR: {metrics.get('win_rate', 0):.0%} \\| "
        f"Sharpe: {metrics.get('sharpe_ratio', 0)}\n\n"
    )

    return header + content[:2500]
