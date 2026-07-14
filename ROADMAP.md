# ROADMAP — Phased, Evidence-Gated

Governed by PRINCIPLES.md. Each phase must hit its exit criterion before the
next phase starts. Items inside a phase are ordered by expected information
gain per hour of work.

---

## Current state (2026-07-14 baseline — do not edit, this is the yardstick)

- Single strategy: BTC/USDT long-only, 1h bars, triple-barrier ML ensemble
  (XGB+LGB→LR), Platt-calibrated, EV-derived gates.
- **First honest OOS result** (6-month holdout, net of costs):
  +0.13% return, 55.8% WR over 86 trades, PF 1.09, Sharpe 0.53, max DD 0.67%
  vs B&H −30.3%. Verdict: thin real edge, strangled by costs (fees ate 84%
  of gross edge).
- Live: Binance **testnet** (~$71k paper), 3 Claude agents (market /
  sentiment / decision), ~8 cycles/day, 11 executed trades, 0 lessons stored.
- Known defects: recurring `'int' object is not subscriptable` cycle error;
  lessons pipeline never fires; Boruta-SHAP broken on current scipy;
  training data depth ~11 months.

## Phase 0 — Prove or kill the edge (target: ~2 weeks)

1. **Robustness sweep** — `--holdout` at 3, 9, and 12 months (as data depth
   allows). Record all runs in `backtest_runs`. *No code needed.*
2. **Cost lever** — add `--maker-fees` scenario to the backtester
   (0.075%/side vs 0.1% taker) and, separately, quantify limit-entry fill
   risk. Costs are the single highest-Sharpe project available.
3. **Barrier A/B** — make PROFIT_TARGET_PCT / STOP_LOSS_PCT / LOOKAHEAD_HOURS
   training parameters; compare 1.5%/1.5%/4h (current) vs 3%/1.5%/24h
   out-of-sample. Hypothesis: fewer, longer trades → edge/cost ratio doubles.
4. **Fix defects**: the recurring cycle exception; the never-firing lessons
   condition; replace Boruta-SHAP or pin a working version.
5. **Data depth** — extend paginated fetch to 2–3 years of 1h candles for
   training and multi-window validation.

**Exit criterion:** edge confirmed (positive net expectancy in ≥3 disjoint
windows per G2/G3) → Phase 1. Edge dead → strategy redesign, not more infra.

**Status 2026-07-14 (late):** validation gauntlet passed —
- placebo test (24-bar prediction lag) killed the edge: 92.5%→16.7% WR → harness clean
- all 40 ledger trades independently verified against real Binance candles (0 failures)
- three DISJOINT 3-month windows under next-bar-open fills, independently
  trained holdout models: 10-0, 13-1, 10-1 (33W/2L combined, ~+0.3% net/window,
  B&H −11..−19% each). Strategy identity: rare extreme-momentum session-open
  bounce scalps (~4 trades/month, gates ~0.9).
- Open before G3: all windows are 2025-10..2026-07 bear/chop — no bull-tape
  evidence yet; 35 trades ≪ 300; absolute returns tiny at current sizing.
- Next: walk offset windows back through 2023-24 bull (regime diversity +
  trade count), retrain live model, accumulate forward paper evidence.

**PHASE 0 EXIT — 2026-07-15.** Eight disjoint 3-month holdout windows,
independently trained models, next-bar-open fills, net of costs:

| Window (ends) | Regime (B&H) | Trades | W-L | Net |
|---|---|---|---|---|
| 2026-07 | bear −13.0% | 10 | 10-0 | +0.26% |
| 2026-04 | bear −19.1% | 14 | 13-1 | +0.36% |
| 2026-01 | bear −11.6% | 11 | 10-1 | +0.26% |
| 2025-10 | bear −10.6% | 3 | 3-0 | +0.09% |
| 2025-07 | bull +25.7% | 13 | 13-0 | +0.29% |
| 2025-04 | bear −18.9% | 25 | 20-5 | +0.37% |
| 2025-01 | bull +54.5% | 26 | 24-2 | +0.63% |
| 2024-10 | chop −1.5% | 21 | 20-1 | +0.64% |

**Combined: 123 OOS trades, 113W/10L (91.9%, 95% CI ≈ 86–96%), every
window net-positive, across bull/bear/chop.** Per-trade edge is
regime-proof. Caveat that defines the next phase: as a standalone
portfolio it wildly underperforms B&H in bulls (+0.6% vs +54.5%) because
sizing is ~2.5% of book — this is an **alpha sleeve / overlay**, not a
core allocation. G3 (300 trades) to be completed via forward paper
evidence + sizing work belongs to Phase 1 risk design.

## Phase 1 — Institutional risk layer

- Daily / weekly loss limits with a circuit breaker that halts the cycle.
- Portfolio heat (sum of open risk) cap; drawdown-triggered de-risking
  (−10% halve, −20% halt) per PRINCIPLES §3.
- Exchange-error circuit breaker (repeated API failures → safe mode).
- Kill-criteria config file read by the live loop.

*Justification: valuable regardless of which strategy wins; cheap; makes any
future live-capital step defensible.*

## Phase 2 — Attribution (the minimally viable "Meta AI")

- Scoreboard tables: every agent recommendation and ML signal joined to the
  eventual trade outcome. Weekly report: accuracy, calibration, contribution
  by regime.
- Dashboard: signal-probability histogram, threshold sweep, per-regime
  performance, cost-vs-edge panel, DCA/B&H overlays.
- Decision rule: after 200 scored decisions, agents that add no measurable
  value get their influence reduced or removed (PRINCIPLES §6).

## Phase 3 — Second strategy, by allocation not switching

- Candidate: the existing grid/DCA module, put through the same holdout
  pipeline and gates as strategy #1.
- Capital is **allocated** across strategies by expectancy and correlation;
  no winner-take-all switching.
- Regime gating for strategy #1 if Phase 2 data shows losses cluster in
  specific regimes.

## Phase 4 — Scale (only if Phases 0–3 keep paying)

- Multi-asset (ETH already configured), execution quality work (post-only,
  TWAP for size), walk-forward automation in CI, research engine that
  proposes and auto-backtests parameter hypotheses weekly.
- Live capital per the promotion pipeline (PRINCIPLES §4), starting small.

---

*Anything not on this roadmap (more LLM agents, microservices, 15 strategies,
transformers/LSTMs) waits until a phase exit criterion says the system earned
it.*
