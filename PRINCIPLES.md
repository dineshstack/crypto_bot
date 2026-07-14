# PRINCIPLES — The Constitution of This System

This document governs every change to the platform. It never describes *what*
to build next (see ROADMAP.md); it defines the rules any change must obey.
If a proposed change conflicts with this document, the change is wrong.

---

## 1. Core philosophy

- We estimate **probabilities, uncertainty and expected value** — never certainty.
- Every trade decision must answer: is it statistically favorable, what is the
  downside, what is the confidence, what historical evidence supports it, and
  how does it affect the portfolio?
- **If expected value is unclear, do nothing. Cash is a position.**
- Capital preservation outranks return generation. Minimizing drawdown
  outranks maximizing win rate.
- No AI output is trusted unverified. Every model and agent is scored against
  realized outcomes.

## 2. Statistical evidence gates (numeric, non-negotiable)

These numbers exist because this system once printed an in-sample Sharpe of
10.55 that evaporated to 0.53 out-of-sample (2026-07-14).

- **G1 — Out-of-sample only.** No performance claim counts unless the model
  never saw the evaluation candles (`backtester.py --holdout`).
- **G2 — Multiple windows.** A parameter or strategy change is accepted only
  if it improves results in **≥ 3 disjoint holdout windows** spanning
  different regimes.
- **G3 — Sample size.** No strategy goes to live capital with fewer than
  **300 out-of-sample trades** of evidence. Win-rate claims require a 95%
  binomial CI that excludes 50%.
- **G4 — Costs included.** All results are net of fees + slippage. Any edge
  whose gross expectancy is < 2× round-trip cost is treated as no edge.
- **G5 — Benchmarks.** Every result is compared against buy-and-hold **and**
  weekly DCA over the same window. Beating cash is not enough.
- **G6 — No survivor tuning.** Thresholds and hyperparameters are chosen on
  training/validation data only; a window used for tuning may never be
  reported as evidence.

## 3. Risk hierarchy

Risk has absolute veto. In order of priority:

1. Probability of ruin ≈ 0 (position sizes capped; fractional Kelly, never
   full Kelly; sizing discounts win rate by 10pp).
2. Max drawdown limits with automatic de-risking (halve size beyond −10%
   equity; halt beyond −20% until human review).
3. Daily / weekly loss limits with circuit breakers.
4. Exposure caps (per-asset and total crypto allocation).
5. Only then: return maximization.

## 4. Promotion pipeline

Backtest (holdout) → paper/testnet → small live capital → scaled capital.
No stage may be skipped. Each promotion requires the gates in §2 plus a
defined **kill criterion** written down *before* promotion (the drawdown or
evidence level at which the strategy is demoted).

## 5. Engineering rules

- **Smallest change that produces a measurable improvement.** Complexity must
  pay rent in measured performance, reliability, or insight.
- The system remains a **modular monolith** (ml_signal, risk_manager,
  executor, portfolio, database, agents) until scale demands otherwise.
  No microservices, queues, or CQRS without a measured bottleneck.
- Every module states its inputs, outputs, failure modes, and is observable
  (structured logs + DB events). Silent fallbacks are bugs — a degraded
  component must log loudly (see the zero-filled-features incident).
- Live state never leaks into simulations; simulations never mutate live
  state. (Backtests use frozen risk stats and their own model files.)
- Errors that repeat (see `bot_events`) are defects with owners, not noise.

## 6. AI usage rules

- ML models output **probabilities with calibration and validation-derived
  decision thresholds** — never hardcoded gates, never final decisions.
- LLM agents advise; only the decision layer recommends; only the executor
  (under risk veto) acts.
- Every agent call is logged with its input, output, and eventual outcome so
  attribution is computable. An agent whose contribution cannot be measured
  after 200 decisions is a candidate for removal.
- Retraining happens on schedule or on measured drift — never mid-drawdown
  panic.

## 7. Learning

Three levels, all persisted: trade lessons (why did this trade lose?),
strategy lessons (does the edge hold across regimes?), portfolio lessons
(allocation, correlation, exposure). A learning feature that stores nothing
(the lessons table was empty for its first month) is a broken feature.
