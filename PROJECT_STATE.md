# Strategy Foundry — Canonical Project State

> This file is the single handoff source for any new ChatGPT conversation, contributor, or coding agent. Read this before changing code. The `main` branch is the canonical product line; old feature/research branches are historical work unless explicitly reopened by a current issue or pull request.

## Current source of truth

- Repository: https://github.com/abkcex-lgtm/Strategy-foundry
- Canonical branch: `main`
- Current research engine workflow: `.github/workflows/research.yml`
- Research entry point: `research/run_factory.py`
- Primary strategy generation/backtest engine: `app/engine.py`
- Independent audit/replay engine: `app/audit.py`
- User-facing API/application: `app/main.py` and the surrounding `app/` package
- Latest merged engine expansion: PR #30 (heartbeat, family balancing, five new strategy families, independent replay parity tests).
- Live-trading orders are not allowed; the app is for research/simulation only.

## Latest verified CI/research state (2026-10-19)

- Main commit `c28e7c43bf153cf6ebfc8d7fd6a562731eb767ff` completed unit tests and fast validation successfully.
- Research run [39277059928](https://github.com/abkcex-lgtm/Strategy-foundry/actions/runs/39277059928) completed successfully as a **bounded 1,000-candidate smoke test**, not a full search.
- Dataset preflight found 263,520 Binance USD-M ETHUSDT 1m rows for Apr–Sep 2026, expected monthly row counts, and zero timestamp gaps; dataset SHA-256: `f2c0bc9ba360a319f8b475322e111b1cbd9d1696183ed38d416e52b266cb22bb`.
- Result: 1,000 unique candidates evaluated, zero candidates passed every required split, and five near-misses returned. This is not evidence that the full 500,000-candidate search has been completed.
- Earlier full run [39268025296](https://github.com/abkcex-lgtm/Strategy-foundry/actions/runs/39268025296) was **cancelled** at 15:20 UTC on 2026-10-19. Do not describe it as completed or use its partial candidate count as a final result.
- Inspect the latest Actions run and its artifact before asserting new search performance. A successful workflow means the run completed, not that a profitable strategy was found.

## User's acceptance contract — do not silently weaken

Default research preset:
- Binance USDⓈ-M ETHUSDT perpetual, 1m OHLCV.
- Initial balance $1,000; fixed risk $10/trade; no compounding.
- At most one open position; entry at next candle open.
- SL/TP-only exits; no break-even, trailing, partial, time, or manual exits unless separately implemented, explicitly enabled, and independently tested.
- Commission 0.04% + slippage 0.02% per side = 0.06% per side and 0.12% round trip under the configured model.
- Net reward/risk target 3.5R; target net TP +$35 and net SL -$10 under the intended default model.
- Loss streak resets only after a full profitable TP.
- Required conditions simultaneously: win rate >40%, profit factor >1, average net TP/day >=3, maximum consecutive losses <=13. Configurable filters in the app must only apply when explicitly disabled/enabled by the active configuration; never silently change the agreed default.
- No lookahead, no synthetic market candles, no future fill/backfill. No OOS-guided selection. Train Apr–Jun 2026, development Jul 2026, untouched OOS Aug–Sep 2026 when the reference dataset is used.
- Preflight, ambiguity resolution, independent replay, canonical-ledger metrics, hashes, and trade-by-trade reconciliation must remain fail-closed.
- If no strategy passes, report exactly five meaningful near-misses with actual metrics and failed criteria. Never label a near-miss as validated.

## Strategy-search architecture direction

Do not try to hand-maintain hundreds of thousands of nearly duplicate “families.” Build a large *compositional search space*:
1. Independently implemented signal primitives: trend, momentum, mean reversion, volatility, volume/flow, candle patterns, market structure, breakouts, pullbacks, failed breakouts, support/resistance, and regime detection.
2. Typed parameters and valid parameter ranges for every primitive; each primitive must have executable semantics, tests, and independent audit support.
3. Strategy grammar/genomes combining primitives, long/short direction, entry filters, confirmations, and stop/target parameters. Use controlled AND/OR compositions, crossover/mutation, novelty, deduplication, and diversity quotas.
4. Search steering uses Train/Development only. OOS is used only for final evaluation after candidate selection is frozen.
5. Report family/primitive coverage and candidate counts, not just total count. Prevent one family from dominating all evaluations.
6. Optimize throughput only after profiling. Cache reusable causal indicator arrays and avoid per-candidate full-frame copies where safe. Any faster simulator must be trade-for-trade equivalent to the reference simulator on regression fixtures and real-data samples before it is trusted.
7. Never claim that a strategy was “found” merely because a search generated it. Candidate generation, development screening, independent audit, OOS evaluation, and certification are distinct states.

## Progress/observability requirements

Every running research job should expose durable, atomically written progress including:
- status, run ID, start/last-heartbeat UTC timestamps;
- requested/evaluated unique candidates, completed batches, current family, family counts, duplicate skips, errors;
- elapsed seconds, recent and overall candidates/second, current phase;
- data/config/source hashes and checkpoint location;
- terminal outcome and reason for any stop.
A watchdog must distinguish a genuinely stale heartbeat from a long but active calculation. Flush human-readable heartbeat logs. A GitHub Actions green check alone is not a progress report.

## Speed and correctness priorities

1. Reproduce the latest run from its artifact and inspect where time is spent.
2. Add benchmarks for indicator preparation, signal generation, trade simulation, and independent audit separately.
3. Establish a baseline on the same real dataset and same seed, then optimize the measured bottleneck.
4. Keep a reference path and compare every trade field, aggregate metric, and final ledger hash. Fail closed on mismatch.
5. Only after measured equivalence, scale from smoke tests to larger runs (e.g. 10k, 50k, then 500k) with checkpoint/resume. Never promise “minutes” until measured.
6. A search returning no winners is a valid outcome; diagnose family coverage, trade counts, and failed conditions rather than relaxing thresholds.

## Repository and branch hygiene

- `main` is the only default handoff branch. Use short-lived branches named for one task and merge through a PR after tests pass.
- Do not create a branch for documentation-only handoff notes unless a PR is needed; do not leave duplicate “latest/final/new-final” versions.
- Before deleting an old branch, verify whether it contains unique unmerged work or an open PR. Do not delete branches based only on age or name.
- Do not rewrite tags/releases or delete data/results that provide audit evidence. Archive or mark obsolete historical work instead.
- Update this file when the canonical branch, run state, acceptance contract, or immediate next task changes.

## Immediate next tasks

1. Inspect and test the actual heartbeat callback contract; ensure updates happen during candidate evaluation rather than only after a batch.
2. Produce a reproducible benchmark and identify the true throughput bottleneck before further speed edits.
3. Add strategy primitives in audited batches, each with primary/independent parity tests and coverage metrics.
4. Add/verify a compositional genome/grammar that can explore far more combinations than the number of hand-coded families.
5. Re-run validation, then staged real-data searches with clear progress artifacts. Full 500k research remains unverified until a full run actually completes.
