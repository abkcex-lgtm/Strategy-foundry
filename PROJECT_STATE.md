# Strategy Foundry — Canonical Project State

> This file is the single handoff source for any new ChatGPT conversation, contributor, or coding agent. Read this before changing code. The `main` branch is the canonical product line; old feature/research branches are historical work unless explicitly reopened by a current issue or pull request.

## Release 0.9.9 objective policy and follow-up engineering (2026-10-20)

- Latest engineering intent: each active performance criterion (WR, PF, net full TPs/day, maximum consecutive losses) has equal weight via the arithmetic mean of normalized target distances. There are no per-metric hidden ranking bonuses or tie-break preferences.
- Separate eligibility gates: independent ledger match, data integrity, no open position at end, and sufficient sample evidence. Minimum evidence is 100 closed trades; with TP/day enabled, observed trade count must also support the configured TP/day goal at the observed win rate. When TP/day is disabled, it must not steer/block selection beyond the 100-trade floor.
- Code changes in engine, API ranking, research runner, tests, README, docs, changelog, VERSION, and UI are intended for release 0.9.9. Check latest CI and live /api/health before calling it deployed. Do not claim a new real-data search has run unless a fresh report/artifact proves it.
- Preserve existing SQLite data and prior run history. Before publishing/rebuilding, verify the hosting platform's durable storage behavior or establish a verified backup/restore path. Never disable live-trading safety (must remain false).

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

## Follow-up engineering changes (2026-10-20)

- Five additional causal strategy families were added to the primary signal engine, the separately implemented independent replay, the search grammar, and the family-coverage regression fixture: ATR-aware trend pullback, volume-weighted range breakout, range-midpoint reversion, ROC acceleration, and ATR-normalized momentum. The grammar now contains 40 executable signal families. The family definitions use only current/past data; rolling breakout reference levels and volume baselines are shifted to exclude the current candle where required.
- Added a parity regression test comparing primary and independent signals/ATR arrays for those five families. This test must pass before the families are considered verified.
- Changed the first-exit search from allocating boolean masks over the entire remaining dataset for every trade to scanning 4,096-candle vectorized chunks. This is a bounded-temporary-memory optimization intended to reduce memory pressure on long histories; do not claim a measured speedup until benchmarked against the prior implementation on the same dataset and seed.
- Added workflow commit `de1cbc7f15a155f4bb5d36745e5adc90ca546c2d` with commit message `[full-research]`; this should trigger the 500,000-candidate research workflow after its test gate. Confirm the actual workflow outcome/artifacts before calling the search complete. If the run fails, preserve the logs and report the exact failure rather than relaxing thresholds.
- Source changes above are in GitHub `main`; they have not yet been verified in the live NetRun deployment. Do not publish over the existing instance until the latest CI passes and the database persistence/backup path is confirmed.

## Latest engineering/run state (2026-10-20)

- Strategy-family count is now 40 in the primary engine, search grammar, and independent replay. Five new families have a dedicated primary-vs-independent signal parity test in `tests/test_search.py`.
- The first-exit scan now uses 4,096-candle chunks. A regression test compares it with the previous full-tail mask behavior across both directions and chunk boundaries.
- `research/run_factory.py` now benchmarks the old full-tail scanner against the chunked scanner on the actual TRAIN split, asserts exact first-hit parity, and records measured seconds/speedup plus the TRAIN data hash in the report. This is a component microbenchmark, not proof of whole-search throughput.
- Full-search trigger commit: `de1cbc7f15a155f4bb5d36745e5adc90ca546c2d` (`[full-research]`, requested 500,000 candidates, 19,500-second search budget). Its final status and artifact are not yet confirmed through the available tool responses.
- Smoke/benchmark trigger commit: `b3c437a41b6df7e6fc0379c99a053c04af7b0301`; this uses the pinned triggering commit SHA and requests the normal 1,000-candidate smoke path. Verify its test gate, benchmark result, and research artifact before treating it as successful.
- Research workflow checkout is now pinned to `${{ github.sha }}` so a run is reproducible against the commit that triggered it rather than whatever happens to be at `main` when the runner starts.
- These GitHub changes are not yet deployed to NetRun. Keep the live database untouched until the latest source checks pass and a backup/persistence path is verified.

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
