# Changelog

## V0.7.0 — 2026-10-17

- Added composite strategy genomes that combine two distinct entry families with AND/OR/XOR logic; genetic crossover/mutation and the ExtraTrees active-learning feature vector now include structural composition.
- Implemented independent audit-side composition logic and added ledger parity tests for all three operators.
- Background search supports requests up to 500,000 candidates using 10,000-candidate chunks, global SHA-256 genome deduplication, atomic batch checkpoints, and a resume endpoint.
- Full 500k production performance, resource limits, and end-to-end resume testing remain acceptance tasks; the version does not claim that such a full run has already passed.


## V0.6.0 — 2026-10-17

- Expanded the executable strategy catalogue from 18 to 30 with 12 price-action/regime-based families, implemented separately in the primary engine and independent audit engine.
- Added Persian family labels and enabled the new families in both manual strategy selection and search-family selection.
- Added family signal-generation and independent-ledger parity test coverage.
- Added background batch search up to 500,000 requested candidates, 10,000-candidate chunks, global SHA-256 genome deduplication, atomic SQLite checkpoint writes, and a resume endpoint for interrupted/failed jobs. The synchronous search endpoint intentionally remains capped at 10,000. A full 500k run has not yet been performance-tested.


## V0.6.0 — 2026-10-17

- Expanded the executable strategy catalogue from 18 to 30 with 12 price-action/regime-based families, implemented separately in the primary engine and independent audit engine.
- Added Persian family labels and enabled the new families in both manual strategy selection and search-family selection.
- Added family signal-generation test coverage. This release does not claim the 500k search target; search remains capped at 10,000 until resumable batch/checkpoint infrastructure is complete.


## V0.5.1 — 2026-10-17

- Persisted asynchronous search and Binance-download job state/progress to SQLite.
- Restore persisted job records after restart and mark unfinished queued/running jobs as interrupted rather than implying they are still running.
- Added a round-trip test for persistent job status and exposed the capability through `/api/health`.
- Fixed fail-closed search reporting: if apparent passing candidates fail independent ledger replay, the engine continues auditing ranked non-passing candidates and returns verified near-misses where available.
- This is not pause/resume: interrupted computations must be started again. Search remains capped at 10,000 candidates; 500k+ search is not implemented or certified.


## V0.5.0 — 2026-10-17

- Bumped FastAPI metadata and `/api/health` version to `0.5.0`.
- Added an explicit `VERSION` file and cache-busted frontend asset URLs.
- Made the canonical trade ledger interactive: click or keyboard-select a trade to center its entry on the server-rendered chart window, highlight its entry/exit and SL/TP levels, and scroll the chart into view.
- Added previous/next trade navigation that follows the active ledger filter.
- Added a distinct TP-only filter based on the ledger's actual `exit_reason == "TP"`.
- Added API tests for selected-trade chart centering and unknown trade IDs.
- Search remains capped at 10,000 candidates. The 500,000+ resumable, checkpointed search architecture is not part of this release.
- NetRun deployment is not claimed until the live `/api/health` endpoint reports version `0.5.0` and a smoke test passes.

## V0.4.0

Previous foundation release; see README for implemented capabilities and remaining limitations.
