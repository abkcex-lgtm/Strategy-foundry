# Changelog

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
