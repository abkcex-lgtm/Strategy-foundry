# Changelog

## V0.5.0 — 2026-10-17

- Bumped FastAPI metadata and `/api/health` version to `0.5.0`.
- Added an explicit `VERSION` file and cache-busted frontend asset URLs.
- Made the canonical trade ledger interactive: click or keyboard-select a trade to center its entry on the server-rendered chart window, highlight its entry/exit and SL/TP levels, and scroll the chart into view.
- Added a distinct TP-only filter based on the ledger's actual `exit_reason == "TP"`.
- Added API tests for selected-trade chart centering and unknown trade IDs.
- Search remains capped at 10,000 candidates. The 500,000+ resumable, checkpointed search architecture is not part of this release.
- NetRun deployment is not claimed until the live `/api/health` endpoint reports version `0.5.0` and a smoke test passes.

## V0.4.0

Previous foundation release; see README for implemented capabilities and remaining limitations.
