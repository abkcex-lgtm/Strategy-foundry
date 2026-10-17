---
title: Strategy Foundry
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Strategy Foundry

A server-side, simulation-only strategy research foundation. It never sends live exchange orders. Passing unit tests do not imply profitability or full validation.

## Editable contract (not hard-coded)

The Settings panel sends these values into backtesting and candidate search, and each run fingerprint includes the full configuration:
- Starting balance
- Risk model: fixed USD, percent of initial balance, or percent of current equity
- Fixed risk amount, risk percentage, and optional compounding
- Reward/risk ratio and target basis: net after modeled costs or gross before costs
- Commission and slippage as separately configurable percentages per side
- Win-rate, profit-factor, TP/day, and maximum-loss-streak acceptance thresholds
- Entry/exit model and optional trade-management flags

Default preset preserves the agreed starting profile: $1,000 initial balance; $10 fixed risk; R:R 3.5; commission 0.04% + slippage 0.02% per side (0.06% total); one position; next-candle-open entry; SL/TP-only exit; target measured net of modeled costs. The preset is editable, not a factory-wide restriction. Each acceptance filter (WR, PF, TP/day, and loss streak) can independently be enabled or disabled; disabled filters do not affect pass/fail or near-miss distance. The active values and enabled flags are included in the run fingerprint.

The net target model sizes quantity so modeled SL net loss equals the selected risk budget and TP net gain equals risk budget × R:R, assuming the configured percentage cost model. Real slippage is not a constant in live markets; users must select realistic estimates and independently validate them. Management toggles that are not implemented are rejected fail-closed rather than silently ignored.

## Principles

- Fail closed on malformed, missing, duplicated, out-of-order, or ambiguous data.
- No future fill/backfill in indicators; indicators use past/current candles only.
- One canonical trade ledger drives metrics, exports, and chart annotations.
- Reference split: train (Apr-Jun 2026), development (Jul 2026), and OOS (Aug-Sep 2026); search uses Train only.
- Each run has deterministic strategy/config/data hashes.
- No synthetic market candles. Unresolved same-candle SL/TP ambiguity stops the run.

## Run locally

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. API schema: /docs.

## Deployment

Vercel serverless deployment files (`api/index.py` and `vercel.json`) are included for a free Hobby-plan deployment attempt, alongside the Render Blueprint and Dockerfile. GitHub does not run the backend itself. The repository may be connected to a separate host (for example, a Netrun deployment); after a commit, confirm that host rebuilt the matching branch/commit and recheck `/api/health`, `/api/ai/status`, and `/api/datasets`. A healthy `/api/health` alone does not prove data download, persistence, backtest, or AI-provider configuration works.

Important limitations of the Vercel serverless trial: SQLite is stored in temporary `/tmp`, in-memory datasets/runs may not survive between separate function instances, and long searches or multi-month data downloads may exceed serverless limits. This is an initial online trial path, not durable production hosting or a certified full factory. Do not enter payment details if a host unexpectedly requests a card.

## Multi-provider AI fallback

The operational assistant now has a server-side, OpenAI-compatible provider router. It tries configured providers in the order given by `AI_PROVIDER_ORDER` and automatically tries the next provider after a request failure. Supported provider slots are OpenAI, Groq, OpenRouter, Cerebras, and a custom OpenAI-compatible endpoint. Status is available at `/api/ai/status`; chat is at `POST /api/ai/chat`.

Configure provider keys as host-side secret environment variables, never in frontend files or Git:
- `OPENAI_API_KEY` and optional `OPENAI_MODEL`
- `GROQ_API_KEY` and optional `GROQ_MODEL`
- `OPENROUTER_API_KEY` and optional `OPENROUTER_MODEL`
- `CEREBRAS_API_KEY` and optional `CEREBRAS_MODEL`
- Optional custom endpoint: `AI_ENDPOINT`, `AI_API_KEY`, and `AI_MODEL`
- Optional order, e.g. `AI_PROVIDER_ORDER=groq,openrouter,cerebras,openai,custom`

The router does not create API access or credits. Unconfigured providers are skipped, and when no key is configured the API reports that clearly. The assistant is not yet an autonomous source-code repair agent; ordinary chat must not be described as code changes or tests being executed.

## V0.7.1 release notes

- Added a recovery panel to find queued, running, interrupted, or failed background searches after reopening the app; interrupted jobs can resume from the last atomically completed batch.
- Updated the candidate-count input to allow up to 500,000 and extended client polling for long background jobs. The full 500k run remains unmeasured in production.

## V0.7.0 release notes

- Search can generate structurally compositional strategy genomes: two distinct strategy families combined with AND, OR, or XOR logic. Composition is part of the genome hash, crossover/mutation, ML feature representation, manual backtest settings, and independent audit replay.
- Background search accepts up to 500,000 candidates in 10,000-candidate batches, persists each completed batch atomically, globally deduplicates genome fingerprints, and exposes resume after interruption. The synchronous endpoint remains capped at 10,000.
- Added tests for composition ledger parity, global genome deduplication, 500k request validation, and atomic checkpoint persistence. A full 500k production-scale run is not yet measured or certified.

## V0.6.0 release notes

- Added 12 independently replayable price-action/regime families: swing-structure break, failed-breakout reversal, wick rejection, range expansion, three-bar momentum, volatility-regime breakout, trend-exhaustion reversal, volume-climax reversal, range-compression breakout, support/resistance rejection, micro double bottom, and micro double top. The engine and independent audit implement separate signal calculations; UI exposes them in manual and search family controls.
- Added signal-generation coverage for all 30 families. The background search endpoint accepts up to 500,000 candidates in 10,000-candidate batches, persists each completed batch atomically, deduplicates genome fingerprints globally, and offers resume after interruption. The synchronous endpoint remains capped at 10,000. A full 500k production run has not yet been performance-tested or certified.

## V0.6.0 release notes

- Added 12 independently replayable price-action/regime families: swing-structure break, failed-breakout reversal, wick rejection, range expansion, three-bar momentum, volatility-regime breakout, trend-exhaustion reversal, volume-climax reversal, range-compression breakout, support/resistance rejection, micro double bottom, and micro double top. The engine and independent audit implement separate signal calculations; UI exposes them in manual and search family controls.
- Added signal-generation coverage for all 30 families. This expands structural diversity but is **not** yet the promised 500,000-candidate resumable search: the API remains capped at 10,000 until durable batch checkpoints, bounded-memory result storage, cancellation/resume, and full-run validation are implemented.

## V0.5.1 release notes

- Search and Binance-download job status/progress are written to SQLite as jobs advance, so the latest visible state remains available after a process restart on hosts with a durable SQLite volume.
- If a process restarts while a job was queued or running, the job is restored as **interrupted** instead of falsely appearing active. This is status persistence, **not** resumable execution; the computation itself must be started again.
- `/api/health` reports `persistent_job_status: true`, and the API/UI version is `0.5.1`.
- Historical note for v0.5.1: the search ceiling was 10,000, without resumable checkpoints.

## V0.5.0 release notes

- The API version and health endpoint now report **0.5.0**.
- Selecting a row in the canonical trade ledger requests a server-side chart window centered on that trade's entry, highlights its entry/exit and SL/TP levels, and scrolls the chart into view. Keyboard Enter/Space selection and previous/next trade navigation are supported.
- Added a distinct **TP-only** ledger filter; it filters by actual `exit_reason == TP` rather than assuming every profitable trade was a TP.
- Versioned frontend asset URLs (`?v=0.5.0`) reduce stale browser-cache confusion.
- This release does **not** claim 500,000-candidate search, durable pause/resume checkpoints, a completed full strategy catalogue, or verified NetRun deployment. Search remains capped at 10,000 until bounded, checkpointed search is implemented and tested.

## Status

Current API/workspace version: **0.7.1**. The header reports the live API version; verify `/api/health` returns `0.5.0` to confirm the server update. The workspace header reads the running server version from `/api/health`.

Implemented in this foundation: server-side API, OHLCV checks, Binance archive download path, tracked download and asynchronous search jobs with status polling, TRAIN-only search selection, persisted source provenance and data SHA-256 fingerprints, live dashboard summary of data/risk/filters/fees, editable settings/profile validation, SQLite dataset/run/profile/search-result persistence with saved search manifests and reopenable result history, risk/cost-aware ledger, deterministic genetic search with crossover/mutation/stratified seeds/immigrant exploration, an ExtraTrees surrogate-guided proposal mode, a stratified-random baseline, user-selectable family/context/confirmation search space, and 18 causal signal families (EMA crossover, channel breakout, RSI reversion, volume breakout, EMA pullback, MACD crossover, Bollinger reversion, inside-bar breakout, engulfing reversal, momentum breakout, stochastic reversion, ROC momentum, ADX/DMI trend, Keltner breakout, volatility squeeze breakout, OBV trend, VWAP reclaim, and support/resistance pivot reversal). Reported search results are independently replayed against a separate ledger engine; UI exposes search mode and family labels. Other implemented elements include per-filter failures/distances, OHLC candlestick view with trade overlays, paging/zoom presets, clickable/keyboard-selectable ledger rows that focus a selected trade on the chart, TP-only/winner/loser filters, CSV and standalone Python replay export, run history and reproducibility manifests.

Not yet complete/certified: 500k+ candidate capacity (current hard limit remains 10,000 until resumable resource-bounded batches/checkpoints are built), full typed genetic-programming grammar and predictive ML signal models (the current ML component is a search-space surrogate, not a market-direction predictor), the full 17-group strategy catalogue in docs/strategy-space-and-research-contract.md, 1-second ambiguity resolution, standalone-export execution and exact ledger equality on real market data, full replay-causal audit, multi-user authentication/isolation, complete AI/voice/files engineer workflow, long-run benchmarks, and deployment smoke test. Eighteen families now have primary signal logic and independent replay logic with synthetic causal/reconciliation tests; this is not a market-data profitability certificate. See app/ACCEPTANCE_GATES.md.

Backtests can overfit and do not guarantee future returns. Never add exchange API keys with trading permissions.


## Free public hosting path (Docker Space)

This repository includes a Docker-based FastAPI app and can run as a Hugging Face Docker Space on its free CPU tier. A Space must be created under the owner's Hugging Face account and linked/imported from this GitHub repository; this repository cannot create that account or authorize hosting on the owner's behalf.

1. Open [Create a new Hugging Face Space](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK, set visibility to **Public**, and create the Space.
3. Connect/import this repository (`abkcex-lgtm/Strategy-foundry`) as the Space source and keep the Dockerfile at the repository root.
4. Wait for the Docker build, then open the Space URL and verify `/api/health` reports `status: ok`.

**Hosting caveats:** free Spaces can sleep when idle and their local filesystem is not a guaranteed durable database. This makes the first hosted build useful for a smoke test, not a promise of durable production storage. Do not put exchange trading keys into this simulation-only app.
