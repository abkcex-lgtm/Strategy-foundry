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

## Status

Implemented in this foundation: server-side API, OHLCV checks, Binance archive download path, tracked download and asynchronous search jobs with status polling, TRAIN-only search selection, persisted source provenance and data SHA-256 fingerprints, editable settings/profile validation, SQLite dataset/run/profile persistence, risk/cost-aware ledger, bounded candidate search across four causal signal families (EMA crossover, channel breakout, RSI reversion, and volume-confirmed breakout), OHLC candlestick view with trade overlays, paging, zoom presets, and latest-candle navigation, CSV and standalone Python replay export, an independent replay/audit endpoint, run history, and automated tests.

Not yet certified/complete: 500k+ candidate search, 1-second ambiguity resolution, standalone-export execution and exact ledger equality on real data, smooth zoom/pan/replay/drawing tools, broader candlestick/price-action/support-resistance/regime families beyond the four current signal families, full authentication/multi-user isolation, complete AI/voice/files engineer workflow, long-run benchmarks, and deployment smoke test. The independent audit engine has a synthetic-data unit test; this is not a market-data certificate. See app/ACCEPTANCE_GATES.md.

Backtests can overfit and do not guarantee future returns. Never add exchange API keys with trading permissions.


## Free public hosting path (Docker Space)

This repository includes a Docker-based FastAPI app and can run as a Hugging Face Docker Space on its free CPU tier. A Space must be created under the owner's Hugging Face account and linked/imported from this GitHub repository; this repository cannot create that account or authorize hosting on the owner's behalf.

1. Open [Create a new Hugging Face Space](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK, set visibility to **Public**, and create the Space.
3. Connect/import this repository (`abkcex-lgtm/Strategy-foundry`) as the Space source and keep the Dockerfile at the repository root.
4. Wait for the Docker build, then open the Space URL and verify `/api/health` reports `status: ok`.

**Hosting caveats:** free Spaces can sleep when idle and their local filesystem is not a guaranteed durable database. This makes the first hosted build useful for a smoke test, not a promise of durable production storage. Do not put exchange trading keys into this simulation-only app.
