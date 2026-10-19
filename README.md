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

Application release: 0.9.7

Research-engine line: 0.9.9 (engine-generation line; not the API/UI release version)

**Canonical handoff / project status:** read [PROJECT_STATE.md](PROJECT_STATE.md) first when continuing work in a new ChatGPT conversation. It records the authoritative branch, current run status, non-negotiable research contract, and next engineering priorities.

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

## Current implementation and verification status

- Current research-engine line: **v0.9.9**. The canonical handoff is [PROJECT_STATE.md](PROJECT_STATE.md); read it first when continuing in a new chat.
- The main branch has 35 executable strategy signal families, compositional two-family genomes (AND/OR/XOR), genetic/surrogate-guided search, duplicate avoidance, family-aware sampling, checkpointed batch orchestration, progress heartbeats, and an independently implemented trade-ledger replay.
- Search can be configured up to 500,000 candidates through bounded batches, but this is a **requested capacity**, not proof that a full 500,000-candidate run has completed within the runtime budget.
- Latest verified research run: [GitHub Actions run 39277059928](https://github.com/abkcex-lgtm/Strategy-foundry/actions/runs/39277059928), a 1,000-candidate real-data smoke test. It completed successfully, passed data preflight, and found **zero strategies passing every required split**; five near-misses were returned. The full 500k search is not yet verified.
- The reference data split is Train Apr–Jun 2026, Development Jul 2026, and untouched OOS Aug–Sep 2026. OOS must not guide candidate generation, ranking, or selection.
- A successful CI run means the checks completed; it does not certify profitability, validate a near-miss, or prove a production-scale search has finished.
- Current non-negotiable acceptance criteria, audit requirements, measured-speed plan, progress requirements, and repository/branch hygiene rules are maintained in [PROJECT_STATE.md](PROJECT_STATE.md).
- The assistant/provider router is not an autonomous code-repair agent. Do not claim that ordinary chat has modified code, executed tests, or repaired a deployment unless those actions were actually performed and verified.
- Backtests can overfit and do not guarantee future returns. This application is simulation-only; never add exchange API keys with trading permissions.

## Free public hosting path (Docker Space)

This repository includes a Docker-based FastAPI app and can run as a Hugging Face Docker Space on its free CPU tier. A Space must be created under the owner's Hugging Face account and linked/imported from this GitHub repository; this repository cannot create that account or authorize hosting on the owner's behalf.

1. Open [Create a new Hugging Face Space](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK, set visibility to **Public**, and create the Space.
3. Connect/import this repository (`abkcex-lgtm/Strategy-foundry`) as the Space source and keep the Dockerfile at the repository root.
4. Wait for the Docker build, then open the Space URL and verify `/api/health` reports `status: ok`.

**Hosting caveats:** free Spaces can sleep when idle and their local filesystem is not a guaranteed durable database. This makes the first hosted build useful for a smoke test, not a promise of durable production storage. Do not put exchange trading keys into this simulation-only app.
