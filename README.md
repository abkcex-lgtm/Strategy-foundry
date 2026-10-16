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

Default preset preserves the agreed starting profile: $1,000 initial balance; $10 fixed risk; R:R 3.5; commission 0.04% + slippage 0.02% per side (0.06% total); one position; next-candle-open entry; SL/TP-only exit; target measured net of modeled costs. The preset is editable, not a factory-wide restriction.

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

## Free-tier deployment
A Render blueprint is included. Free instances may sleep, have resource/time limits, and may not provide durable local disk. The app uses SQLite by default; set FOUNDRY_DB_PATH to a mounted persistent path or switch to a managed database for durable storage. Managed databases may not remain free forever. GitHub alone does not execute the Python backend.

## Status
Implemented in this foundation: server-side API, OHLCV checks, Binance archive download path, editable settings/profile validation, SQLite dataset/run/profile persistence, risk/cost-aware ledger, bounded candidate search, OHLC candlestick view with trade overlays and paging, CSV and standalone Python replay export, an independent replay/audit endpoint, run history, and automated tests. SQLite durability depends on the host disk; free ephemeral deployments can lose data after restart/redeploy.
Not yet certified/complete: 500k+ candidate search, 1-second ambiguity resolution, standalone-export execution and exact ledger equality on real source data, smooth zoom/pan/replay/drawing tools, indicator selection beyond the current EMA crossover family, full authentication/multi-user isolation, complete AI/voice/files engineer workflow, long-run benchmarks, and deployment smoke test. The independent audit engine has a synthetic-data unit test; this is not a market-data certificate. See app/ACCEPTANCE_GATES.md.

Backtests can overfit and do not guarantee future returns. Never add exchange API keys with trading permissions.
