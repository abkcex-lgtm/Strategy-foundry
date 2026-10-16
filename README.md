# Strategy Foundry

Reproducible, server-side strategy research and paper/backtest validation. **Simulation only: no exchange orders are sent.** This repository is a working foundation, not a claim of profitable or fully validated strategies.

## Principles
- Fail closed on malformed, missing, duplicated, out-of-order, or ambiguous data.
- Never fill future values backward; indicators use past/current candles only.
- One canonical trade ledger drives metrics and chart markers.
- Keep train (Apr-Jun 2026), development (Jul 2026), and untouched OOS (Aug-Sep 2026) separate.
- Identify each run with deterministic configuration and data hashes.
- Defaults: initial balance $1,000; risk $10; reward/risk 3.5; max one open position; next-candle-open entry; fee+slippage 0.06% per side; SL/TP-only exits.
- No synthetic candles. If same-candle SL/TP ambiguity cannot be resolved, the run fails.

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
A Render blueprint is included. Free instances may sleep, have resource/time limits, and may not provide durable local disk. Configure managed PostgreSQL for reliable persistent multi-user storage; it may not remain free forever. GitHub alone does not execute the Python backend.

## Status
Implemented in this foundation: API shell, deterministic OHLCV checks, Binance archive download path, fixed-contract backtest ledger, metrics, bounded candidate search, JSON/CSV exports, browser trade markers, and unit tests.
Not certified: exhaustive 500k+ search, independent second-engine reconciliation, 1-second ambiguity resolution, exact net SL/TP reconciliation, long-run benchmarks, durable storage, AI provider integration, and full candlestick/replay parity with Binance. See app/ACCEPTANCE_GATES.md.

Backtests can overfit and do not guarantee future returns. Never add exchange API keys with trading permissions.