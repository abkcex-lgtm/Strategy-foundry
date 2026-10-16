# Acceptance gates

A checked box means implementation exists; it does not establish profitability or global completeness.

- [x] Server-side API and Persian responsive UI
- [x] OHLCV validation and SHA-256 fingerprints
- [x] Strict gap rejection for 1m archive downloads, CSV uploads marked as 1m, and backtests
- [x] Editable contract settings: initial balance, fixed/percentage risk, compounding, net/gross R:R, commission, slippage, pass thresholds
- [x] Each acceptance filter can be enabled/disabled independently; disabled filters are excluded from pass/fail and near-miss distance
- [x] Settings passed into backtest and candidate search; settings included in run fingerprint
- [x] Exact modeled net-risk/net-reward sizing for the selected SL/TP prices
- [x] Unsupported trade-management toggles fail closed instead of silently changing behavior
- [x] Same-candle SL/TP ambiguity fails closed
- [x] CSV ledger and self-contained Python replay template
- [x] Unit tests for editable contract, exact net risk/reward sizing, API smoke paths
- [x] GitHub Actions passed for the configurable-settings/test commit (cfda2eb3)
- [x] Independent second replay engine and trade-by-trade audit endpoint (synthetic-data test only; real-market certification pending)
- [ ] Resolve ambiguous 1m bars using 1s Binance data, failing if unavailable
- [ ] Slippage model with direction/volatility/order-type sensitivity; current setting is a configurable percentage approximation
- [ ] Search 500,000+ candidates under measured resource budget
- [ ] Train/dev/OOS evaluation with frozen strategy and truly untouched OOS report
- [ ] Exactly five near-misses when none pass, ranked by OOS proximity
- [ ] Two-year 1m performance and memory benchmark
- [x] SQLite persistence for datasets, run history, and editable profiles (durability depends on host disk; authentication/multi-user isolation remains open)
- [x] OHLC candlestick chart with trade SL/TP/entry/exit overlays and candle-window paging\n- [ ] Smooth pan/zoom, replay controls, indicator and drawing tools, all setup annotations for candidate-library strategies
- [x] Multi-provider chat router with automatic fallback across configured OpenAI, Groq, OpenRouter, Cerebras, and custom OpenAI-compatible endpoints
- [ ] AI assistant autonomous code repair, voice/files/media, and guarded bug-fix workflow
- [ ] Standalone export replay run on exact source CSV and ledger equality verified
- [ ] Deployment smoke test and persistence check
