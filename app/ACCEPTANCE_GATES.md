# Acceptance gates

A checked box means implementation exists; it does not establish profitability or global completeness.

- [x] Server-side API and Persian responsive UI
- [x] OHLCV validation and SHA-256 fingerprints
- [x] Editable contract settings: initial balance, fixed/percentage risk, compounding, net/gross R:R, commission, slippage, pass thresholds
- [x] Settings passed into backtest and candidate search; settings included in run fingerprint
- [x] Exact modeled net-risk/net-reward sizing for the selected SL/TP prices
- [x] Unsupported trade-management toggles fail closed instead of silently changing behavior
- [x] Same-candle SL/TP ambiguity fails closed
- [x] CSV ledger and self-contained Python replay template
- [x] Unit tests for editable contract, exact net risk/reward sizing, API smoke paths
- [x] GitHub Actions passed for the configurable-settings/test commit (cfda2eb3)
- [ ] Independent second engine and trade-by-trade reconciliation
- [ ] Resolve ambiguous 1m bars using 1s Binance data, failing if unavailable
- [ ] Slippage model with direction/volatility/order-type sensitivity; current setting is a configurable percentage approximation
- [ ] Search 500,000+ candidates under measured resource budget
- [ ] Train/dev/OOS evaluation with frozen strategy and truly untouched OOS report
- [ ] Exactly five near-misses when none pass, ranked by OOS proximity
- [ ] Two-year 1m performance and memory benchmark
- [ ] Persistent database and authenticated user-owned runs/settings
- [ ] Full candlestick chart, pan/zoom, replay, indicators, drawing tools, all setup annotations
- [ ] AI assistant, voice/files/media, run history, and guarded bug-fix workflow
- [ ] Standalone export replay run on exact source CSV and ledger equality verified
- [ ] Deployment smoke test and persistence check
