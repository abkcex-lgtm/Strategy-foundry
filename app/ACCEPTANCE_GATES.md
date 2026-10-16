# Acceptance gates

- [x] Server-side API and static Persian UI
- [x] OHLCV validator and SHA-256 fingerprints
- [x] Fixed-contract single-position backtest and canonical ledger
- [x] Same-candle SL/TP ambiguity fails closed
- [x] CSV ledger and replay-wrapper export with hash guard
- [x] Unit tests authored
- [ ] Run CI and record passing build
- [ ] Independent second engine and trade-by-trade reconciliation
- [ ] Resolve ambiguous 1m bars using 1s Binance data, failing if unavailable
- [ ] Enforce exact net -$10 SL and +$35 TP after all costs; current sizing is an estimate
- [ ] Search 500,000+ candidates under measured resource budget
- [ ] Train/dev/OOS split enforcement and untouched OOS report
- [ ] Exactly five near-misses when none pass, ranked with OOS proximity
- [ ] Two-year 1m performance and memory benchmark
- [ ] Persistent database and authenticated user-owned runs
- [ ] Interactive candlestick chart, pan/zoom, replay, indicators and drawing tools
- [ ] AI assistant, voice/files/media, run history and guarded bug-fix workflow
- [ ] Truly standalone Python export and trade-by-trade replay reconciliation
- [ ] Deployment smoke test and persistence check

A checked box means code exists, not profitability is established. Unit tests have not been executed by this file-authoring action.
