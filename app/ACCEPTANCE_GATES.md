# Acceptance gates

A checked box means implementation exists; it does not establish profitability or global completeness.

- [x] Server-side API and Persian responsive UI
- [x] OHLCV validation and SHA-256 fingerprints
- [x] Persisted dataset source metadata, monthly Binance archive hashes, and exposed full normalized-data SHA-256 in dataset listing
- [x] Live dashboard summary for selected dataset/splits, risk, fees, active target filters, management toggles, and search configuration
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
- [x] Train/dev/OOS evaluation freezes one effective strategy/contract first, hashes each split, and emits a reproducibility manifest; OOS is evaluation-only and never feeds app-side selection (real-market certification remains pending)
- [x] When no TRAIN candidate passes, return exactly five independently audited near misses when five are auditable; rank the frozen near-miss pool on DEVELOPMENT and keep OOS untouched (honestly report any shortfall)
- [ ] Two-year 1m performance and memory benchmark
- [x] SQLite persistence for datasets, run history, and editable profiles (durability depends on host disk; authentication/multi-user isolation remains open)
- [x] SQLite persistence and UI/API reopening for full strategy-search results and reproducibility manifests
- [x] OHLC candlestick chart with trade SL/TP/entry/exit overlays, candle-window paging, zoom presets, latest-candle jump, and trades spanning the visible window\n- [ ] Smooth pan/zoom, replay controls, indicator and drawing tools, all setup annotations for candidate-library strategies
- [x] Tracked asynchronous strategy-search jobs with status polling; search and ranking explicitly use TRAIN only
- [x] SQLite-persisted search/download job status and progress; interrupted jobs are labeled after restart (execution resume/checkpoints remain open)
- [x] Four causal strategy families: EMA crossover, price-channel breakout, RSI reversion, and volume-confirmed breakout; independent replay updated for all four
- [x] Future-candle mutation regression test confirms past signals remain unchanged across all four current families
- [x] Candidate results include each failed active filter, per-filter distance, and transparent near-miss shortfall
- [x] Independent audit selects the exact saved Train/Dev/OOS data slice for split-evaluation runs; real-market replay certification remains pending
- [x] Multi-provider chat router with automatic fallback across configured OpenAI, Groq, OpenRouter, Cerebras, and custom OpenAI-compatible endpoints
- [ ] AI assistant autonomous code repair, voice/files/media, and guarded bug-fix workflow
- [ ] Standalone export replay run on exact real-market source CSV and non-empty ledger equality verified (generated-script/hash regression covered by synthetic CI fixture only)
- [x] Standalone replay embeds causal indicator logic for all 30 strategy families and AND/OR/XOR composition; gap-at-open stop/target fills and same-candle ambiguity handling align with the primary engine
- [ ] Deployment smoke test and persistence check

- [x] 12 additional price-action/regime signal families added to engine and independent audit; signal generation and ledger parity covered by parameterized tests (execution not yet CI-verified)
- [x] Two-family AND/OR/XOR composition included in genome, genetic variation, surrogate feature vectors, manual strategy settings, and independent audit
- [x] Background batch search request up to 500,000 candidates, global genome fingerprint deduplication, atomic completed-batch checkpoints, and resume endpoint
- [ ] Run and measure a full 500,000-candidate search on production-sized data; verify wall time, memory, disk, cancellation behavior, and complete all-pass/near-miss aggregation
- [x] UI recovery panel and resume-from-checkpoint control for interrupted jobs
