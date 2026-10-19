# Strategy Foundry — Strategy-Space and Research Engine Contract

Status: **target architecture / acceptance contract**, not a claim that every item below is already implemented. The current engine only implements the family IDs explicitly listed in `app/engine.py`; any family shown here must not be exposed as runnable until its signal logic, causal tests, independent replay, and UI wiring are implemented.

## Equal-weight objective and evidence eligibility (release 0.9.9)

For the four user-selectable performance criteria—win rate, profit factor, average net full TPs/day, and maximum consecutive losses—calculate each enabled criterion's normalized distance to its threshold, then use the arithmetic mean. Each active criterion therefore contributes exactly 1/N to the score. Disabled criteria contribute neither distance nor pass/fail requirements. Do not add hidden bonuses or tie-break preferences for any one of the four metrics in the genetic fitness, surrogate target, API ordering, or research-runner ordering.

Separate eligibility gates do not change the four equal weights: input/data integrity, no open position at end, independent trade-by-trade ledger parity, and sufficient evidence. Evidence requires at least 100 closed trades. If TP/day is active, trade count must also be compatible with the requested TP/day rate and observed win rate; if TP/day is disabled, that target must not be used to steer or block selection beyond the minimum sample floor. Insufficient evidence means the candidate is not certified even if its point estimates happen to cross the four thresholds.

Required lifecycle: generate diverse causal candidates on Train; rank using equal-weight active-target distance and evidence eligibility; freeze the candidate pool; audit and compare on Development; evaluate untouched OOS only after selection freezes. A green CI or successful deployment is not proof of a profitable strategy, and no real-data outcome may be reported unless the actual search and its independent audit complete.

## Product principle

The factory searches a broad, compositional hypothesis space. It must not equate "500,000 strategies" with 500,000 random parameter variations of a few indicators. A strategy identity includes its entry logic, optional context/regime filters, direction, exit/risk model, timeframe/data contract, and all parameters. Candidates that are syntactic variations but produce identical signals/trades should be deduplicated and counted as equivalent, not advertised as meaningful diversity.

No optimizer may promise a profitable strategy for every set of constraints. If no candidate survives, report the measured failure and nearest valid candidates without relaxing user constraints or peeking at untouched OOS data.

## Strategy-family catalogue

The categories below define the intended search grammar. They are deliberately broader than the currently implemented ten families. Each family must have a human-readable Persian description, explicit parameter ranges, valid combinations, warm-up requirements, causal/no-lookahead tests, and a strategy-specific unit test before it can be enabled.

### 1. Price action and candle structure
- Single-candle structure: pin bars / rejection wicks, marubozu, doji variants, long-body impulse candles.
- Multi-candle formations: engulfing, inside/outside bars, harami, piercing/dark-cloud patterns, morning/evening star, three-bar reversals, three soldiers/crows.
- Break-and-retest; failed breakout / spring / upthrust; compression-to-expansion; range rejection; consecutive closes and displacement candles.
- Swing structure: higher-high/higher-low, lower-high/lower-low, swing failure, structure break (BOS), character change (CHoCH), protected swing retest.
- Pattern thresholds must be normalized by ATR or candle range where appropriate; avoid brittle fixed-dollar thresholds.

### 2. Trend following and trend structure
- Moving-average crossovers and alignment: EMA, SMA, WMA, HMA, KAMA, TEMA, VWMA.
- Slope/angle and persistence; trend-strength confirmation (ADX/DMI); price above/below adaptive trend baselines.
- Pullback continuation, moving-average reclaim/loss, trendline/channel break, breakout continuation, Donchian/Turtle-style systems.
- Parabolic SAR and Supertrend-style trailing regime signals (only if compatible with the selected exit model).
- Multi-timeframe trend alignment, with strictly closed higher-timeframe bars and no forward fill from future values.

### 3. Mean reversion and statistical stretch
- RSI/Stochastic/Williams %R extremes and recross; CCI and rate-of-change extremes.
- Bollinger/Keltner envelope rejection, z-score distance from rolling mean, deviation from VWAP/anchored VWAP where data supports it.
- Overshoot and re-entry, failed continuation, range-edge fade, volatility-normalized displacement.
- Optional trend/regime vetoes to avoid fading strong directional markets; test the veto independently and in combination.

### 4. Breakout, range, volatility compression and expansion
- Donchian/high-low channel breakouts; opening/session/range breakouts where session data is meaningful.
- Volatility contraction followed by expansion; Bollinger bandwidth, ATR percentile, squeeze release.
- Consolidation box, inside-bar clusters, narrow-range bars, pivot/range breaks, breakout-retest.
- False-break filters, close-confirmation, volume confirmation, distance/ATR constraints.

### 5. Momentum, impulse and rate-of-change
- ROC/momentum thresholds, MACD family, PPO, moving-average convergence/divergence, acceleration and deceleration.
- Candle/body displacement, consecutive directional closes, impulse-follow-through, momentum divergence where causally defined.
- Relative momentum across instruments only when synchronized data is available and its alignment is validated.
- Avoid using the candle's final close to enter at that same close when the contract requires next-candle-open execution.

### 6. Volume, order-flow proxies and participation
- Relative volume, volume z-score, volume spikes, volume-confirmed breakouts, volume dry-up before expansion.
- OBV, Chaikin money flow/accumulation-distribution, volume-price trend, VWMA/VWAP interactions.
- Candle spread/range versus volume, effort-versus-result proxies, climax/rejection patterns.
- True bid/ask delta, footprint, liquidation and order-book imbalance are **not** derivable from OHLCV alone; require the corresponding raw data source and integrity checks. Never pretend an OHLCV proxy is real order flow.

### 7. Support/resistance, pivots and liquidity-style price levels
- Causal swing highs/lows, rolling extrema, pivot levels, range boundaries, round-number proximity.
- Rejection, breakout, retest, level clustering and confluence with volatility-normalized tolerance.
- Liquidity-sweep-like patterns may be OHLCV proxies only; label them as proxies, not observed resting liquidity.
- A level is usable only after it has become knowable; pivot algorithms must delay confirmation until the required right-side bars have closed.

### 8. Channels, bands, trendlines and geometry
- Donchian, regression, standard-deviation, ATR and adaptive channels.
- Trendline touch/break/retest; parallel channel bounce/break; slope and width constraints.
- Rolling linear regression slope/residual, deviation bands, channel compression/expansion.
- Dynamic lines and pivots must be computed causally; no drawing a line using future pivots and then trading it in the past.

### 9. Market regime and state classification
- Trend/range/transition, high/low volatility, high/low directional strength, volume state, volatility clustering.
- Rule-based regime classifiers first; optional HMM or other statistical state models only with walk-forward fitting and training-only normalization.
- Regime-aware strategy selection or gating; measure whether a regime gate adds value out of sample rather than assuming it does.
- Unknown/low-confidence regime should have an explicit policy, not a silent default.

### 10. Multi-timeframe and session context
- Closed-bar trend/context filters across selected timeframes; higher-timeframe swing and range context.
- Session/day-of-week/time-of-day effects only when timestamps and market-session assumptions are valid.
- Funding, basis, open interest, liquidation and macro-event filters only when matching historical data is actually available; otherwise disable them and explain why.
- Resampling must preserve timezone, boundaries and closed-bar availability; no leakage from an unfinished higher-timeframe candle.

### 11. Divergence and inter-series relationships
- Regular/hidden divergence for RSI, MACD, ROC and price swings, using delayed/confirmed pivots.
- Correlation/beta/spread or relative-strength conditions for synchronized related assets.
- Cointegration/pairs/statistical-arbitrage methods are a distinct research mode requiring multi-symbol datasets, synchronized calendars, and separate portfolio accounting; do not fake them from a single ETHUSDT series.

### 12. Volume-profile and VWAP-derived context
- Session/rolling VWAP, anchored VWAP after a causally identified event, VWAP reclaim/rejection.
- Volume-profile value area / point-of-control only when source granularity and the profile construction are available and tested.
- If only 1-minute OHLCV exists, profile estimates must be labeled approximate; do not claim exchange-trade-level precision.

### 13. Classical statistical and quantitative strategies
- Rolling z-score, robust z-score, percentile/rank thresholds, rolling regression, residual mean reversion, volatility targeting.
- Breakout/momentum after statistical compression; time-series trend filters; simple autoregressive forecasts as candidate features.
- Fit parameters only on training windows; freeze them for validation windows. Any model fitting belongs inside each walk-forward fold.

### 14. Machine-learning feature strategies
- Supervised classification/regression for direction, return or barrier outcomes using causal OHLCV-derived features.
- Tree ensembles and regularized linear models as baseline model families; neural sequence models only when data volume, compute, calibration and validation justify them.
- Feature selection, scaling, class balancing and hyperparameter tuning must happen within training folds only.
- Probability calibration, threshold selection, feature-importance diagnostics, model/version hashes and reproducible seeds are required.
- ML predictions are candidate signals, not evidence of edge; compare against simple baselines and include all tuning in the multiple-testing/overfit assessment.

### 15. Evolutionary, genetic and program-synthesis strategies
- Genetic algorithm / genetic programming over typed signal expressions, operators, indicators, comparisons, lookbacks, direction, gates and exits.
- Mutation, crossover, elitism, tournament/novelty selection, archive of diverse Pareto candidates, duplicate-signal elimination and lineage tracking.
- Enforce grammar/type constraints, maximum expression depth, warm-up bounds and compute budgets. Reject invalid or future-looking expressions before backtest.
- Fitness must be multi-objective (e.g. integrity, out-of-sample robustness, drawdown/streak, PF/expectancy, trade sufficiency, diversity) rather than optimizing win rate alone.
- Periodically re-evaluate elites on independent validation; never use the final untouched OOS period to guide evolution or candidate ranking.

### 16. Hybrid and ensemble strategies
- Combine independently defined entry triggers with optional context gates and confirmations.
- AND/OR/sequence/state-machine composition with explicit order and time semantics; avoid uncontrolled combinatorial explosion.
- Ensembles may combine signals or model scores only with documented weighting and training-only fitting.
- Include ablation tests to show whether each component adds value; avoid ensembles that only hide a weak component.

### 17. Execution-aware and risk-aware strategy variants
- Next-candle-open entry, stop/target placement, volatility-normalized stops, fixed-R targets, structural stops, and user-configured direction/entry policies where implemented.
- Fees, spread/slippage assumptions, funding and gap behavior must be explicit. If a cost source is unavailable, label it an assumption.
- Intrabar SL/TP ambiguity must be resolved using finer-grained historical data or fail closed; never choose the favorable outcome by default.
- Trade management features (break-even, trailing, partial exit, time exit, cooldown, max trades/day) remain disabled unless their exact semantics are implemented and independently replayed.

## Compositional candidate grammar

A candidate is not merely an indicator-period tuple. It may consist of:
1. **Entry trigger** (one family or a bounded combination).
2. **Direction** (long, short, both; separate side-specific logic is permitted).
3. **Context/regime gate** (optional, bounded count).
4. **Confirmation** (optional, causal and explicit).
5. **Stop/target model** (selected from implemented execution-safe models).
6. **Trade/risk settings** (from the user's active contract).
7. **Data/timeframe assumptions** (only what is actually available).
8. **Unique normalized genome**, source/config/data hashes, parent lineage and search seed.

The engine should use a deterministic stratified baseline search, followed by evolutionary/program-synthesis search and optional ML-guided proposal. Maintain family quotas and novelty pressure so one easy-to-score family cannot monopolize the population. Search count means candidates actually evaluated, not the requested count; show attempted, valid, unique, deduplicated, failed-preflight, and completed counts separately.

## 500,000-candidate operating requirements

- Do not raise the UI limit to 500,000 until the server has bounded batch processing, resumable jobs, cancellation, progress checkpoints, memory limits, persistent candidate summaries, deduplication and test coverage.
- Precompute causal feature columns once per dataset/config where safe; cache only by exact data and feature-config hash.
- Batch/vectorize candidate evaluation where possible; isolate expensive ML fits and avoid refitting the same feature model unnecessarily.
- Persist job status and checkpoints so refresh/restart does not erase completed work; expose failures and allow resume from a deterministic seed/checkpoint.
- Record requested/evaluated/valid/unique counts, runtime, hardware/runtime version, data hash, code version, config hash and seed in the manifest.
- Apply minimum trade count and diversity checks before ranking; no ranking metric may override hard user filters.
- Search/train on the training window; use development data for controlled model selection; final OOS remains untouched until the search and selection procedure is frozen.

## Settings UX: complete capability, comprehensible grouping

Use progressive disclosure: simple defaults and a one-screen summary first, expert controls available under clearly named expandable groups. Never hide active constraints. Every non-obvious setting gets a Persian explanation, allowed range, default, effect on compute, and a warning if it changes comparability.

1. **Market and data** — symbol, exchange/market type, timeframe, date range, data source, missing candles, timezone, data quality, finer-granularity ambiguity resolution.
2. **Search scope** — family groups, include/exclude families, long/short, parameter bounds, composition depth, maximum gates, minimum family diversity, candidate budget, seed, time/memory budget, resume/cancel.
3. **Search intelligence** — stratified search, GA/GP, mutation/crossover, novelty/diversity, Pareto objectives, ML proposal, walk-forward folds; each option disabled until implemented and tested.
4. **Entry and confirmation** — next-candle-open contract, trigger, filters, multi-timeframe confirmation, warm-up and bar-close rules.
5. **Risk and exits** — starting balance, fixed/percentage risk, compounding, stop/target model, R:R basis, max positions, fee/slippage/funding assumptions, management add-ons with precise semantics.
6. **Acceptance criteria** — win rate, PF, net TP/day, max consecutive losses, max drawdown, minimum trade count, stability across folds, OOS policy, and whether each is a hard filter or ranking objective.
7. **Validation and evidence** — integrity gates, leakage tests, independent engine, ambiguity policy, reproducibility, standalone replay/export, hashes and audit level.
8. **Chart and review** — full trade ledger, outcome filters, chart overlay mode, focus selected trade, previous/next trade, replay speed and time range.
9. **AI assistant and integrations** — provider/model and credentials only when AI features are ready; never block core search/backtest on missing AI credentials. Secret values must never be echoed.
10. **Storage and compute** — persistence state, job concurrency, resource budget, history retention, export, and resume status.

## One-glance final test panel

Always show the exact dataset and hash; train/dev/OOS date ranges; active constraints; risk/fee/slippage/entry/exit model; management switches; requested vs evaluated vs unique candidates; elapsed time; number passing all hard filters; and validation status.

Each criterion must be visibly PASS / FAIL / NOT TESTED with actual value, target, distance and data window. Overall state is:
- **PASS — independently validated** only if all hard conditions pass and independent replay matches.
- **NEAR-MISS** only when no candidate passes, showing exactly five valid closest candidates when five can be evaluated; disclose if fewer than five valid candidates exist.
- **PREFLIGHT FAILED / SEARCH NOT STARTED** for failed data, leakage, execution, dependency, or ambiguity gates.
- **INCOMPLETE** when a required validation has not run. Never label this as pass.

Training, development and untouched OOS metrics must be separate. OOS must not feed search, feature selection, ranking, or parameter tuning.

## Full trade chart acceptance criteria

- The complete canonical ledger is the source of truth; the chart must not fabricate or omit trades because of current viewport.
- Every trade is drawn as a position box equivalent to a Long Position / Short Position tool: entry-to-exit span, direction, entry, stop, target, exit, and result. Actual executed trades must be visually distinct from manually drawn planning overlays.
- Filters: all trades, TP winners, SL losers, long, short, and other exit reasons where applicable. Filtering changes only visibility, never the ledger.
- Selecting any ledger row focuses the chart on that trade, selects/highlights its position box and opens the full trade detail. Previous/next trade navigation respects the active filter.
- Pan/zoom across the entire test period, jump to date, and replay chronologically. Replay must not reveal future candles, exits, or trade annotations before their event time.
- Dense periods need aggregation/decluttering at wide zoom, while zoom-in reveals every individual setup; show visible/total trade counts so users know what is on screen.
- The chart and ledger must agree trade-for-trade on entry/exit timestamps, prices, direction, stop/target, reason, costs and net P&L. Clicking a chart trade selects the corresponding ledger row and vice versa.

## Release gate

A catalogue entry is not "implemented" because it has a label or appears in settings. It is enabled only after: signal implementation; parameter validation; no-lookahead/warm-up tests; synthetic edge-case tests; Engine A vs independent Engine B ledger reconciliation; chart/ledger integration; documentation and Persian help text. CI must report the enabled families and the count of passing tests.