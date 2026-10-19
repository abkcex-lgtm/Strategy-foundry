import math
import pandas as pd
import pytest
from app.data import validate_ohlcv, DataIntegrityError
from app.engine import backtest, indicators, StrategySpec, CONTRACT, normalize_contract, _size_and_target, _resolve_bar_exit, prepare_backtest_data, _first_exit_candidate
from app.audit import independent_replay

def candles(n=300):
    # Synthetic prices are used only for unit tests, never as market data.
    rows=[]
    for i in range(n):
        p=100+0.02*i+2*math.sin(i/7)
        rows.append({'open_time':i*60000,'open':p,'high':p+0.3,'low':p-0.3,'close':p+0.05*math.sin(i),'volume':10.0})
    return pd.DataFrame(rows)


def test_prepared_market_data_preserves_full_backtest_output_and_rejects_other_frame():
    df=candles(800)
    spec=StrategySpec(fast=5,slow=18,atr_period=7,stop_atr=1.5)
    cfg=normalize_contract({"reward_risk":2.0})
    prepared=prepare_backtest_data(df)
    standard=backtest(df,spec,cfg)
    optimized=backtest(df,spec,cfg,_prepared_context=prepared)
    assert optimized==standard
    with pytest.raises(ValueError,match="exact source DataFrame"):
        backtest(df.copy(),spec,cfg,_prepared_context=prepared)


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_vectorized_first_exit_candidate_matches_scalar_bar_scan(side):
    import numpy as np
    rng=np.random.default_rng(91)
    for _ in range(100):
        opens=100+rng.normal(0,2,250)
        highs=opens+rng.uniform(0,3,250)
        lows=opens-rng.uniform(0,3,250)
        start=17; stop=98.5; target=102.0
        expected=None
        for j in range(start,len(opens)):
            try:
                resolution=_resolve_bar_exit(side,float(opens[j]),float(highs[j]),float(lows[j]),stop,target)
            except DataIntegrityError as exc:
                if "ambiguity" not in str(exc).lower(): raise
                resolution=("AMBIGUOUS",None)
            if resolution is not None:
                expected=j; break
        assert _first_exit_candidate(side,opens,highs,lows,start,stop,target)==expected


def test_search_validates_shared_market_data_once_for_all_candidates(monkeypatch):
    import app.engine as engine_module
    df=candles(700)
    original=engine_module.validate_ohlcv
    calls={"n":0}
    def counted(*args,**kwargs):
        calls["n"]+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(engine_module,"validate_ohlcv",counted)
    result=engine_module.search_candidates(df,12,seed=17,contract={"use_min_tp_per_day":False})
    assert result["evaluated_candidates"]==12
    assert calls["n"]==1


def test_valid_data_report():
    r=validate_ohlcv(candles())
    assert r['rows']==300 and r['gap_count']==0

def test_backtest_and_independent_replay_normalize_stringified_ohlcv():
    # Reproduce CSV/SQLite round-trips where timestamps and prices arrive as text.
    df=candles(800).astype(str)
    spec=StrategySpec(fast=5,slow=18,atr_period=7,stop_atr=1.5)
    cfg=normalize_contract({"reward_risk":2.0})
    primary=backtest(df,spec,cfg)
    independent=independent_replay(df,primary["strategy"],primary["contract"])
    assert len(primary["ledger"])==len(independent["ledger"])
    numeric_primary=backtest(candles(800),spec,cfg)
    assert primary["metrics"]["data_hash_sha256"]==numeric_primary["metrics"]["data_hash_sha256"]
    for a,b in zip(primary["ledger"],independent["ledger"]):
        assert a["entry_time"]==b["entry_time"]
        assert a["exit_time"]==b["exit_time"]
        assert a["net_pnl"]==pytest.approx(b["net_pnl"],rel=1e-10,abs=1e-8)

def test_duplicate_timestamp_rejected():
    df=candles(); df.loc[3,'open_time']=df.loc[2,'open_time']
    with pytest.raises(DataIntegrityError): validate_ohlcv(df)

def test_bad_ohlc_rejected():
    df=candles(); df.loc[10,'high']=df.loc[10,'low']-1
    with pytest.raises(DataIntegrityError): validate_ohlcv(df)

def test_strategy_parameters():
    with pytest.raises(ValueError): StrategySpec(fast=36,slow=9).validate()

@pytest.mark.parametrize("family", ["ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout", "ema_pullback", "macd_crossover", "bollinger_reversion", "inside_bar_breakout", "engulfing_reversal", "momentum_breakout", "stochastic_reversion", "roc_momentum", "adx_trend", "keltner_breakout", "atr_squeeze_breakout", "obv_trend", "vwap_reclaim", "pivot_reversal", "swing_structure_break", "failed_breakout_reversal", "wick_rejection", "range_expansion_breakout", "three_bar_momentum", "volatility_regime_breakout", "trend_exhaustion_reversal", "volume_climax_reversal", "range_compression_breakout", "support_resistance_rejection", "micro_double_bottom", "micro_double_top"])
def test_supported_strategy_families_produce_causal_signal_columns(family):
    df = candles(500)
    spec = StrategySpec(fast=5, slow=18, atr_period=7, stop_atr=1.5, family=family)
    out = indicators(df, spec)
    assert "long_signal" in out and "short_signal" in out
    assert not out["long_signal"].isna().any()
    assert not out["short_signal"].isna().any()
    assert out["long_signal"].dtype == bool
    assert out["short_signal"].dtype == bool


def test_unknown_strategy_family_is_rejected():
    with pytest.raises(ValueError, match="Unsupported strategy family"):
        StrategySpec(family="future_peeking").validate()

@pytest.mark.parametrize("family", ["ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout", "ema_pullback", "macd_crossover", "bollinger_reversion", "inside_bar_breakout", "engulfing_reversal", "momentum_breakout", "stochastic_reversion", "roc_momentum", "adx_trend", "keltner_breakout", "atr_squeeze_breakout", "obv_trend", "vwap_reclaim", "pivot_reversal", "swing_structure_break", "failed_breakout_reversal", "wick_rejection", "range_expansion_breakout", "three_bar_momentum", "volatility_regime_breakout", "trend_exhaustion_reversal", "volume_climax_reversal", "range_compression_breakout", "support_resistance_rejection", "micro_double_bottom", "micro_double_top"])
def test_future_candle_changes_do_not_rewrite_past_signals(family):
    df = candles(500)
    spec = StrategySpec(fast=5, slow=18, atr_period=7, stop_atr=1.5, family=family)
    original = indicators(df, spec)
    changed = df.copy()
    changed.loc[300:, ["open", "high", "low", "close"]] = changed.loc[300:, ["open", "high", "low", "close"]] * 1.7
    changed.loc[300:, "volume"] = changed.loc[300:, "volume"] * 4.0
    revised = indicators(changed, spec)
    for column in ("long_signal", "short_signal"):
        assert original.loc[:299, column].tolist() == revised.loc[:299, column].tolist()

def test_backtest_is_deterministic():
    df=candles(); spec=StrategySpec(fast=5,slow=18,atr_period=7,stop_atr=1.5)
    try:
        a=backtest(df,spec); b=backtest(df,spec)
    except DataIntegrityError as exc:
        if 'ambiguity' in str(exc): pytest.skip('Synthetic path intentionally triggered conservative ambiguity guard')
        raise
    assert a['run_hash_sha256']==b['run_hash_sha256']
    assert a['ledger']==b['ledger']
    assert a['metrics']['data_hash_sha256']==b['metrics']['data_hash_sha256']

def test_contract_defaults():
    assert CONTRACT['risk_usd']==10
    assert CONTRACT['reward_risk']==3.5
    assert CONTRACT['max_positions']==1
    assert CONTRACT['commission_pct_per_side']+CONTRACT['slippage_pct_per_side']==0.06

def test_custom_contract_settings_are_respected():
    cfg=normalize_contract({'initial_balance':2500,'risk_mode':'fixed_usd','risk_usd':25,'reward_risk':2.0,'commission_pct_per_side':0.1,'slippage_pct_per_side':0.05,'min_win_rate_pct':45,'max_consecutive_losses':8})
    assert cfg['initial_balance']==2500
    assert cfg['risk_usd']==25
    assert cfg['reward_risk']==2.0
    assert abs(cfg['commission_pct_per_side']+cfg['slippage_pct_per_side']-0.15)<1e-12
    assert cfg['min_win_rate_pct']==45
    assert cfg['max_consecutive_losses']==8

def test_net_rr_prices_include_costs():
    cfg=normalize_contract({'risk_usd':10,'reward_risk':3.5,'commission_pct_per_side':0.04,'slippage_pct_per_side':0.02,'target_mode':'net_rr'})
    entry=2000.0; stop=1980.0
    qty,target,entry_rate,exit_rate=_size_and_target(entry,stop,'LONG',10,cfg)
    loss=qty*((entry-stop)+entry*entry_rate+stop*exit_rate)
    net_win=qty*((target-entry)-entry*entry_rate-target*exit_rate)
    assert abs(loss-10)<1e-9
    assert abs(net_win-35)<1e-8

def test_unsupported_addon_fails_closed():
    with pytest.raises(ValueError,match='not implemented'):
        normalize_contract({'trailing_stop_enabled':True})

@pytest.mark.parametrize("family", ["ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout", "ema_pullback", "macd_crossover", "bollinger_reversion", "inside_bar_breakout", "engulfing_reversal", "momentum_breakout", "stochastic_reversion", "roc_momentum", "adx_trend", "keltner_breakout", "atr_squeeze_breakout", "obv_trend", "vwap_reclaim", "pivot_reversal"])
def test_independent_engine_matches_primary_ledger(family):
    df=candles(800)
    spec=StrategySpec(fast=5,slow=18,atr_period=7,stop_atr=1.5,family=family)
    cfg=normalize_contract({'risk_usd':10,'reward_risk':2.0})
    try:
        primary=backtest(df,spec,cfg)
        independent=independent_replay(df,primary['strategy'],primary['contract'])
    except DataIntegrityError as exc:
        if 'ambiguity' in str(exc): pytest.skip('Synthetic path intentionally triggered conservative ambiguity guard')
        raise
    assert len(primary['ledger'])==len(independent['ledger'])
    for a,b in zip(primary['ledger'],independent['ledger']):
        for key in ('trade_id','signal_time','entry_time','exit_time','direction','exit_reason','loss_streak_after'):
            assert a[key]==b[key]
        for key in ('entry_price','stop_loss','take_profit','exit_price','net_pnl','balance_after'):
            assert a[key]==pytest.approx(b[key],rel=1e-10,abs=1e-8)


def test_hybrid_genetic_search_is_seeded_and_reports_real_evaluations():
    from app.engine import search_candidates
    df=candles(700)
    a=search_candidates(df,25,seed=123,contract={"use_min_tp_per_day":False},algorithm="hybrid_genetic")
    b=search_candidates(df,25,seed=123,contract={"use_min_tp_per_day":False},algorithm="hybrid_genetic")
    assert a["algorithm"]=="hybrid_genetic"
    assert a["seed"]==123
    assert a["evaluated_candidates"]==b["evaluated_candidates"]
    assert [r["run_hash_sha256"] for r in a["results"]]==[r["run_hash_sha256"] for r in b["results"]]
    assert sum(a["family_evaluated_counts"].values())==a["evaluated_candidates"]
    assert a["evaluated_candidates"]<=a["requested_candidates"]
    assert "not production-certified" in a["warning"]


@pytest.mark.parametrize("family", ["swing_structure_break", "failed_breakout_reversal", "wick_rejection", "range_expansion_breakout", "three_bar_momentum", "volatility_regime_breakout", "trend_exhaustion_reversal", "volume_climax_reversal", "range_compression_breakout", "support_resistance_rejection", "micro_double_bottom", "micro_double_top"])
def test_new_price_action_families_match_independent_ledger(family):
    df = candles(900)
    spec = StrategySpec(fast=5, slow=18, atr_period=7, stop_atr=1.5, family=family)
    try:
        primary = backtest(df, spec, normalize_contract({"reward_risk": 2.0}))
        independent = independent_replay(df, primary["strategy"], primary["contract"])
    except DataIntegrityError as exc:
        if "ambiguity" in str(exc).lower():
            pytest.skip("Synthetic path triggered conservative ambiguity guard")
        raise
    assert len(primary["ledger"]) == len(independent["ledger"])
    for a, b in zip(primary["ledger"], independent["ledger"]):
        assert a["entry_time"] == b["entry_time"]
        assert a["exit_time"] == b["exit_time"]
        assert a["direction"] == b["direction"]
        assert a["net_pnl"] == pytest.approx(b["net_pnl"], rel=1e-10, abs=1e-8)


def test_search_rejects_unknown_algorithm():
    from app.engine import search_candidates
    with pytest.raises(ValueError,match="Unsupported search algorithm"):
        search_candidates(candles(300),10,algorithm="magic")


@pytest.mark.parametrize("context_filter", ["none", "ema_trend", "adx_strength", "volume_confirm", "vwap_side", "low_volatility"])
def test_compositional_context_filters_are_causal_and_independently_replayed(context_filter):
    df = candles(800)
    spec = StrategySpec(fast=5, slow=18, atr_period=7, stop_atr=1.5,
                        family="momentum_breakout", context_filter=context_filter,
                        confirmation_filter="rsi_midline")
    signals = indicators(df, spec)
    changed = df.copy()
    changed.loc[500:, ["open", "high", "low", "close"]] *= 1.4
    changed.loc[500:, "volume"] *= 3
    revised = indicators(changed, spec)
    assert signals.loc[:499, "long_signal"].tolist() == revised.loc[:499, "long_signal"].tolist()
    assert signals.loc[:499, "short_signal"].tolist() == revised.loc[:499, "short_signal"].tolist()
    try:
        primary = backtest(df, spec, normalize_contract({"reward_risk": 2.0}))
        independent = independent_replay(df, primary["strategy"], primary["contract"])
    except DataIntegrityError as exc:
        if "ambiguity" in str(exc):
            pytest.skip("Synthetic path triggered conservative ambiguity guard")
        raise
    assert len(primary["ledger"]) == len(independent["ledger"])
    for a, b in zip(primary["ledger"], independent["ledger"]):
        assert a["entry_time"] == b["entry_time"]
        assert a["exit_time"] == b["exit_time"]
        assert a["direction"] == b["direction"]
        assert a["net_pnl"] == pytest.approx(b["net_pnl"], rel=1e-10, abs=1e-8)


def test_unknown_composition_filter_is_rejected():
    with pytest.raises(ValueError, match="Unsupported context filter"):
        StrategySpec(context_filter="future_leak").validate()


def test_surrogate_genetic_search_fits_ml_model(monkeypatch):
    import hashlib
    import app.engine as engine
    import app.audit as audit
    def fake_backtest(frame, spec, contract):
        data_hash = engine.sha256_bytes(frame.reset_index(drop=True).to_csv(index=False, float_format="%.12g").encode())
        metrics = {"trades": 12, "wins": 3, "losses": 9, "win_rate_pct": 25.0,
                   "profit_factor": 0.6, "avg_net_tp_per_day": 0.2,
                   "max_consecutive_losses": 7, "net_pnl": -30.0,
                   "final_balance": 970.0, "total_costs": 1.0,
                   "data_hash_sha256": data_hash, "rows": 300, "span_days": 1.0,
                   "passes_all_targets": False}
        return {"strategy": engine.asdict(spec), "contract": contract, "metrics": metrics,
                "run_hash_sha256": hashlib.sha256(repr(spec).encode()).hexdigest(), "ledger": []}
    monkeypatch.setattr(engine, "backtest", fake_backtest)
    monkeypatch.setattr(audit, "independent_replay", lambda frame, spec, cfg: {"ledger": []})
    result = engine.search_candidates(pd.DataFrame(), max_candidates=80, seed=19,
                                      algorithm="surrogate_genetic")
    assert result["algorithm"] == "surrogate_genetic"
    assert result["surrogate_model_fits"] >= 1
    assert result["evaluated_candidates"] == 80
    assert result["preflight_summary"]["evaluated_candidates"] == 80
    assert result["preflight_summary"]["promising_candidates"] == 0
    assert "ExtraTrees" in result["warning"]

    second = engine.search_candidates(
        pd.DataFrame(), max_candidates=30, seed=20, algorithm="surrogate_genetic",
        excluded_genome_hashes=set(result["evaluated_genome_hashes"]),
        prior_candidates=result["results"])
    assert second["prior_candidates_used"] == len(result["results"])
    assert second["evaluated_candidates"] == 30


@pytest.mark.parametrize("operation", ["and", "or", "xor"])
def test_composite_family_signals_match_independent_replay(operation):
    df = candles(900)
    spec = StrategySpec(fast=5, slow=18, atr_period=7, stop_atr=1.5,
                        family="wick_rejection", secondary_family="channel_breakout",
                        signal_composition=operation)
    signals = indicators(df, spec)
    assert signals["long_signal"].dtype == bool
    assert signals["short_signal"].dtype == bool
    try:
        primary = backtest(df, spec, normalize_contract({"reward_risk": 2.0}))
        independent = independent_replay(df, primary["strategy"], primary["contract"])
    except DataIntegrityError as exc:
        if "ambiguity" in str(exc).lower():
            pytest.skip("Synthetic path triggered conservative ambiguity guard")
        raise
    assert len(primary["ledger"]) == len(independent["ledger"])
    for a, b in zip(primary["ledger"], independent["ledger"]):
        assert a["entry_time"] == b["entry_time"]
        assert a["exit_time"] == b["exit_time"]
        assert a["direction"] == b["direction"]
        assert a["net_pnl"] == pytest.approx(b["net_pnl"], rel=1e-10, abs=1e-8)


def test_global_genome_fingerprints_exclude_candidates_across_batches():
    from app.engine import search_candidates
    df = candles(450)
    first = search_candidates(df, 12, seed=101, algorithm="stratified_random",
                              contract={"use_min_tp_per_day": False})
    first_hashes = set(first["evaluated_genome_hashes"])
    assert len(first_hashes) == first["evaluated_candidates"]
    second = search_candidates(df, 12, seed=202, algorithm="stratified_random",
                               contract={"use_min_tp_per_day": False},
                               excluded_genome_hashes=first_hashes)
    assert first_hashes.isdisjoint(set(second["evaluated_genome_hashes"]))
    assert len(set(second["evaluated_genome_hashes"])) == second["evaluated_candidates"]


def test_non_reference_dataset_gets_chronological_train_dev_oos_splits():
    from app.data import split_reference_period
    df = candles(100)
    splits = split_reference_period(df)
    assert len(splits["train"]) == 60
    assert len(splits["development"]) == 20
    assert len(splits["oos"]) == 20
    assert splits["train"].open_time.max() < splits["development"].open_time.min()
    assert splits["development"].open_time.max() < splits["oos"].open_time.min()


@pytest.mark.parametrize("side,bar_open,bar_high,bar_low,stop,target,expected", [
    ("LONG", 95.0, 105.0, 94.0, 98.0, 104.0, ("SL", 95.0)),
    ("SHORT", 105.0, 106.0, 95.0, 102.0, 96.0, ("SL", 105.0)),
    ("LONG", 106.0, 108.0, 99.0, 98.0, 104.0, ("TP", 104.0)),
    ("SHORT", 94.0, 101.0, 92.0, 102.0, 96.0, ("TP", 96.0)),
    ("LONG", 100.0, 103.0, 97.0, 98.0, 104.0, ("SL", 98.0)),
    ("SHORT", 100.0, 103.0, 97.0, 102.0, 96.0, ("SL", 102.0)),
])
def test_exit_fill_resolves_gaps_before_intrabar_ambiguity(side, bar_open, bar_high, bar_low, stop, target, expected):
    assert _resolve_bar_exit(side, bar_open, bar_high, bar_low, stop, target) == expected


def test_exit_fill_rejects_unresolved_intrabar_sl_tp_ambiguity():
    with pytest.raises(DataIntegrityError, match="same-candle SL/TP ambiguity"):
        _resolve_bar_exit("LONG", 100.0, 105.0, 97.0, 98.0, 104.0)


def test_search_returns_exactly_five_audited_near_misses_when_no_candidate_passes():
    from app.engine import search_candidates
    result = search_candidates(
        candles(700),
        25,
        seed=20261017,
        algorithm="stratified_random",
        contract={
            "reward_risk": 2.0,
            "min_win_rate_pct": 99.9,
            "min_profit_factor": 1000.0,
            "min_tp_per_day": 1000.0,
            "max_consecutive_losses": 0,
        },
    )
    assert result["all_passing"] is False
    assert result["returned_near_misses"] == 5
    assert len(result["results"]) == 5
    assert result["near_miss_shortfall"] == 0
    assert all(item.get("independent_audit", {}).get("status") == "MATCH" for item in result["results"])


def test_fixed_risk_backtest_stops_before_equity_falls_below_risk_budget(monkeypatch):
    import app.engine as engine

    df = candles(300)
    spec = StrategySpec(fast=2, slow=3, atr_period=2, stop_atr=1.0)

    def always_long(frame, _spec):
        x = frame.copy()
        x["atr"] = 1.0
        x["long_signal"] = True
        x["short_signal"] = False
        return x

    monkeypatch.setattr(engine, "indicators", always_long)
    monkeypatch.setattr(engine, "_resolve_bar_exit",
                        lambda side, bar_open, bar_high, bar_low, stop, target: ("SL", stop))
    cfg = normalize_contract({
        "initial_balance": 1000.0, "risk_mode": "fixed_usd", "risk_usd": 10.0,
        "commission_pct_per_side": 0.0, "slippage_pct_per_side": 0.0,
    })
    result = backtest(df, spec, cfg)

    assert len(result["ledger"]) == 99
    assert result["metrics"]["final_balance"] == pytest.approx(10.0)
    assert result["metrics"]["net_pnl"] == pytest.approx(-990.0)
    assert result["metrics"]["final_balance"] >= 0.0


def test_independent_replay_stops_before_equity_falls_below_risk_budget(monkeypatch):
    import app.audit as audit

    df = candles(300)
    spec = StrategySpec(fast=2, slow=3, atr_period=2, stop_atr=1.0)
    monkeypatch.setattr(
        audit, "_independent_signals",
        lambda d, s: (
            pd.Series(True, index=d.index), pd.Series(False, index=d.index),
            pd.Series(1.0, index=d.index), int(s["fast"]), int(s["slow"]), int(s["atr_period"])
        ),
    )
    monkeypatch.setattr(
        audit, "_resolve_bar_exit_independent",
        lambda side, bar_open, bar_high, bar_low, stop, target: ("SL", stop),
    )
    cfg = normalize_contract({
        "initial_balance": 1000.0, "risk_mode": "fixed_usd", "risk_usd": 10.0,
        "commission_pct_per_side": 0.0, "slippage_pct_per_side": 0.0,
    })
    result = independent_replay(df, spec.__dict__, cfg)

    assert len(result["ledger"]) == 99
    assert result["ledger"][-1]["balance_after"] == pytest.approx(10.0)
    assert result["ledger"][-1]["balance_after"] >= 0.0
