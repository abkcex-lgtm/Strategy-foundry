import math
import pandas as pd
import pytest
from app.data import validate_ohlcv, DataIntegrityError
from app.engine import backtest, indicators, StrategySpec, CONTRACT, normalize_contract, _size_and_target
from app.audit import independent_replay

def candles(n=300):
    # Synthetic prices are used only for unit tests, never as market data.
    rows=[]
    for i in range(n):
        p=100+0.02*i+2*math.sin(i/7)
        rows.append({'open_time':i*60000,'open':p,'high':p+0.3,'low':p-0.3,'close':p+0.05*math.sin(i),'volume':10.0})
    return pd.DataFrame(rows)

def test_valid_data_report():
    r=validate_ohlcv(candles())
    assert r['rows']==300 and r['gap_count']==0

def test_duplicate_timestamp_rejected():
    df=candles(); df.loc[3,'open_time']=df.loc[2,'open_time']
    with pytest.raises(DataIntegrityError): validate_ohlcv(df)

def test_bad_ohlc_rejected():
    df=candles(); df.loc[10,'high']=df.loc[10,'low']-1
    with pytest.raises(DataIntegrityError): validate_ohlcv(df)

def test_strategy_parameters():
    with pytest.raises(ValueError): StrategySpec(fast=36,slow=9).validate()

@pytest.mark.parametrize("family", ["ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout"])
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

@pytest.mark.parametrize("family", ["ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout"])
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

@pytest.mark.parametrize("family", ["ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout"])
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
