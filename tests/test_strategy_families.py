import math
import pandas as pd
from app.engine import STRATEGY_FAMILIES, StrategySpec, indicators
from app.audit import independent_replay
from app.engine import backtest
from app.data import DataIntegrityError


def sample_candles(n=600):
    rows=[]
    for i in range(n):
        close=100+0.01*i+2*math.sin(i/7)+0.7*math.sin(i/2.1)
        op=close+0.2*math.sin(i/3)
        rows.append({"open_time":i*60000,"open":op,"high":max(op,close)+0.5,
                     "low":min(op,close)-0.5,"close":close,"volume":10+(i%11)*2})
    return pd.DataFrame(rows)


def test_every_family_emits_valid_signal_columns():
    df=sample_candles()
    for family in STRATEGY_FAMILIES:
        spec=StrategySpec(fast=5,slow=18,atr_period=7,stop_atr=1.5,
                          family=family,lookback=20,rsi_period=14,
                          rsi_low=30,rsi_high=70,volume_mult=1.5,band_std=2.0)
        spec.validate()
        result=indicators(df,spec)
        assert len(result)==len(df)
        assert result["long_signal"].dtype==bool
        assert result["short_signal"].dtype==bool


def test_independent_audit_supports_new_strategy_families():
    df=sample_candles()
    for family in STRATEGY_FAMILIES:
        spec={"name":family,"fast":5,"slow":18,"atr_period":7,"stop_atr":1.5,
              "direction":"both","seed":42,"family":family,"lookback":20,
              "rsi_period":14,"rsi_low":30,"rsi_high":70,"volume_mult":1.5,
              "band_std":2.0,"momentum_threshold":0.0}
        try:
            primary=backtest(df,spec)
            audit=independent_replay(df,primary["strategy"],primary["contract"])
        except DataIntegrityError as exc:
            if "ambiguity" in str(exc):
                continue
            raise
        assert len(primary["ledger"])==len(audit["ledger"]), "family mismatch: " + family
        for a,b in zip(primary["ledger"],audit["ledger"]):
            assert a["entry_time"]==b["entry_time"]
            assert a["exit_time"]==b["exit_time"]
            assert a["direction"]==b["direction"]
            assert a["exit_reason"]==b["exit_reason"]
            assert math.isclose(a["net_pnl"],b["net_pnl"],rel_tol=1e-10,abs_tol=1e-8)
