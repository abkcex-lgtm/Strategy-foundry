import pandas as pd
import pytest

from app.engine import _resolve_execution_interval
from app.data import DataIntegrityError
from app.audit import _resolve_execution_interval_independent


def one_second_path():
    rows=[]
    start=1_800_000_000_000
    for i in range(60):
        price=100.0
        high=101.0
        low=99.0
        if i==5:
            low=97.0
        if i==10:
            high=105.0
        rows.append({"open_time":start+i*1000,"open":price,"high":high,"low":low,"close":price,"volume":1.0})
    return pd.DataFrame(rows),start


def test_one_second_path_resolves_fast_stop_before_later_target():
    frame,start=one_second_path()
    assert _resolve_execution_interval("LONG",frame,start,60_000,98.0,104.0)==("SL",98.0,start+5_000)
    assert _resolve_execution_interval_independent("LONG",frame,start,60_000,98.0,104.0)==("SL",98.0,start+5_000)


def test_one_second_path_resolves_fast_target_before_later_stop():
    frame,start=one_second_path()
    frame.loc[5,"low"]=99.0
    frame.loc[5,"high"]=105.0
    frame.loc[10,"high"]=101.0
    frame.loc[10,"low"]=97.0
    assert _resolve_execution_interval("LONG",frame,start,60_000,98.0,104.0)==("TP",104.0,start+5_000)
    assert _resolve_execution_interval_independent("LONG",frame,start,60_000,98.0,104.0)==("TP",104.0,start+5_000)


def test_missing_one_second_candle_fails_closed():
    frame,start=one_second_path()
    frame=frame.drop(index=7).reset_index(drop=True)
    with pytest.raises(DataIntegrityError,match="Missing/incomplete 1-second"):
        _resolve_execution_interval("LONG",frame,start,60_000,98.0,104.0)
    with pytest.raises(ValueError,match="Missing/incomplete 1-second"):
        _resolve_execution_interval_independent("LONG",frame,start,60_000,98.0,104.0)


def test_same_second_stop_and_target_still_fails_closed():
    frame,start=one_second_path()
    frame.loc[5,"low"]=97.0
    frame.loc[5,"high"]=105.0
    with pytest.raises(DataIntegrityError,match="same-candle SL/TP ambiguity"):
        _resolve_execution_interval("LONG",frame,start,60_000,98.0,104.0)
    with pytest.raises(ValueError,match="UNRESOLVED_AMBIGUITY"):
        _resolve_execution_interval_independent("LONG",frame,start,60_000,98.0,104.0)


def test_primary_backtest_fails_closed_on_unresolved_execution(monkeypatch):
    import pandas as pd
    from app import engine
    from app.engine import StrategySpec, backtest
    from app.data import DataIntegrityError

    start = 1_800_000_000_000
    base = pd.DataFrame({
        "open_time": [start + i * 60_000 for i in range(8)],
        "open": [100.0] * 8, "high": [101.0] * 8,
        "low": [99.0] * 8, "close": [100.0] * 8, "volume": [1.0] * 8,
    })

    def signals(df, spec):
        out = df.copy()
        out["atr"] = 1.0
        out["long_signal"] = False
        out["short_signal"] = False
        out.loc[3, "long_signal"] = True
        return out

    monkeypatch.setattr(engine, "indicators", signals)
    monkeypatch.setattr(engine, "_resolve_execution_interval",
                        lambda *args, **kwargs: (_ for _ in ()).throw(DataIntegrityError("same-candle SL/TP ambiguity")))

    with pytest.raises(DataIntegrityError, match="Execution preflight failed"):
        backtest(base, StrategySpec(fast=2, slow=3, atr_period=2, stop_atr=1), execution_frame=one_second_path()[0])


def test_independent_audit_fails_closed_on_unresolved_execution(monkeypatch):
    from app import audit
    start = 1_800_000_000_000
    base = pd.DataFrame({
        "open_time": [start + i * 60_000 for i in range(8)],
        "open": [100.0] * 8, "high": [101.0] * 8,
        "low": [99.0] * 8, "close": [100.0] * 8, "volume": [1.0] * 8,
    })

    def signals(df, spec):
        long = pd.Series(False, index=df.index); long.iloc[3] = True
        short = pd.Series(False, index=df.index)
        atr = pd.Series(1.0, index=df.index)
        return long, short, atr, 2, 3, 2

    monkeypatch.setattr(audit, "_independent_signals", signals)
    monkeypatch.setattr(audit, "_resolve_execution_interval_independent",
                        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("UNRESOLVED_AMBIGUITY")))
    spec = {"fast": 2, "slow": 3, "atr_period": 2, "stop_atr": 1.0, "direction": "both",
            "family": "ema_crossover", "context_filter": "none", "confirmation_filter": "none",
            "secondary_family": "none", "signal_composition": "single"}
    cfg = {"initial_balance": 1000.0, "risk_mode": "fixed_usd", "risk_usd": 10.0,
           "risk_pct": 1.0, "compound": False, "reward_risk": 3.5, "target_mode": "net_rr",
           "commission_pct_per_side": 0.04, "slippage_pct_per_side": 0.02}
    with pytest.raises(ValueError, match="EXECUTION_PREFLIGHT_FAILED"):
        audit.independent_replay(base, spec, cfg, execution_frame=one_second_path()[0])



def test_primary_backtest_automatically_resolves_ambiguous_minute_with_seconds(monkeypatch):
    from app import engine
    from app.engine import StrategySpec, backtest

    start = 1_800_000_000_000
    base = pd.DataFrame({
        "open_time": [start + i * 60_000 for i in range(8)],
        "open": [100.0] * 8, "high": [101.0] * 8,
        "low": [99.0] * 8, "close": [100.0] * 8, "volume": [1.0] * 8,
    })
    base.loc[4, "high"] = 105.0
    base.loc[4, "low"] = 97.0
    seconds, _ = one_second_path()
    seconds["open_time"] = [start + 4 * 60_000 + i * 1000 for i in range(60)]

    def signals(df, spec):
        out = df.copy()
        out["atr"] = 1.0
        out["long_signal"] = False
        out["short_signal"] = False
        out.loc[3, "long_signal"] = True
        return out

    monkeypatch.setattr(engine, "indicators", signals)
    monkeypatch.setattr(engine, "download_binance_execution_window", lambda symbol, start_ms, interval_ms: seconds.copy())

    result = backtest(base, StrategySpec(fast=2, slow=3, atr_period=2, stop_atr=1))
    assert len(result["ledger"]) == 1
    trade = result["ledger"][0]
    assert trade["exit_reason"] == "SL"
    assert trade["resolution_status"] == "resolved_1s_ohlc_auto"
    assert trade["execution_ambiguity"] is False
    assert result["metrics"]["execution_resolution"] == "1s_ohlc_auto"
    assert result["metrics"]["execution_data_hash_sha256"]


def test_download_binance_execution_window_validates_exact_response(monkeypatch):
    from app import data

    start = 1_900_000_000_000
    payload = [[start + i * 1000, "100", "101", "99", "100", "1", 0, 0, 0, 0, 0, 0] for i in range(60)]

    class Response:
        def raise_for_status(self): pass
        def json(self): return payload

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get(self, url, params):
            assert url.endswith("/fapi/v1/klines")
            assert params["interval"] == "1s"
            assert params["startTime"] == start
            assert params["limit"] == 60
            return Response()

    monkeypatch.setattr(data.httpx, "Client", Client)
    data._cached_binance_execution_rows.cache_clear()
    frame = data.download_binance_execution_window("ETHUSDT", start, 60_000)
    assert len(frame) == 60
    assert int(frame.open_time.iloc[0]) == start
    assert int(frame.open_time.iloc[-1]) == start + 59_000



def test_open_position_at_end_is_reported_and_cannot_pass_targets(monkeypatch):
    from app import engine
    from app.engine import StrategySpec, backtest
    start=1_800_000_000_000
    base=pd.DataFrame({"open_time":[start+i*60_000 for i in range(8)],"open":[100.0]*8,
        "high":[100.5]*8,"low":[99.5]*8,"close":[100.0]*8,"volume":[1.0]*8})
    def signals(df,spec):
        out=df.copy(); out["atr"]=1.0; out["long_signal"]=False; out["short_signal"]=False
        out.loc[6,"long_signal"]=True
        return out
    monkeypatch.setattr(engine,"indicators",signals)
    result=backtest(base,StrategySpec(fast=2,slow=3,atr_period=2,stop_atr=1))
    assert result["open_position"] is not None
    assert result["open_position"]["entry_time"]==start+7*60_000
    assert result["metrics"]["open_positions_at_end"]==1
    assert result["metrics"]["passes_all_targets"] is False
