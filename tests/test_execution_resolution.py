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
