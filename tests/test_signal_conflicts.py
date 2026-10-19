import pandas as pd

from app.engine import _cancel_conflicting_signals


def test_conflicting_long_and_short_signals_are_both_cancelled():
    long_signal = pd.Series([True, True, False, False, True])
    short_signal = pd.Series([False, True, True, False, True])

    long_result, short_result = _cancel_conflicting_signals(long_signal, short_signal)

    assert long_result.tolist() == [True, False, False, False, False]
    assert short_result.tolist() == [False, False, True, False, False]
    assert not (long_result & short_result).any()
