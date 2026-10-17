import math

import pandas as pd
import pytest

from app.engine import StrategySpec, backtest, normalize_contract
from app.data import DataIntegrityError


def candles(n=800):
    rows = []
    for i in range(n):
        price = 100 + 0.02 * i + 2 * math.sin(i / 7)
        rows.append({
            "open_time": i * 60_000, "open": price, "high": price + 0.3,
            "low": price - 0.3, "close": price + 0.05 * math.sin(i),
            "volume": 10.0,
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("family", ["ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout", "ema_pullback", "macd_crossover", "bollinger_reversion", "inside_bar_breakout", "engulfing_reversal", "momentum_breakout"])
def test_standalone_replay_matches_factory_for_each_family(family):
    frame = candles(800)
    spec = StrategySpec(
        name=f"standalone {family}", fast=5, slow=18, atr_period=7,
        stop_atr=1.5, family=family,
    )
    cfg = normalize_contract({"risk_usd": 10, "reward_risk": 2.0})
    try:
        expected = backtest(frame, spec, cfg)
    except DataIntegrityError as exc:
        if "ambiguity" in str(exc):
            pytest.skip("Synthetic path intentionally triggered conservative ambiguity guard")
        raise

    with open("app/standalone_template.py", encoding="utf-8") as handle:
        source = handle.read()
    source = source.replace("__SPEC__", repr(expected["strategy"]))
    source = source.replace("__CONTRACT__", repr(expected["contract"]))
    source = source.replace("__DATA_HASH__", expected["metrics"]["data_hash_sha256"])
    source = source.replace("__RUN_HASH__", expected["run_hash_sha256"])
    namespace = {"__name__": "standalone_replay"}
    exec(compile(source, "standalone_replay.py", "exec"), namespace)
    actual = namespace["replay"](frame)

    assert actual["run_hash_sha256"] == expected["run_hash_sha256"]
    assert set(actual["metrics"]) == set(expected["metrics"])
    for key, value in expected["metrics"].items():
        other = actual["metrics"][key]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            assert other == value
        elif isinstance(value, (int, float)):
            assert other == pytest.approx(value, rel=1e-10, abs=1e-8)
        else:
            assert other == value
    assert len(actual["ledger"]) == len(expected["ledger"])
    for left, right in zip(expected["ledger"], actual["ledger"]):
        assert set(left) == set(right)
        for key, value in left.items():
            other = right[key]
            if isinstance(value, bool) or value is None or isinstance(value, str):
                assert other == value
            elif isinstance(value, (int, float)):
                assert other == pytest.approx(value, rel=1e-10, abs=1e-8)
            else:
                assert other == value
