import numpy as np
import pandas as pd
import pytest

from app.engine import StrategySpec, indicators
from app.audit import _independent_signals


NEW_FAMILIES = (
    "williams_r_reversion",
    "donchian_midline_trend",
    "dual_thrust_breakout",
    "chaikin_money_flow",
    "money_flow_index_reversion",
)


def _market_frame(n=1600):
    rng = np.random.default_rng(20261019)
    returns = rng.normal(0.00002, 0.0012, n)
    close = 2500.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.00025, n))
    spread = rng.uniform(0.0002, 0.002, n)
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(4.5, 0.55, n)
    return pd.DataFrame({
        "open_time": np.arange(n, dtype=np.int64) * 60_000,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.mark.parametrize("family", NEW_FAMILIES)
def test_expanded_family_is_valid_and_primary_matches_independent_signal_engine(family):
    frame = _market_frame()
    spec = StrategySpec(family=family, fast=9, slow=36, atr_period=14,
                        stop_atr=1.5, direction="both")
    spec.validate()

    primary = indicators(frame, spec)
    independent_long, independent_short, *_ = _independent_signals(frame, {
        "family": family,
        "fast": spec.fast,
        "slow": spec.slow,
        "atr_period": spec.atr_period,
        "direction": spec.direction,
        "context_filter": "none",
        "confirmation_filter": "none",
        "signal_composition": "single",
    })

    np.testing.assert_array_equal(primary["long_signal"].fillna(False).to_numpy(),
                                  independent_long.to_numpy())
    np.testing.assert_array_equal(primary["short_signal"].fillna(False).to_numpy(),
                                  independent_short.to_numpy())
    assert len(primary) == len(frame)
