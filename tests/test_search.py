import hashlib

import numpy as np
import pandas as pd

import app.engine as engine
import app.audit as audit
from app.engine import normalize_contract, search_candidates


def test_search_explores_all_supported_families_and_returns_five_near_misses(monkeypatch):
    seen = []

    def fake_backtest(frame, spec, contract):
        seen.append(spec.family)
        metrics = {
            "trades": 10,
            "wins": 2,
            "losses": 8,
            "win_rate_pct": 20.0,
            "profit_factor": 0.5,
            "avg_net_tp_per_day": 0.1,
            "max_consecutive_losses": 8,
            "net_pnl": -50.0,
            "final_balance": 950.0,
            "total_costs": 1.0,
            "data_hash_sha256": "d" * 64,
            "rows": 500,
            "span_days": 1.0,
            "passes_all_targets": False,
        }
        return {
            "strategy": engine.asdict(spec),
            "contract": contract,
            "metrics": metrics,
            "run_hash_sha256": hashlib.sha256(repr(spec).encode()).hexdigest(),
            "ledger": [],
        }

    monkeypatch.setattr(engine, "backtest", fake_backtest)
    monkeypatch.setattr(audit, "independent_replay", lambda frame, spec, cfg: {"ledger": []})
    result = search_candidates(pd.DataFrame(), max_candidates=60, seed=42, contract=normalize_contract())

    assert set(seen) == {"ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout", "ema_pullback", "macd_crossover", "bollinger_reversion", "inside_bar_breakout", "engulfing_reversal", "momentum_breakout", "stochastic_reversion", "roc_momentum", "adx_trend", "keltner_breakout", "atr_squeeze_breakout", "obv_trend", "vwap_reclaim", "pivot_reversal", "swing_structure_break", "failed_breakout_reversal", "wick_rejection", "range_expansion_breakout", "three_bar_momentum", "volatility_regime_breakout", "trend_exhaustion_reversal", "volume_climax_reversal", "range_compression_breakout", "support_resistance_rejection", "micro_double_bottom", "micro_double_top", "williams_r_reversion", "donchian_midline_trend", "dual_thrust_breakout", "chaikin_money_flow", "money_flow_index_reversion", "atr_trend_pullback", "volume_weighted_breakout", "range_midpoint_reversion", "roc_acceleration", "atr_normalized_momentum"}
    assert result["evaluated_candidates"] == 60
    assert result["all_passing"] is False
    assert result["requested_near_misses"] == 5
    assert result["returned_near_misses"] == 5
    assert result["near_miss_shortfall"] == 0
    assert len(result["results"]) == 5
    assert "TRAIN" in result["warning"]
    assert set(result["results"][0]["failed_conditions"]) == {
        "win_rate_pct", "profit_factor", "avg_net_tp_per_day",
        "insufficient_activity_evidence"
    }
    assert "criterion_distances" in result["results"][0]


def test_disabled_profit_factor_does_not_affect_candidate_distance(monkeypatch):
    def fake_backtest(frame, spec, contract):
        metrics = {
            "trades": 10, "wins": 2, "losses": 8, "win_rate_pct": 20.0,
            "profit_factor": None if spec.family == "rsi_reversion" else 0.5,
            "avg_net_tp_per_day": 0.1, "max_consecutive_losses": 8,
            "net_pnl": -50.0, "final_balance": 950.0, "total_costs": 1.0,
            "data_hash_sha256": "d" * 64, "rows": 500, "span_days": 1.0,
            "passes_all_targets": False,
        }
        return {
            "strategy": engine.asdict(spec), "contract": contract, "metrics": metrics,
            "run_hash_sha256": hashlib.sha256(repr(spec).encode()).hexdigest(),
            "ledger": [],
        }

    monkeypatch.setattr(engine, "backtest", fake_backtest)
    monkeypatch.setattr(audit, "independent_replay", lambda frame, spec, cfg: {"ledger": []})
    cfg = normalize_contract({"use_min_profit_factor": False})
    result = search_candidates(pd.DataFrame(), max_candidates=60, seed=42, contract=cfg)
    assert "profit_factor" not in result["results"][0]["criterion_distances"]
    assert len(result["results"]) == 5


def test_search_space_restricts_families_and_composition_filters(monkeypatch):
    seen = []
    def fake_backtest(frame, spec, contract):
        seen.append((spec.family, spec.context_filter, spec.confirmation_filter))
        metrics = {
            "trades": 10, "wins": 2, "losses": 8, "win_rate_pct": 20.0,
            "profit_factor": 0.5, "avg_net_tp_per_day": 0.1,
            "max_consecutive_losses": 8, "net_pnl": -50.0,
            "final_balance": 950.0, "total_costs": 1.0,
            "data_hash_sha256": "d" * 64, "rows": 500, "span_days": 1.0,
            "passes_all_targets": False,
        }
        return {"strategy": engine.asdict(spec), "contract": contract, "metrics": metrics,
                "run_hash_sha256": hashlib.sha256(repr(spec).encode()).hexdigest(), "ledger": []}
    monkeypatch.setattr(engine, "backtest", fake_backtest)
    monkeypatch.setattr(audit, "independent_replay", lambda frame, spec, cfg: {"ledger": []})
    result = search_candidates(pd.DataFrame(), max_candidates=12, seed=7,
        search_space={"families": ["ema_crossover"],
                      "context_filters": ["none", "ema_trend"],
                      "confirmation_filters": ["none"]})
    assert result["evaluated_candidates"] == 12
    assert set(f for f, _, _ in seen) == {"ema_crossover"}
    assert set(cf for _, cf, _ in seen) <= {"none", "ema_trend"}
    assert set(xf for _, _, xf in seen) == {"none"}


def test_empty_search_space_selection_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="family selection is empty"):
        search_candidates(pd.DataFrame(), max_candidates=1,
                          search_space={"families": []})


def test_new_family_signal_arrays_match_independent_replay():
    # Deterministic unit-test fixture only; never used as market research data.
    n = 1200
    idx = np.arange(n, dtype=float)
    close = 2500.0 + 0.04 * idx + 15.0 * np.sin(idx / 17.0) + 4.0 * np.sin(idx / 3.7)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.8 + (idx % 7) * 0.05
    low = np.minimum(open_, close) - 0.7 - (idx % 5) * 0.04
    frame = pd.DataFrame({
        "open_time": np.arange(n, dtype=np.int64) * 60_000,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": 100.0 + (idx % 31) * 13.0 + 20.0 * np.sin(idx / 5.0),
    })
    families = (
        "atr_trend_pullback", "volume_weighted_breakout",
        "range_midpoint_reversion", "roc_acceleration",
        "atr_normalized_momentum",
    )
    for family in families:
        spec = engine.StrategySpec(family=family, fast=8, slow=24, atr_period=14)
        primary = engine.indicators(frame, spec)
        long_, short_, atr, *_ = audit._independent_signals(frame, {
            "family": family, "fast": spec.fast, "slow": spec.slow,
            "atr_period": spec.atr_period, "context_filter": "none",
            "confirmation_filter": "none", "direction": "both",
            "secondary_family": "none", "signal_composition": "single",
        })
        np.testing.assert_array_equal(primary["long_signal"].fillna(False).to_numpy(), long_.fillna(False).to_numpy())
        np.testing.assert_array_equal(primary["short_signal"].fillna(False).to_numpy(), short_.fillna(False).to_numpy())
        np.testing.assert_allclose(primary["atr"].to_numpy(), atr.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)
