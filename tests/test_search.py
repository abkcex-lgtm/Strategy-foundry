import hashlib

import pandas as pd

import app.engine as engine
import app.audit as audit
from app.engine import normalize_contract, search_candidates


def test_search_explores_eighteen_families_and_returns_five_near_misses(monkeypatch):
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

    assert set(seen) == {"ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout", "ema_pullback", "macd_crossover", "bollinger_reversion", "inside_bar_breakout", "engulfing_reversal", "momentum_breakout", "stochastic_reversion", "roc_momentum", "adx_trend", "keltner_breakout", "atr_squeeze_breakout", "obv_trend", "vwap_reclaim", "pivot_reversal"}
    assert result["evaluated_candidates"] == 60
    assert result["all_passing"] is False
    assert result["requested_near_misses"] == 5
    assert result["returned_near_misses"] == 5
    assert result["near_miss_shortfall"] == 0
    assert len(result["results"]) == 5
    assert "TRAIN" in result["warning"]
    assert set(result["results"][0]["failed_conditions"]) == {
        "win_rate_pct", "profit_factor", "avg_net_tp_per_day"
    }
    assert "criterion_distances" in result["results"][0]


def test_infinite_profit_factor_is_not_ranked_as_zero(monkeypatch):
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
    assert result["results"][0]["metrics"]["profit_factor"] is None
