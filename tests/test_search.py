import hashlib

import pandas as pd

import app.engine as engine
from app.engine import normalize_contract, search_candidates


def test_search_explores_four_families_and_returns_five_near_misses(monkeypatch):
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
        }

    monkeypatch.setattr(engine, "backtest", fake_backtest)
    result = search_candidates(pd.DataFrame(), max_candidates=60, seed=42, contract=normalize_contract())

    assert set(seen) == {"ema_crossover", "channel_breakout", "rsi_reversion", "volume_breakout"}
    assert result["evaluated_candidates"] == 60
    assert result["all_passing"] is False
    assert result["requested_near_misses"] == 5
    assert result["returned_near_misses"] == 5
    assert result["near_miss_shortfall"] == 0
    assert len(result["results"]) == 5
    assert "TRAIN" in result["warning"]
