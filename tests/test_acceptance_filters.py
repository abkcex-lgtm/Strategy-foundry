import pandas as pd

import app.engine as engine


def test_disabled_acceptance_filters_do_not_add_search_distance(monkeypatch):
    def fake_backtest(frame, spec, contract):
        return {
            "strategy": {"name": spec.name, "fast": spec.fast, "slow": spec.slow, "atr_period": spec.atr_period, "stop_atr": spec.stop_atr, "direction": spec.direction, "seed": spec.seed},
            "contract": contract,
            "metrics": {
                "passes_all_targets": False,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "avg_net_tp_per_day": 0.0,
                "max_consecutive_losses": 99,
                "trades": 0,
            },
            "run_hash_sha256": "test-hash",
        }
    monkeypatch.setattr(engine, "backtest", fake_backtest)
    cfg = {
        "use_min_win_rate": False,
        "use_min_profit_factor": False,
        "use_min_tp_per_day": False,
        "use_max_consecutive_losses": False,
    }
    result = engine.search_candidates(pd.DataFrame(), max_candidates=1, seed=42, contract=cfg)
    assert result["results"][0]["distance"] == 0.0


def test_acceptance_filters_default_to_enabled():
    cfg = engine.normalize_contract({})
    assert cfg["use_min_win_rate"] is True
    assert cfg["use_min_profit_factor"] is True
    assert cfg["use_min_tp_per_day"] is True
    assert cfg["use_max_consecutive_losses"] is True
