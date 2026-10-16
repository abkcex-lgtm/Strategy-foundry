import pandas as pd
import pytest

from app.factory_pipeline import FactoryPreflightError, run_validation_pipeline


def frame_for_reference_splits():
    times = []
    for start in ("2026-04-01", "2026-07-01", "2026-08-01"):
        times.extend(pd.date_range(start, periods=100, freq="min", tz="UTC"))
    ms = [int(ts.timestamp() * 1000) for ts in times]
    return pd.DataFrame({
        "open_time": ms,
        "open": [100.0] * len(ms),
        "high": [101.0] * len(ms),
        "low": [99.0] * len(ms),
        "close": [100.5] * len(ms),
        "volume": [1.0] * len(ms),
    })


def metrics(passes, wr=45.0, cl=4):
    return {
        "passes_all_targets": passes,
        "win_rate_pct": wr,
        "profit_factor": 1.2,
        "avg_net_tp_per_day": 3.1,
        "max_consecutive_losses": cl,
    }


def test_preflight_fails_before_search_when_reference_split_missing():
    calls = {"search": 0}

    def search_fn(*args, **kwargs):
        calls["search"] += 1
        raise AssertionError("search must not start")

    with pytest.raises(FactoryPreflightError, match="PREFLIGHT FAILED"):
        run_validation_pipeline(frame_for_reference_splits().iloc[:150], search_fn=search_fn)
    assert calls["search"] == 0


def test_pipeline_uses_train_then_dev_and_reports_oos_without_oos_ranking():
    spec = {"name": "fixture", "fast": 5, "slow": 20, "atr_period": 5, "stop_atr": 1.5, "direction": "both", "seed": 42}
    calls = []

    def search_fn(frame, max_candidates, seed, contract):
        calls.append(("train", int(frame.open_time.iloc[0])))
        return {
            "requested_candidates": max_candidates,
            "evaluated_candidates": 1,
            "failed_candidates": 0,
            "seed": seed,
            "contract": {},
            "results": [{
                "strategy": spec,
                "metrics": metrics(True),
                "distance": 0.0,
                "run_hash_sha256": "trainhash",
            }],
        }

    def backtest_fn(frame, strategy, contract):
        first = int(frame.open_time.iloc[0])
        calls.append(("backtest", first))
        # OOS deliberately reports failure: it must not change the already-selected candidate ordering.
        is_oos = pd.to_datetime(first, unit="ms", utc=True) >= pd.Timestamp("2026-08-01", tz="UTC")
        return {
            "strategy": strategy,
            "metrics": metrics(not is_oos, wr=20.0 if is_oos else 45.0),
            "run_hash_sha256": "ooshash" if is_oos else "devhash",
            "ledger": [],
        }

    result = run_validation_pipeline(frame_for_reference_splits(), search_fn=search_fn, backtest_fn=backtest_fn)
    assert result["status"] == "COMPLETED"
    assert result["selection_policy"].startswith("Train search; Development ranking; OOS final report only")
    assert len(result["results"]) == 1
    assert result["results"][0]["oos_pass"] is False
    assert result["results"][0]["passes_train_development_oos"] is False
    assert calls[0][0] == "train"
    assert [x[0] for x in calls].count("backtest") == 2
