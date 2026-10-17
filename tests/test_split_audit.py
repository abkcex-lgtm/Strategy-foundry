import hashlib

import pandas as pd
from fastapi.testclient import TestClient

import app.main as main


def test_split_evaluation_run_can_be_audited_against_exact_split(monkeypatch):
    dataset_id = "audit-split-test"
    start = int(pd.Timestamp("2026-04-01T00:00:00Z").timestamp() * 1000)
    frame = pd.DataFrame({
        "open_time": [start + i * 60_000 for i in range(120)],
        "open": [100.0] * 120,
        "high": [101.0] * 120,
        "low": [99.0] * 120,
        "close": [100.5] * 120,
        "volume": [1.0] * 120,
    })
    monkeypatch.setitem(main.DATASETS, dataset_id, frame)
    run_hash = "a" * 64

    def fake_backtest(part, spec=None, contract=None):
        data_hash = hashlib.sha256(
            part.to_csv(index=False, float_format="%.12g").encode()
        ).hexdigest()
        return {
            "strategy": {"name": "test", "fast": 5, "slow": 10, "atr_period": 5, "stop_atr": 1.5, "direction": "both"},
            "contract": {},
            "metrics": {"data_hash_sha256": data_hash, "max_consecutive_losses": 0, "trades": 0},
            "ledger": [],
            "run_hash_sha256": run_hash,
        }

    monkeypatch.setattr(main, "backtest", fake_backtest)
    monkeypatch.setattr(main, "independent_replay", lambda *args: {"ledger": [], "max_consecutive_losses": 0})
    monkeypatch.setattr(main, "save_run", lambda *args, **kwargs: None)
    client = TestClient(main.app)
    response = client.post(
        f"/api/evaluate/{dataset_id}",
        json={"strategy": {"name": "test"}, "contract": {}},
    )
    assert response.status_code == 200, response.text
    train = response.json()["periods"]["train"]
    assert train["status"] == "COMPLETED"
    run_id = train["run_id"]
    assert main.RUNS[run_id]["evaluation_split"] == "train"

    audit = client.get(f"/api/runs/{run_id}/audit")
    assert audit.status_code == 200, audit.text
    assert audit.json()["status"] == "MATCH"
    assert audit.json()["matched"] is True

    main.RUNS.pop(run_id, None)
    main.RUN_DATASET.pop(run_id, None)
    monkeypatch.delitem(main.DATASETS, dataset_id, raising=False)
