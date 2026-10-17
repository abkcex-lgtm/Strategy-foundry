import time

import pandas as pd
from fastapi.testclient import TestClient

import app.main as main


def test_search_job_is_tracked_and_uses_train_only(monkeypatch):
    dataset_id = "search-job-test"
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

    def fake_search(train, candidates, seed, contract, algorithm="hybrid_genetic", search_space=None, progress_callback=None):
        assert len(train) == 120
        if progress_callback:
            progress_callback({"evaluated_candidates": candidates, "requested_candidates": candidates})
        return {
            "requested_candidates": candidates,
            "evaluated_candidates": candidates,
            "failed_candidates": 0,
            "seed": seed,
            "algorithm": algorithm,
            "search_space": search_space or {},
            "contract": contract or {},
            "all_passing": False,
            "results": [],
            "warning": "test-only stub",
        }

    monkeypatch.setattr(main, "search_candidates", fake_search)
    try:
        with TestClient(main.app) as client:
            response = client.post(
                f"/api/search-jobs/{dataset_id}",
                json={"candidates": 7, "seed": 42, "contract": {}, "search_space": {"families": ["ema_crossover"]}},
            )
            assert response.status_code == 200
            job_id = response.json()["job_id"]

            state = None
            for _ in range(500):
                poll = client.get(f"/api/jobs/{job_id}")
                assert poll.status_code == 200
                state = poll.json()
                if state["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.01)

            assert state is not None
            assert state["status"] == "completed", state
            assert state["result"]["evaluated_candidates"] == 7
            assert "OOS is not read for candidate selection" in state["result"]["selection_policy"]
    finally:
        monkeypatch.delitem(main.DATASETS, dataset_id, raising=False)

def test_search_job_requires_existing_dataset():
    client = TestClient(main.app)
    response = client.post(
        "/api/search-jobs/does-not-exist",
        json={"candidates": 5, "seed": 42, "contract": {}},
    )
    assert response.status_code == 404


def test_search_history_endpoints_expose_saved_results(monkeypatch):
    saved = {
        "search_id": "persisted-search",
        "dataset_id": "ETHUSDT-1m",
        "results": [{"strategy": {"name": "EMA test"}}],
        "all_passing": False,
    }
    monkeypatch.setattr(main, "list_search_history", lambda limit: [{"search_id": "persisted-search", "result_count": 1}])
    monkeypatch.setattr(main, "load_search_result", lambda search_id: saved if search_id == "persisted-search" else None)
    client = TestClient(main.app)

    listing = client.get("/api/search-history?limit=5")
    assert listing.status_code == 200
    assert listing.json()[0]["search_id"] == "persisted-search"

    detail = client.get("/api/search-history/persisted-search")
    assert detail.status_code == 200
    assert detail.json()["results"][0]["strategy"]["name"] == "EMA test"

    missing = client.get("/api/search-history/missing")
    assert missing.status_code == 404
