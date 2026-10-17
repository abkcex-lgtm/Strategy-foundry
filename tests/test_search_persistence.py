from app import persistence


def test_search_results_and_manifest_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "search-history.sqlite3")
    search_id = "search-test-001"
    manifest = {"dataset_id": "ETHUSDT-1m-test", "candidates": 20, "seed": 42}
    result = {
        "search_id": search_id,
        "dataset_id": manifest["dataset_id"],
        "requested_candidates": 20,
        "evaluated_candidates": 18,
        "failed_candidates": 2,
        "seed": 42,
        "all_passing": False,
        "results": [{"strategy": {"name": "candidate 1"}, "distance": 0.2}],
        "search_period": "TRAIN only",
        "search_manifest_sha256": "deadbeef",
    }

    persistence.save_search(search_id, manifest["dataset_id"], manifest, result)
    loaded = persistence.load_search_result(search_id)
    assert loaded == result
    history = persistence.list_search_history(10)
    assert len(history) == 1
    assert history[0]["search_id"] == search_id
    assert history[0]["evaluated_candidates"] == 18
    assert history[0]["result_count"] == 1
    assert history[0]["search_manifest_sha256"] == "deadbeef"
