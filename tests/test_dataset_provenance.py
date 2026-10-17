import hashlib

import pandas as pd
from fastapi.testclient import TestClient

import app.main as main


def test_dataset_listing_exposes_data_hash_and_provenance(monkeypatch):
    dataset_id = "provenance-test"
    frame = pd.DataFrame({
        "open_time": [1_790_000_000_000, 1_790_000_060_000],
        "open": [100.0, 100.5],
        "high": [101.0, 101.5],
        "low": [99.0, 100.0],
        "close": [100.5, 101.0],
        "volume": [1.0, 2.0],
    })
    metadata = {"source": "test", "monthly_archive_hashes": [{"sha256": "abc"}]}
    monkeypatch.setitem(main.DATASETS, dataset_id, frame)
    monkeypatch.setitem(main.DATASET_METADATA, dataset_id, metadata)
    client = TestClient(main.app)

    response = client.get("/api/datasets")
    assert response.status_code == 200
    row = next(item for item in response.json() if item["dataset_id"] == dataset_id)
    expected = hashlib.sha256(frame.to_csv(index=False, float_format="%.12g").encode()).hexdigest()
    assert row["data_hash_sha256"] == expected
    assert row["metadata"] == metadata

    monkeypatch.delitem(main.DATASETS, dataset_id, raising=False)
    monkeypatch.delitem(main.DATASET_METADATA, dataset_id, raising=False)
