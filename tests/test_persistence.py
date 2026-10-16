import pandas as pd
from app import persistence

def test_sqlite_round_trip_dataset_profile_run(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "foundry.sqlite3")
    df=pd.DataFrame([
      {"open_time":0,"open":10.0,"high":11.0,"low":9.0,"close":10.5,"volume":3.0},
      {"open_time":60000,"open":10.5,"high":12.0,"low":10.0,"close":11.0,"volume":4.0},
    ])
    persistence.save_dataset("d1",df,{"source":"test"})
    loaded=persistence.load_datasets()
    assert list(loaded)==["d1"]
    assert loaded["d1"].to_dict("records")==df.to_dict("records")
    persistence.save_profile("editable",{"risk_usd":12.5})
    assert persistence.list_profiles()[0]["config"]["risk_usd"]==12.5
    persistence.save_run("r1","d1",{"strategy":{"name":"test"},"metrics":{"trades":2},"ledger":[],"run_hash_sha256":"abc"})
    assert persistence.load_runs()["r1"]["run_hash_sha256"]=="abc"
    assert persistence.list_history()[0]["run_id"]=="r1"
