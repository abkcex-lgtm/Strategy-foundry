import pandas as pd
from fastapi.testclient import TestClient
import app.main as main
from app.main import app

client=TestClient(app)

def test_health_endpoint():
    r=client.get('/api/health')
    assert r.status_code==200
    assert r.json()['live_trading'] is False
    assert r.json()['features']['tracked_binance_download_jobs'] is True
    assert r.json()['features']['tracked_search_jobs'] is True
    assert r.json()['features']['persistent_search_history'] is True
    assert r.json()['features']['persistent_job_status'] is True
    assert r.json()['features']['search_history_endpoint']=='/api/search-history'
    assert r.json()['features']['search_job_endpoint']=='/api/search-jobs/{dataset_id}'
    assert r.json()['version']=='0.9.3'
    assert main.app.version == r.json()['version']
    assert r.json()['storage']['engine'] == 'sqlite'
    assert r.json()['storage']['database_ready'] is True
    assert r.json()['storage']['storage_mode'] in {'explicit_path', 'ephemeral_tmp', 'mounted_volume', 'local_fallback'}

def test_home_page_serves_persian_ui():
    r=client.get('/')
    assert r.status_code==200
    assert 'Strategy Foundry' in r.text
    assert 'خلاصهٔ زندهٔ پژوهش' in r.text
    assert 'id="chartZoom"' in r.text
    assert 'id="chartLatest"' in r.text
    assert 'id="tradePrev"' in r.text
    assert 'id="tradeNext"' in r.text
    assert 'value="tp"' in r.text
    assert 'id="searchHistoryList"' in r.text
    assert 'class="quick-nav"' in r.text
    assert 'id="riskSettings"' in r.text
    assert 'id="searchSettings"' in r.text
    assert 'id="chartSection"' in r.text
    assert 'id="ledgerSection"' in r.text
    assert 'id="searchResultsSection"' in r.text
    assert '<option>30m</option>' in r.text
    assert '<option>2h</option>' in r.text

def test_contract_defaults_endpoint():
    r=client.get('/api/contract')
    assert r.status_code==200
    assert r.json()['contract']['risk_usd']==10

def test_settings_validation_accepts_custom_costs_and_rr():
    r=client.post('/api/settings/validate',json={'contract':{
        'initial_balance':5000,'risk_mode':'fixed_usd','risk_usd':20,
        'reward_risk':2.25,'commission_pct_per_side':0.03,
        'slippage_pct_per_side':0.07,'target_mode':'net_rr',
        'max_positions':1,'entry_model':'next_candle_open','exit_model':'sl_tp_only'
    }})
    assert r.status_code==200
    body=r.json()
    assert body['valid'] is True
    assert body['contract']['initial_balance']==5000
    assert body['contract']['reward_risk']==2.25
    assert abs(body['effective_cost_pct_per_side']-0.10)<1e-12

def test_settings_validation_rejects_unimplemented_management():
    r=client.post('/api/settings/validate',json={'contract':{'trailing_stop_enabled':True}})
    assert r.status_code==422


def test_download_job_unknown_id_is_clear_404():
    r=client.get('/api/jobs/does-not-exist')
    assert r.status_code==404
    assert 'شناسهٔ عملیات' in r.json()['detail']

def test_download_rejects_unsupported_interval_before_scheduling():
    r=client.post('/api/data/binance',json={
        'symbol':'ETHUSDT','interval':'7m',
        'start_year':2026,'start_month':4,'end_year':2026,'end_month':4
    })
    assert r.status_code==400
    assert 'پشتیبانی‌نشده' in r.json()['detail']


def test_chart_focus_centers_selected_trade(monkeypatch):
    dataset_id='chart-focus-test'
    run_id='chart-focus-run'
    df=pd.DataFrame({
        'open_time':list(range(30)),
        'open':[100.0+i for i in range(30)],
        'high':[101.0+i for i in range(30)],
        'low':[99.0+i for i in range(30)],
        'close':[100.5+i for i in range(30)],
        'volume':[10.0]*30,
    })
    trade={
        'trade_id':7,'signal_time':10,'entry_time':12,'exit_time':15,
        'direction':'LONG','entry_price':112.0,'stop_loss':110.0,
        'take_profit':119.0,'exit_price':119.0,'exit_reason':'TP','net_pnl':35.0
    }
    monkeypatch.setitem(main.DATASETS,dataset_id,df)
    monkeypatch.setitem(main.RUNS,run_id,{'run_id':run_id,'dataset_id':dataset_id,'ledger':[trade]})
    response=client.get(f'/api/runs/{run_id}/chart?offset=0&limit=10&trade_id=7')
    assert response.status_code==200
    body=response.json()
    assert body['focused_trade_id']==7
    assert body['offset']==7
    assert len(body['candles'])==10
    assert any(t['trade_id']==7 for t in body['trades'])


def test_chart_focus_rejects_unknown_trade(monkeypatch):
    dataset_id='chart-focus-empty-test'
    run_id='chart-focus-empty-run'
    df=pd.DataFrame({
        'open_time':list(range(10)),'open':[100.0]*10,'high':[101.0]*10,
        'low':[99.0]*10,'close':[100.0]*10,'volume':[10.0]*10
    })
    monkeypatch.setitem(main.DATASETS,dataset_id,df)
    monkeypatch.setitem(main.RUNS,run_id,{'run_id':run_id,'dataset_id':dataset_id,'ledger':[]})
    response=client.get(f'/api/runs/{run_id}/chart?trade_id=999')
    assert response.status_code==404


def test_persistent_job_status_round_trip(tmp_path, monkeypatch):
    import app.persistence as persistence
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "jobs.sqlite3")
    payload = {"job_id": "job-test", "kind": "strategy_search", "status": "running", "evaluated_candidates": 25}
    persistence.save_job("job-test", "strategy_search", payload)
    restored = persistence.load_jobs()
    assert restored["job-test"]["status"] == "running"
    assert restored["job-test"]["evaluated_candidates"] == 25
    assert restored["job-test"]["kind"] == "strategy_search"


def test_search_request_accepts_batch_search_ceiling():
    from app.main import SearchRequest
    request = SearchRequest(candidates=500000, seed=42)
    assert request.candidates == 500000


def test_atomic_job_batch_checkpoint_persists_job_and_genomes(tmp_path, monkeypatch):
    import app.persistence as persistence
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "batch.sqlite3")
    payload = {"job_id": "batch-job", "status": "running",
               "completed_batches": 1, "completed_candidates": 10}
    persistence.save_job_batch_checkpoint("batch-job", "strategy_search", payload,
                                          ["abc", "def", "abc"])
    restored = persistence.load_jobs()
    assert restored["batch-job"]["completed_batches"] == 1
    assert persistence.load_job_genome_hashes("batch-job") == {"abc", "def"}


def test_search_resume_frontend_reuses_existing_job_and_handles_shortfall():
    response = client.get('/static/app.js')
    assert response.status_code == 200
    source = response.text
    assert "existingJobId?{job_id:existingJobId}:await api('/api/search-jobs/'" in source
    search_handler = source.split("async function search(")[1].split("async function evaluateStrategy(")[0]
    assert "completed_with_shortfall" in search_handler
    assert "includes(state.status)" in search_handler


def test_split_evaluation_freezes_strategy_and_hashes_each_split(monkeypatch):
    dataset_id = "frozen-evaluation-test"
    n = 600
    frame = pd.DataFrame({
        "open_time": [i * 60_000 for i in range(n)],
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [10.0] * n,
    })
    monkeypatch.setitem(main.DATASETS, dataset_id, frame)
    monkeypatch.setattr(main, "_store_result", lambda result, dataset_id: "eval-" + result["evaluation_split"])
    response = client.post(f"/api/evaluate/{dataset_id}", json={
        "strategy": {"fast": 5, "slow": 18, "atr_period": 7, "family": "micro_double_top",
                     "secondary_family": "wick_rejection", "signal_composition": "or"},
        "contract": {"reward_risk": 2.0},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["certified"] is False
    assert body["oos_used_for_selection"] is False
    assert len(body["dataset_hash_sha256"]) == 64
    assert len(body["evaluation_manifest_sha256"]) == 64
    assert body["evaluation_manifest"]["oos_used_for_selection"] is False
    assert body["evaluation_manifest"]["frozen_strategy"] == body["strategy"]
    assert body["evaluation_manifest"]["effective_contract"] == body["contract"]
    assert set(body["periods"]) == {"train", "development", "oos"}
    for split in ("train", "development", "oos"):
        result = body["periods"][split]
        assert result["status"] == "COMPLETED", result
        assert len(result["data_hash_sha256"]) == 64
        assert result["run_id"] == "eval-" + split
