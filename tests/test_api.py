from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)

def test_health_endpoint():
    r=client.get('/api/health')
    assert r.status_code==200
    assert r.json()['live_trading'] is False
    assert r.json()['features']['tracked_binance_download_jobs'] is True
    assert r.json()['features']['tracked_search_jobs'] is True
    assert r.json()['features']['persistent_search_history'] is True
    assert r.json()['features']['search_history_endpoint']=='/api/search-history'
    assert r.json()['features']['search_job_endpoint']=='/api/search-jobs/{dataset_id}'
    assert r.json()['version']=='0.4.0'

def test_home_page_serves_persian_ui():
    r=client.get('/')
    assert r.status_code==200
    assert 'Strategy Foundry' in r.text
    assert 'خلاصهٔ زندهٔ پژوهش' in r.text
    assert 'id="chartZoom"' in r.text
    assert 'id="chartLatest"' in r.text
    assert 'id="searchHistoryList"' in r.text
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
