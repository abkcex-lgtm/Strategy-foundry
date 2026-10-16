from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)

def test_health_endpoint():
    r=client.get('/api/health')
    assert r.status_code==200
    assert r.json()['live_trading'] is False

def test_home_page_serves_persian_ui():
    r=client.get('/')
    assert r.status_code==200
    assert 'Strategy Foundry' in r.text

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
