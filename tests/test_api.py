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
