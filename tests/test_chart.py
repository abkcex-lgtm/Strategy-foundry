import pandas as pd
from fastapi.testclient import TestClient

import app.main as main


def test_chart_includes_trades_that_overlap_window_without_entry_or_exit_inside():
    dataset_id = "chart-overlap-test"
    run_id = "chart-overlap-run"
    frame = pd.DataFrame({
        "open_time": [100, 200, 300, 400],
        "open": [100.0, 100.0, 100.0, 100.0],
        "high": [101.0, 101.0, 101.0, 101.0],
        "low": [99.0, 99.0, 99.0, 99.0],
        "close": [100.5, 100.5, 100.5, 100.5],
        "volume": [1.0, 1.0, 1.0, 1.0],
    })
    trade = {
        "trade_id": 1, "signal_time": 0, "entry_time": 50, "exit_time": 450,
        "direction": "LONG", "entry_price": 100.0, "stop_loss": 95.0,
        "take_profit": 110.0, "exit_price": 110.0, "exit_reason": "TP",
        "net_pnl": 35.0,
    }
    main.DATASETS[dataset_id] = frame
    main.RUNS[run_id] = {"run_id": run_id, "dataset_id": dataset_id, "ledger": [trade]}

    try:
        response = TestClient(main.app).get(f"/api/runs/{run_id}/chart?offset=1&limit=2")
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["candles"]) == 2
        assert len(body["trades"]) == 1
        assert body["trades"][0]["trade_id"] == 1
    finally:
        main.DATASETS.pop(dataset_id, None)
        main.RUNS.pop(run_id, None)
