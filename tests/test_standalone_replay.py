import json
import subprocess
import sys

import pandas as pd

from app.engine import StrategySpec, backtest, normalize_contract
from app.audit import independent_replay
from pathlib import Path


def flat_candles(n=320):
    return pd.DataFrame({
        "open_time": [i * 60_000 for i in range(n)],
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [10.0] * n,
    })


def test_standalone_template_contains_all_engine_families_and_composition():
    template = (Path(__file__).resolve().parents[1] / "app" / "standalone_template.py").read_text()
    engine = (Path(__file__).resolve().parents[1] / "app" / "engine.py").read_text()
    assert "micro_double_top" in template
    assert "signal_composition" in template
    assert "secondary_family" in template
    assert "bar_open<=stop" in template
    assert "bar_open>=stop" in template
    assert "Unresolved same-candle SL/TP ambiguity" in template
    assert "def indicators(df, spec: StrategySpec)" in engine
    compile(template, "standalone_template.py", "exec")


def test_generated_standalone_replay_verifies_hash_and_runs_without_factory_import(tmp_path):
    frame = flat_candles()
    spec = StrategySpec(fast=5, slow=18, atr_period=7, family="micro_double_top",
                        secondary_family="wick_rejection", signal_composition="or")
    contract = normalize_contract({"reward_risk": 2.0})
    result = backtest(frame, spec, contract)
    template_path = Path(__file__).resolve().parents[1] / "app" / "standalone_template.py"
    source = template_path.read_text()
    source = source.replace("__SPEC__", repr(result["strategy"]))
    source = source.replace("__CONTRACT__", repr(result["contract"]))
    source = source.replace("__DATA_HASH__", result["metrics"]["data_hash_sha256"])
    source = source.replace("__RUN_HASH__", result["run_hash_sha256"])
    script_path = tmp_path / "replay.py"
    csv_path = tmp_path / "source.csv"
    script_path.write_text(source)
    frame.to_csv(csv_path, index=False)
    completed = subprocess.run(
        [sys.executable, str(script_path), str(csv_path)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["run_hash_sha256"] == result["run_hash_sha256"]
    assert payload["metrics"]["data_hash_sha256"] == result["metrics"]["data_hash_sha256"]
    assert payload["ledger"] == result["ledger"]
