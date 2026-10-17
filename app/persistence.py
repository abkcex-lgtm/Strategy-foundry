from __future__ import annotations
import json, os, sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

# Vercel's function filesystem is read-only except for /tmp; its storage is ephemeral.
_DEFAULT_DB = Path("/tmp/strategy_foundry.sqlite3") if os.environ.get("VERCEL") else (Path(__file__).resolve().parent.parent / "data" / "foundry.sqlite3")
DB_PATH = Path(os.environ.get("FOUNDRY_DB_PATH", str(_DEFAULT_DB)))

def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    c.executescript("""
    CREATE TABLE IF NOT EXISTS datasets (
      dataset_id TEXT PRIMARY KEY, csv_text TEXT NOT NULL, metadata_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS runs (
      run_id TEXT PRIMARY KEY, dataset_id TEXT, result_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS profiles (
      profile_name TEXT PRIMARY KEY, config_json TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS searches (
      search_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, request_json TEXT NOT NULL,
      result_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """)
    return c

def load_dataset_metadata() -> dict[str, dict[str, Any]]:
    c = _connect()
    try:
        rows = c.execute("SELECT dataset_id,metadata_json FROM datasets").fetchall()
        out: dict[str, dict[str, Any]] = {}
        for dataset_id, payload in rows:
            try:
                value = json.loads(payload)
                out[dataset_id] = value if isinstance(value, dict) else {}
            except (ValueError, TypeError):
                out[dataset_id] = {}
        return out
    finally:
        c.close()


def save_dataset(dataset_id: str, frame: pd.DataFrame, metadata: dict[str, Any] | None = None):
    c = _connect()
    try:
        csv_text = frame.to_csv(index=False, float_format="%.12g")
        c.execute("INSERT OR REPLACE INTO datasets(dataset_id,csv_text,metadata_json) VALUES(?,?,?)",
                  (dataset_id, csv_text, json.dumps(metadata or {}, sort_keys=True)))
        c.commit()
    finally: c.close()

def load_datasets() -> dict[str, pd.DataFrame]:
    c = _connect()
    try:
        rows = c.execute("SELECT dataset_id,csv_text FROM datasets").fetchall()
        return {k: pd.read_csv(__import__("io").StringIO(v)) for k,v in rows}
    finally: c.close()

def save_run(run_id: str, dataset_id: str | None, result: dict[str, Any]):
    c = _connect()
    try:
        c.execute("INSERT OR REPLACE INTO runs(run_id,dataset_id,result_json) VALUES(?,?,?)",
                  (run_id, dataset_id, json.dumps(result, ensure_ascii=False, default=_json_default, allow_nan=False)))
        c.commit()
    finally: c.close()

def _json_default(obj):
    if hasattr(obj, "item"): return obj.item()
    if hasattr(obj, "isoformat"): return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")

def load_runs() -> dict[str, dict[str, Any]]:
    c = _connect()
    try:
        rows = c.execute("SELECT run_id,result_json FROM runs").fetchall()
        out = {}
        for k,v in rows:
            try: out[k] = json.loads(v)
            except (ValueError, TypeError): continue
        return out
    finally: c.close()

def list_history(limit: int = 100):
    c = _connect()
    try:
        rows = c.execute("SELECT run_id,dataset_id,created_at,result_json FROM runs ORDER BY created_at DESC LIMIT ?",
                         (max(1,min(int(limit),500)),)).fetchall()
        out=[]
        for run_id,dataset_id,created_at,payload in rows:
            try: r=json.loads(payload)
            except Exception: r={}
            m=r.get("metrics",{})
            out.append({"run_id":run_id,"dataset_id":dataset_id,"created_at":created_at,
                        "strategy":r.get("strategy",{}),"contract":r.get("contract",{}),
                        "metrics":m,"run_hash_sha256":r.get("run_hash_sha256")})
        return out
    finally: c.close()

def save_profile(name: str, config: dict[str, Any]):
    c = _connect()
    try:
        c.execute("INSERT OR REPLACE INTO profiles(profile_name,config_json,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
                  (name, json.dumps(config, ensure_ascii=False, sort_keys=True)))
        c.commit()
    finally: c.close()

def list_profiles():
    c = _connect()
    try:
        return [{"profile_name":n,"config":json.loads(cfg),"updated_at":t}
                for n,cfg,t in c.execute("SELECT profile_name,config_json,updated_at FROM profiles ORDER BY profile_name")]
    finally: c.close()

def save_search(search_id: str, dataset_id: str, request: dict[str, Any], result: dict[str, Any]):
    c = _connect()
    try:
        c.execute(
            "INSERT OR REPLACE INTO searches(search_id,dataset_id,request_json,result_json) VALUES(?,?,?,?)",
            (
                search_id,
                dataset_id,
                json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False),
                json.dumps(result, ensure_ascii=False, sort_keys=True, default=_json_default, allow_nan=False),
            ),
        )
        c.commit()
    finally:
        c.close()


def list_search_history(limit: int = 30):
    c = _connect()
    try:
        rows = c.execute(
            "SELECT search_id,dataset_id,created_at,request_json,result_json FROM searches ORDER BY created_at DESC LIMIT ?",
            (max(1, min(int(limit), 200)),),
        ).fetchall()
        out = []
        for search_id, dataset_id, created_at, request_json, result_json in rows:
            try:
                request = json.loads(request_json)
                result = json.loads(result_json)
            except (ValueError, TypeError):
                request, result = {}, {}
            out.append({
                "search_id": search_id,
                "dataset_id": dataset_id,
                "created_at": created_at,
                "requested_candidates": result.get("requested_candidates", request.get("candidates")),
                "evaluated_candidates": result.get("evaluated_candidates", 0),
                "failed_candidates": result.get("failed_candidates", 0),
                "seed": result.get("seed", request.get("seed")),
                "all_passing": bool(result.get("all_passing", False)),
                "result_count": len(result.get("results", [])),
                "search_period": result.get("search_period"),
            })
        return out
    finally:
        c.close()


def load_search_result(search_id: str):
    c = _connect()
    try:
        row = c.execute("SELECT result_json FROM searches WHERE search_id=?", (search_id,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except (ValueError, TypeError):
            return None
    finally:
        c.close()

