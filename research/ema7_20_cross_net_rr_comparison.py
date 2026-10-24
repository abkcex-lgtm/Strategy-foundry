"""ETHUSDT 1m EMA 7/20 crossover with swing stop and net RR 2R/3R."""
import asyncio, json, hashlib, time, inspect
from pathlib import Path
import numpy as np
import pandas as pd
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
# Keep the real $1,000 account run and add a clearly labeled full-horizon shadow ledger.
ACTUAL_RUN_COMBINED = sm.run_combined
_run_src = inspect.getsource(sm.run_combined)
if "if bal<=10+1e-9: capital=True; break" not in _run_src:
    raise RuntimeError("Expected capital-stop guard not found; refusing an unverified patch")
_full_src = _run_src.replace("if bal<=10+1e-9: capital=True; break", "if False: capital=True; break")
exec(_full_src, sm.__dict__)
FULL_HORIZON_RUN_COMBINED = sm.run_combined
sm.run_combined = ACTUAL_RUN_COMBINED
BASE_CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
 "reward_risk":2.0,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
 "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only",
 "break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}

def cross_signals(df):
 c=df.close.astype(float)
 fast=c.ewm(span=7,adjust=False,min_periods=7).mean()
 slow=c.ewm(span=20,adjust=False,min_periods=20).mean()
 long=(fast>slow)&(fast.shift(1)<=slow.shift(1))
 short=(fast<slow)&(fast.shift(1)>=slow.shift(1))
 return long.fillna(False),short.fillna(False)

def safe_metrics(m):
 return {k:(float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v) for k,v in m.items()}

async def main():
 started=time.monotonic()
 df,manifest=await get_data()
 splits=split_reference_period(df)
 rows=[]
 for split in ("train","development"):
  part=splits[split]
  sig_l,sig_s=cross_signals(part)
  for rr in (2.0,3.0):
   sm.CONTRACT=normalize_contract({**BASE_CFG,"reward_risk":rr})
   metrics,ledger=ACTUAL_RUN_COMBINED(part,sig_l,sig_s,pd.Series(np.nan,index=part.index),"swing_5")
   rows.append({"split":split,"account_mode":"initial_$1000_executable_until_ruin","strategy":f"EMA7_20_CROSS_NET_{rr:g}R","ema_fast":7,"ema_slow":20,
    "entry":"cross close signal, next candle open","stop":"lowest low / highest high of signal candle and prior 4 candles (swing_5)",
    "reward_risk_net":rr,"metrics":safe_metrics(metrics),
    "ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()})
   metrics,ledger=FULL_HORIZON_RUN_COMBINED(part,sig_l,sig_s,pd.Series(np.nan,index=part.index),"swing_5")
   rows.append({"split":split,"account_mode":"full_horizon_shadow_fixed_$10_risk_not_capital_feasible","strategy":f"EMA7_20_CROSS_NET_{rr:g}R","ema_fast":7,"ema_slow":20,"entry":"cross close signal, next candle open","stop":"swing_5","reward_risk_net":rr,"metrics":safe_metrics(metrics),"ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()})
  # Historical candidate as recorded in the existing pair sweep; retain its original 3.5R contract.
  sm.CONTRACT=normalize_contract({**BASE_CFG,"reward_risk":3.5})
  sigs,atr=sm.signals(part); filters=sm.confirmation_filters(part)
  raw_l,raw_s=sigs["structure_break_pb8_br15"]
  fl,_=filters["strong_candle_close"]; _,fs=filters["baseline"]
  metrics,ledger=ACTUAL_RUN_COMBINED(part,raw_l & fl.fillna(False),raw_s & fs.fillna(False),atr,"swing_5")
  rows.append({"split":split,"account_mode":"initial_$1000_executable_until_ruin","strategy":"REFERENCE_HISTORICAL_11_LOSS_CANDIDATE","entry":"structure_break_pb8_br15; strong-candle long filter; baseline short filter",
   "stop":"swing_5","reward_risk_net":3.5,"metrics":safe_metrics(metrics),
   "ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()})
  metrics,ledger=FULL_HORIZON_RUN_COMBINED(part,raw_l & fl.fillna(False),raw_s & fs.fillna(False),atr,"swing_5")
  rows.append({"split":split,"account_mode":"full_horizon_shadow_fixed_$10_risk_not_capital_feasible","strategy":"REFERENCE_HISTORICAL_11_LOSS_CANDIDATE","entry":"structure_break_pb8_br15; strong-candle long filter; baseline short filter","stop":"swing_5","reward_risk_net":3.5,"metrics":safe_metrics(metrics),"ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()})
 report={"schema":"eth_ema7_20_cross_swingstop_net_rr_comparison_v2_actual_and_full_horizon","data_manifest":manifest,
  "data_scope":"Binance USD-M ETHUSDT 1m Apr-Sep 2026; train and development only; OOS untouched",
  "split_rows":{k:len(v) for k,v in splits.items()},"fixed_costs":{"commission_pct_per_side":0.04,"slippage_pct_per_side":0.02},
  "execution":"shared chronological LONG+SHORT single-position ledger; next candle open; SL/TP only; fixed $10 risk; no compounding",
  "stop_definition":"swing_5: lowest low / highest high over signal candle and previous four candles",
  "same_bar_ambiguity":"conservatively resolved as stop loss by engine; no 1-second audit claimed",
  "elapsed_seconds":round(time.monotonic()-started,2),"results":rows}
 p=OUT/"ema7_20_cross_net_rr_comparison.json"
 p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
 print("EMA7_20_COMPARISON_FINAL "+json.dumps({"report":str(p),"elapsed_seconds":report["elapsed_seconds"],"results":rows},sort_keys=True,allow_nan=False),flush=True)
if __name__=="__main__": asyncio.run(main())
