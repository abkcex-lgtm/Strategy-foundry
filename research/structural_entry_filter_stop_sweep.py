"""Targeted structural-entry/filter/stop sweep; OOS untouched."""
import asyncio, json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,
"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
STOPS=("signal_candle","swing_3","swing_5","atr_0.5","atr_0.75","atr_1.0","atr_1.5","atr_2.0")
def clean(m):
 return {k:(float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v) for k,v in m.items()}
def gates(m):
 return m["win_rate_pct"]>40 and (m["profit_factor"] or 0)>1 and m["tp_per_day"]>=3 and m["trades_per_day"]>=10 and m["max_consecutive_losses"]<=12 and not m["capital_exhausted"] and all(v>=300 for v in m["monthly_trade_counts"].values())
def score(m):
 # Preserve frequency first; never promote a sparse candidate as an improvement.
 return (m["trades_per_day"]>=7.5, m["tp_per_day"]>=2, m["max_consecutive_losses"]<=12,
         m["win_rate_pct"], m["profit_factor"] or 0, m["tp_per_day"],m["trades_per_day"],m["net_pnl"])
async def main():
 started=time.monotonic(); sm.CONTRACT=normalize_contract(CFG)
 df,manifest=await get_data(); splits=split_reference_period(df)
 report={"schema":"structural_entry_filter_stop_sweep_v1","data_manifest":manifest,"contract":CFG,
 "oos_evaluated":False,"oos_used_for_selection":False,
 "ambiguity_note":"1m OHLC; same-bar SL/TP resolves as SL; research only, not final validation.","splits":{}}
 allrows={}
 for sn in ("train","development"):
  part=splits[sn]; entries,atr=sm.signals(part); filters=sm.confirmation_filters(part); rows=[]
  # Reproduce the 11-loss baseline exactly on each split.
  a,b=entries["structure_break_after_pullback"]; fa,fb=filters["ema21_slope5_atr"]
  bm,_=sm.run_combined(part,a&fa.fillna(False),b&fb.fillna(False),atr,"swing_5",loss_cooldown_bars=1,cooldown_trigger_losses=1)
  baseline=clean(bm)
  for en,(rawl,raws) in entries.items():
   for fn,(fl,fs) in filters.items():
    longs=pd.Series(rawl,index=part.index).fillna(False)&fl.reindex(part.index).fillna(False)
    shorts=pd.Series(raws,index=part.index).fillna(False)&fs.reindex(part.index).fillna(False)
    for stop in STOPS:
     m,ledger=sm.run_combined(part,longs,shorts,atr,stop,loss_cooldown_bars=0,cooldown_trigger_losses=1)
     m=clean(m); row={"entry":en,"filter":fn,"stop":stop,"metrics":m,
       "ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()}
     rows.append(row); allrows[(sn,en,fn,stop)]=row
  rows.sort(key=lambda r:score(r["metrics"]),reverse=True)
  passed=[r for r in rows if gates(r["metrics"])]
  report["splits"][sn]={"tested":len(rows),"baseline_11_loss_metrics":baseline,
    "pass_all_gates":len(passed),"top_by_frequency_preserving_score":rows[:15],
    "top_with_streak_le_12_and_trades_per_day_ge_7_5":[r for r in rows if r["metrics"]["max_consecutive_losses"]<=12 and r["metrics"]["trades_per_day"]>=7.5][:15]}
  print("STRUCTURAL_SPLIT "+json.dumps({"split":sn,"tested":len(rows),"baseline":baseline,"pass_all_gates":len(passed),"top":rows[:3]},sort_keys=True),flush=True)
 cross=[]
 for en in entries:
  for fn in filters:
   for stop in STOPS:
    tr=allrows[("train",en,fn,stop)]; dv=allrows[("development",en,fn,stop)]
    cross.append({"entry":en,"filter":fn,"stop":stop,"train":tr["metrics"],"development":dv["metrics"],
      "train_pass":gates(tr["metrics"]),"development_pass":gates(dv["metrics"]),
      "train_ledger_sha256":tr["ledger_sha256"],"development_ledger_sha256":dv["ledger_sha256"]})
 cross.sort(key=lambda x:(x["train_pass"] and x["development_pass"],
   min(x["train"]["trades_per_day"],x["development"]["trades_per_day"])>=7.5,
   min(x["train"]["max_consecutive_losses"],x["development"]["max_consecutive_losses"])<=12,
   min(x["train"]["win_rate_pct"],x["development"]["win_rate_pct"]),
   min(x["train"]["profit_factor"] or 0,x["development"]["profit_factor"] or 0),
   min(x["train"]["tp_per_day"],x["development"]["tp_per_day"])),reverse=True)
 report["cross_split"]={"tested_matched_configurations":len(cross),
 "both_splits_pass_all_gates":sum(x["train_pass"] and x["development_pass"] for x in cross),"top":cross[:30]}
 report["elapsed_seconds"]=round(time.monotonic()-started,2)
 p=OUT/"structural_entry_filter_stop_sweep.json"; p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
 print("STRUCTURAL_FINAL "+json.dumps({"report":str(p),"elapsed_seconds":report["elapsed_seconds"],"both_pass":report["cross_split"]["both_splits_pass_all_gates"],"top":cross[:10]},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
