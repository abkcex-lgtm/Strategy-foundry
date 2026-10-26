"""Three-pass, train-only evolutionary strategy search for ETHUSDT. No OOS selection."""
import asyncio, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from app.engine import search_candidates, backtest, normalize_contract, prepare_backtest_data
from research.run_factory import get_data, split_reference_period, frame_hash

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CFG=normalize_contract({"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
 "reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
 "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only",
 "min_win_rate_pct":40,"min_profit_factor":1,"min_tp_per_day":3,"max_consecutive_losses":12,
 "use_min_win_rate":True,"use_min_profit_factor":True,"use_min_tp_per_day":True,"use_max_consecutive_losses":True,
 "min_trades_per_day":10,"min_trades_per_month":300,"break_even_enabled":False,"trailing_stop_enabled":False,
 "partial_exit_enabled":False,"time_exit_enabled":False})
def val(x):
 if isinstance(x,dict): return {str(k):val(v) for k,v in x.items() if not str(k).startswith("_")}
 if isinstance(x,(list,tuple)): return [val(y) for y in x]
 if isinstance(x,(np.integer,)): return int(x)
 if isinstance(x,(np.floating,)): return float(x)
 if isinstance(x,(np.bool_,)): return bool(x)
 return x
def tpday(m):
 return float(m.get("avg_net_tp_per_day",m.get("tp_per_day",0)) or 0)
def gates(m):
 monthly=m.get("monthly_trade_counts") or {}
 return bool(m.get("win_rate_pct",0)>40 and (m.get("profit_factor") or 0)>1 and tpday(m)>=3
  and m.get("trades_per_day",0)>=10 and m.get("max_consecutive_losses",999)<=12
  and monthly and min(int(v or 0) for v in monthly.values())>=300
  and m.get("net_pnl",0)>0 and not m.get("capital_exhausted",False) and m.get("final_balance",0)>10)
def score(m):
 # Lexicographic: full hard-gate pass first; then minimize the largest shortfall.
 gaps=[
  max(0,40-float(m.get("win_rate_pct",0)))/40,
  max(0,1-float(m.get("profit_factor") or 0)),
  max(0,3-tpday(m))/3,
  max(0,10-float(m.get("trades_per_day",0)))/10,
  max(0,int(m.get("max_consecutive_losses",999))-12)/12,
  1 if m.get("capital_exhausted",False) else 0,
 ]
 return (gates(m),-max(gaps),-sum(gaps),float(m.get("net_pnl",0)),int(m.get("trades",0)))
def brief(row):
 return {"strategy":row.get("strategy"),"metrics":row.get("metrics"),"failed_conditions":row.get("failed_conditions"),
         "criterion_distances":row.get("criterion_distances"),"run_hash_sha256":row.get("run_hash_sha256")}
async def main():
 started=time.monotonic()
 df,manifest=await get_data(); splits=split_reference_period(df)
 train=splits["train"]; dev=splits["development"]
 trainctx=prepare_backtest_data(train); devctx=prepare_backtest_data(dev)
 if (len(train),len(dev),len(splits["oos"]))!=(131040,44640,87840): raise RuntimeError("Split size mismatch")
 report={"schema":"eth_three_pass_evolutionary_search_v1","contract":CFG,"data_manifest":manifest,
  "split_hashes":{"train":frame_hash(train),"development":frame_hash(dev)},
  "oos_evaluated":False,"oos_used_for_selection":False,
  "execution_note":"Fixed net risk $10 and net 3.5R target; costs 0.04% commission and 0.02% slippage per side; one position; next-bar-open; SL/TP only. Same-candle ambiguity is conservatively resolved by the engine; 1m-only is not final 1s ambiguity validation.",
  "iteration_plan":{"count":3,"algorithms":["hybrid_genetic","stratified_random","surrogate_genetic"],"selection_data":"TRAIN only","confirmation_data":"DEVELOPMENT only","oos_policy":"do not load or evaluate OOS"},
  "loops":[]}
 prior=[]; excluded=set(); all_dev=[]
 for ix,algorithm in enumerate(("hybrid_genetic","stratified_random","surrogate_genetic"),1):
  loop_start=time.monotonic()
  # Reuse the same validated TRAIN dataset and fixed contract. Fresh random seed each pass.
  res=search_candidates(train,5000,8200+ix*311,CFG,algorithm,
    excluded_genome_hashes=excluded,prior_candidates=prior,prepared_context=trainctx)
  excluded.update(res.get("evaluated_genome_hashes",[]))
  selected=res.get("results",[]) or []
  # Dev is used as a confirmation screen only; never used to propose candidates.
  dev_rows=[]
  for row in selected:
   try:
    spec=row.get("strategy")
    bt=backtest(dev,spec,CFG,_prepared_context=devctx)
    d={"strategy":bt.get("strategy"),"train_metrics":row.get("metrics"),"development_metrics":bt.get("metrics"),
       "train_pass":gates(row.get("metrics",{})),"development_pass":gates(bt.get("metrics",{})),
       "train_run_hash":row.get("run_hash_sha256"),"development_run_hash":bt.get("run_hash_sha256")}
    d["both_splits_pass"]=d["train_pass"] and d["development_pass"]
    dev_rows.append(d); all_dev.append(d)
   except Exception as e:
    dev_rows.append({"strategy":row.get("strategy"),"validation_error":type(e).__name__+":"+str(e)})
  report["loops"].append({"loop":ix,"algorithm":algorithm,"requested_candidates":5000,
   "evaluated_candidates":res.get("evaluated_candidates"),"failed_candidates":res.get("failed_candidates"),
   "duplicate_genomes_skipped":res.get("duplicate_genomes_skipped"),"independent_audit_failures":res.get("independent_audit_failures"),
   "family_evaluated_counts":res.get("family_evaluated_counts"),"all_passing_on_train":res.get("all_passing"),
   "elapsed_seconds":round(time.monotonic()-loop_start,2),"train_shortlist":[brief(x) for x in selected],
   "development_retests":dev_rows})
  # Use previous top train-ranked candidates to seed the next evolutionary loop.
  prior=selected[:]
  print("ETH_LOOP "+json.dumps({"loop":ix,"algorithm":algorithm,"evaluated":res.get("evaluated_candidates"),
   "dev_retested":len(dev_rows),"best_train":(selected[0].get("metrics") if selected else None),
   "best_dev":(dev_rows[0].get("development_metrics") if dev_rows and "development_metrics" in dev_rows[0] else None)},default=str),flush=True)
 if len(report["loops"]) != 3: raise RuntimeError("Required three research loops did not complete")
 # Best by strict simultaneous gates, then balanced train/development performance; do not let tiny samples win just on PF.
 def joint(row):
  if "development_metrics" not in row: return (False,-999,-999,-999,-999)
  a,b=row.get("train_metrics",{}),row["development_metrics"]
  return (row.get("both_splits_pass",False),min(a.get("win_rate_pct",0),b.get("win_rate_pct",0)),
   min(a.get("tp_per_day",0),b.get("tp_per_day",0)),min(a.get("trades_per_day",0),b.get("trades_per_day",0)),
   min(a.get("net_pnl",0),b.get("net_pnl",0)))
 all_dev.sort(key=joint,reverse=True)
 report["final_summary"]={"tested_dev_retests":len(all_dev),"both_splits_pass_all_gates":sum(x.get("both_splits_pass",False) for x in all_dev),
  "best_candidate":all_dev[0] if all_dev else None,"top_candidates":all_dev[:10],
  "interpretation":"No candidate is declared successful unless all hard requirements pass on both Train and Development. OOS remains untouched."}
 report["elapsed_seconds"]=round(time.monotonic()-started,2)
 path=OUT/"eth_three_pass_evolutionary_search_report.json"
 path.write_text(json.dumps(val(report),indent=2,sort_keys=True,allow_nan=False))
 print("ETH_FINAL "+json.dumps({"report":str(path),"elapsed_seconds":report["elapsed_seconds"],
  "both_split_pass_count":report["final_summary"]["both_splits_pass_all_gates"],
  "best_candidate":report["final_summary"]["best_candidate"]},default=str),flush=True)
if __name__=="__main__": asyncio.run(main())
