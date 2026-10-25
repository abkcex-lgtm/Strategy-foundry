"""Focused three-pass Donchian-20 audit on TRAIN holdout and DEVELOPMENT only.
Never loads or evaluates OOS. Passes: raw breakout, frozen ML threshold 0.25,
then threshold tuned only on a chronological TRAIN tuning segment. Contract is fixed.
"""
import asyncio, hashlib, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research.causal_ml_meta_label_search as ml
import research.ema_9_21_123 as sm
from research.run_factory import get_data, split_reference_period
from app.engine import normalize_contract

OUT=ROOT/"research-results"; OUT.mkdir(exist_ok=True)
FAMILY="donchian_break_20"
CONTRACT=normalize_contract({
 "symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,
 "compound":False,"reward_risk":3.5,"target_mode":"net_rr",
 "commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
 "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only",
 "break_even_enabled":False,"trailing_stop_enabled":False,
 "partial_exit_enabled":False,"time_exit_enabled":False
})
def num(x):
 if isinstance(x,dict): return {str(k):num(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [num(v) for v in x]
 if isinstance(x,(np.integer,)): return int(x)
 if isinstance(x,(np.floating,)): return float(x)
 if isinstance(x,(np.bool_,)): return bool(x)
 return x
def raw_backtest(d,a,start_idx=0):
 sigs,_=ml.get_signals(d)
 raw_l,raw_s=sigs[FAMILY]
 L=pd.Series(raw_l,index=d.index).fillna(False); S=pd.Series(raw_s,index=d.index).fillna(False)
 L &= (L.index.to_numpy()>=max(80,int(start_idx))) & (L.index.to_numpy()<len(d)-1)
 S &= (S.index.to_numpy()>=max(80,int(start_idx))) & (S.index.to_numpy()<len(d)-1)
 return sm.run_combined(d,L,S,a,"atr_1.5",loss_cooldown_bars=1,cooldown_trigger_losses=1)
def gates(m,monthly_gate=True):
 monthly=m.get("monthly_trade_counts") or {}
 return bool(m.get("win_rate_pct",0)>40 and (m.get("profit_factor") or 0)>1
  and m.get("tp_per_day",m.get("avg_net_tp_per_day",0))>=3
  and m.get("trades_per_day",0)>=10 and m.get("max_consecutive_losses",999)<=12
  and (not monthly_gate or (monthly and min(int(v or 0) for v in monthly.values())>=300)))
async def main():
 started=time.monotonic()
 sm.CONTRACT=CONTRACT
 df,manifest=await get_data()
 splits=split_reference_period(df)
 train=splits["train"].reset_index(drop=True)
 dev=splits["development"].reset_index(drop=True)
 if (len(train),len(dev),len(splits["oos"]))!=(131040,44640,87840):
  raise RuntimeError("Unexpected dataset split sizes; aborting.")
 sig,atr=ml.get_signals(train)
 ev=ml.events(train,atr)
 ev=ev.loc[ev.family==FAMILY].sort_values("idx").reset_index(drop=True)
 if len(ev)<100: raise RuntimeError(f"Insufficient Donchian20 TRAIN events: {len(ev)}")
 n=len(ev); fit_end=int(n*.60); tune_end=int(n*.80)
 fit_raw=ev.iloc[:fit_end].copy(); tune_raw=ev.iloc[fit_end:tune_end].copy(); holdout=ev.iloc[tune_end:].copy()
 tune_boundary=int(tune_raw.idx.min()); holdout_boundary=int(holdout.idx.min())
 fit=fit_raw.loc[fit_raw.label_exit_idx<tune_boundary].copy()
 tune=tune_raw.loc[tune_raw.label_exit_idx<holdout_boundary].copy()
 if len(fit)<50 or len(tune)<20 or len(holdout)<20:
  raise RuntimeError(f"Insufficient purged chronological events: fit={len(fit)}, tune={len(tune)}, holdout={len(holdout)}")
 model=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),
  HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2,random_state=42))
 model.fit(ml.model_matrix(fit),fit.label.astype(int))
 p=model.predict_proba(ml.model_matrix(tune))[:,1]
 opts=[]
 for th in (.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75):
  keep=p>=th; count=int(keep.sum()); wins=int(tune.label.to_numpy()[keep].sum())
  opts.append({"threshold":th,"tuning_events_selected":count,
   "label_win_rate_pct":100*wins/count if count else 0,
   "label_net_expectancy_usd":(35*wins-10*(count-wins))/count if count else None})
 viable=[x for x in opts if x["tuning_events_selected"]>=20]
 tuned=max(viable,key=lambda x:(x["label_net_expectancy_usd"],x["tuning_events_selected"]))["threshold"] if viable else .25
 ml.MODEL=model
 holdout_start=int(holdout.idx.min())
 results={}
 for split_name,d in (("train_holdout",train),("development",dev)):
  _,a=ml.get_signals(d)
  start=holdout_start if split_name=="train_holdout" else 0
  passes={}
  for label,method in (("pass1_raw_donchian20","raw"),("pass2_frozen_ml_025","025"),("pass3_train_tuned_ml","tuned")):
   if method=="raw": m,ledger=raw_backtest(d,a,start)
   else: m,ledger=ml.backtest(d,a,FAMILY,{"__default__":0.25 if method=="025" else tuned},start_idx=start)
   passes[label]={"metrics":num(m),"trade_count":len(ledger),
    "ledger_sha256":hashlib.sha256(json.dumps(num(ledger),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
    "passes_all_gates":gates(m,monthly_gate=(split_name=="development"))}
  results[split_name]=passes
 report={"schema":"donchian20_three_pass_train_dev_audit_v1","status":"completed",
  "data_manifest":manifest,"oos_loaded_or_evaluated":False,
  "contract":{"risk_usd_net":10,"target_usd_net":35,"commission_pct_per_side":0.04,
   "slippage_pct_per_side":0.02,"initial_balance":1000,"compounding":False,
   "max_positions":1,"entry":"next candle open","exit":"SL/TP only",
   "stop":"ATR(14)*1.5","same_bar_policy":"SL priority","loss_cooldown_bars":1},
  "event_counts":{"train_donchian20_events":n,"fit_raw":len(fit_raw),"fit_after_purge":len(fit),
   "tuning_raw":len(tune_raw),"tuning_after_purge":len(tune),"holdout":len(holdout),
   "train_holdout_start_idx":holdout_start},
  "threshold_tuning":{"selected_threshold":tuned,"candidates":opts,
   "selection_split":"chronological middle 20% of TRAIN Donchian20 events only"},
  "three_pass_results":results,
  "all_passes_gates_on_both_splits":sum(all(v["passes_all_gates"] for v in results[s].values()) for s in results),
  "elapsed_seconds":round(time.monotonic()-started,2)}
 path=OUT/"donchian20_three_pass_train_dev_report.json"
 path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
 print("DONCHIAN20_THREE_PASS_FINAL "+json.dumps({"report":str(path),"elapsed_seconds":report["elapsed_seconds"],
  "events":report["event_counts"],"tuned_threshold":tuned,"results":results,
  "all_passes_gates_on_both_splits":report["all_passes_gates_on_both_splits"]},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
