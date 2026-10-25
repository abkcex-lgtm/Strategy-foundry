# Verification rerun requested 2026-10-25: repeat the exact deterministic Train/Development experiment without evaluating OOS.
"""Causal ML meta-label experiment; never loads/evaluates OOS."""
import asyncio,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import research.ema_9_21_123 as sm
from research.run_factory import get_data,split_reference_period
from app.engine import normalize_contract
OUT=Path("research-results");OUT.mkdir(exist_ok=True)
# Expand the ML meta-label search across causal price-action/structure families already implemented in the research engine.
# Keep the execution contract fixed and use chronological, purged TRAIN validation; DEVELOPMENT remains a separate confirmation split.
FAMILIES=("breakout_123","pullback_reclaim","higher_low_breakout","cross_pullback_breakout5","cross_pullback_breakout10","cross_pullback_breakout20","trend_pullback_reversal","trend_pullback_cross5","trend_pullback_cross10","trend_pullback_cross20","structure_break_after_pullback","structure_break_pb3_br5","structure_break_pb3_br10","structure_break_pb3_br15","structure_break_pb5_br10","structure_break_pb5_br15","structure_break_pb5_br20","structure_break_pb8_br10","structure_break_pb8_br15","structure_break_pb8_br20","structure_break_pb10_br15","structure_break_pb10_br20","range_break_retest","swing_structure_break","squeeze_breakout","swing_failure_reclaim","atr_expansion_breakout","donchian_break_10","donchian_break_20")
FEATS=("r1","r3","r5","r15","r30","range_atr","body_atr","uw_atr","lw_atr","atr_pct","atr_ratio","volz20","volz60","rsi","rsi_slope","ema_gap","ema_slope","high_dist","low_dist","hour_sin","hour_cos","weekday")
FAMILY_FEATS=tuple("family__"+x for x in FAMILIES)
MODEL_FEATS=FEATS+FAMILY_FEATS+("side_sign",)
def model_matrix(frame):
 x=frame[list(FEATS)].copy()
 for fam in FAMILIES:x["family__"+fam]=(frame["family"].to_numpy()==fam).astype(float)
 x["side_sign"]=frame["side"].map({"LONG":1.0,"SHORT":-1.0}).astype(float)
 return x[list(MODEL_FEATS)]
def features(d,a):
 c=d.close.astype(float);o=d.open.astype(float);h=d.high.astype(float);l=d.low.astype(float);v=d.volume.astype(float);a=a.astype(float).replace(0,np.nan)
 tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
 de=c.diff();g=de.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean();z=(-de.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean();r=100-100/(1+g/z.replace(0,np.nan))
 e=c.ewm(span=21,adjust=False,min_periods=21).mean();dt=pd.to_datetime(d.open_time,unit="ms",utc=True);mi=dt.dt.hour*60+dt.dt.minute
 f=pd.DataFrame(index=d.index)
 for n in (1,3,5,15,30):f["r"+str(n)]=c.pct_change(n)*100
 f["range_atr"]=(h-l)/a;f["body_atr"]=(c-o)/a;f["uw_atr"]=(h-np.maximum(o,c))/a;f["lw_atr"]=(np.minimum(o,c)-l)/a;f["atr_pct"]=a/c*100;f["atr_ratio"]=a/a.rolling(60,min_periods=30).mean()
 for n in (20,60):f["volz"+str(n)]=(v-v.rolling(n,min_periods=max(15,n//2)).mean())/v.rolling(n,min_periods=max(15,n//2)).std().replace(0,np.nan)
 f["rsi"]=r;f["rsi_slope"]=r-r.shift(3);f["ema_gap"]=(c-e)/a;f["ema_slope"]=(e-e.shift(5))/a;f["high_dist"]=(c-h.rolling(20,min_periods=20).max())/a;f["low_dist"]=(c-l.rolling(20,min_periods=20).min())/a;f["hour_sin"]=np.sin(2*np.pi*mi/1440);f["hour_cos"]=np.cos(2*np.pi*mi/1440);f["weekday"]=dt.dt.dayofweek
 return f.replace([np.inf,-np.inf],np.nan)
def get_signals(d):
 sigs,a=sm.signals(d);c=d.close.astype(float);h=d.high.astype(float);l=d.low.astype(float)
 for n in (10,20):
  hi=h.shift(1).rolling(n,min_periods=n).max();lo=l.shift(1).rolling(n,min_periods=n).min()
  sigs["donchian_break_"+str(n)]=((c>hi)&(c.shift(1)<=hi.shift(1)),(c<lo)&(c.shift(1)>=lo.shift(1)))
 return sigs,a
def events(d,a):
 sigs,_=get_signals(d);f=features(d,a);rows=[]
 for fam in FAMILIES:
  if fam not in sigs:continue
  for side,sig in (("LONG",sigs[fam][0]),("SHORT",sigs[fam][1])):
   for i in np.flatnonzero(pd.Series(sig,index=d.index).fillna(False).to_numpy()):
    if i<80 or i+1>=len(d) or not np.isfinite(a.iloc[i]) or a.iloc[i]<=0:continue
    en=float(d.open.iloc[i+1]);st=en-float(a.iloc[i])*1.5 if side=="LONG" else en+float(a.iloc[i])*1.5
    try:q,tp,er,xr=sm._size_and_target(en,st,side,10.,sm.CONTRACT)
    except ValueError:continue
    lab=None
    for j in range(i+1,len(d)):
     try:out=sm._resolve_bar_exit(side,float(d.open.iloc[j]),float(d.high.iloc[j]),float(d.low.iloc[j]),st,tp)
     except Exception:out=("SL",st)
     if out is not None:lab=int(out[0]=="TP");break
    if lab is not None:
     row=f.iloc[i].to_dict();row.update(idx=int(i),label_exit_idx=int(j),family=fam,side=side,label=lab);rows.append(row)
 return pd.DataFrame(rows)
MODEL=None
def backtest(d,a,fam,threshold,start_idx=0):
 sigs,_=get_signals(d);f=features(d,a);L=np.zeros(len(d),bool);S=np.zeros(len(d),bool)
 for side,raw,out in (("LONG",sigs[fam][0],L),("SHORT",sigs[fam][1],S)):
  ix=np.flatnonzero(pd.Series(raw,index=d.index).fillna(False).to_numpy());ix=ix[(ix>=max(80,int(start_idx)))&(ix<len(d)-1)]
  if len(ix):
   xf=f.iloc[ix].copy();xf["family"]=fam;xf["side"]=side
   out[ix]=MODEL.predict_proba(model_matrix(xf))[:,1]>=(threshold.get(fam,threshold.get("__default__",0.25)) if isinstance(threshold,dict) else threshold)
 return sm.run_combined(d,pd.Series(L,index=d.index),pd.Series(S,index=d.index),a,"atr_1.5",loss_cooldown_bars=1,cooldown_trigger_losses=1)
async def main():
 start=time.monotonic()
 sm.CONTRACT=normalize_contract({"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":.04,"slippage_pct_per_side":.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False})
 df,manifest=await get_data();sp=split_reference_period(df);parts={};rep={"schema":"causal_ml_meta_label_v2_expanded_families","data_manifest":manifest,"oos_evaluated":False,"method":"Expanded 29-family HistGradientBoosting meta-label search across causal EMA pullbacks, price action, swing structure, squeeze/ATR expansion, failed-breakout reclaim and Donchian breakouts; past-only OHLCV/regime/time features plus family and side; fixed $10 net risk / $35 net target; same-bar SL priority; chronological purged 60/20/20 TRAIN fit/tune/untouched holdout, then separate July Development confirmation; OOS not loaded/evaluated.","features":list(MODEL_FEATS),"splits":{}}
 for name in ("train","development"):
  d=sp[name];_,a=get_signals(d);ev=events(d,a);parts[name]=(d,a)
  rep["splits"][name]={"rows":len(d),"events":len(ev),"label_win_rate_pct":float(ev.label.mean()*100) if len(ev) else 0}
  if name=="train":tr=ev
 if len(tr)<100:raise RuntimeError("Insufficient supervised training events: "+str(len(tr)))
 tr=tr.sort_values("idx").reset_index(drop=True)
 # Strict chronological three-way split: fit on first 60%, choose threshold on next 20%,
 # and report TRAIN strategy performance only on the final untouched 20% tail.
 n_events=len(tr); fit_end=int(n_events*.60); tune_end=int(n_events*.80)
 fit_raw=tr.iloc[:fit_end]; tune_raw=tr.iloc[fit_end:tune_end]; holdout=tr.iloc[tune_end:]
 # Purge boundary-crossing labels to prevent future-outcome leakage across time splits.
 tune_boundary=int(tune_raw.idx.min()) if len(tune_raw) else 10**18
 holdout_boundary=int(holdout.idx.min()) if len(holdout) else 10**18
 fit=fit_raw[fit_raw.label_exit_idx<tune_boundary].copy()
 tune=tune_raw[tune_raw.label_exit_idx<holdout_boundary].copy()
 global MODEL
 MODEL=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2,random_state=42))
 MODEL.fit(model_matrix(fit),fit.label.astype(int));p=MODEL.predict_proba(model_matrix(tune))[:,1];opts=[]
 for th in (.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75):
  keep=p>=th;n=int(keep.sum());w=int(tune.label.to_numpy()[keep].sum());opts.append({"threshold":th,"events":n,"win_rate_pct":100*w/n if n else 0,"net_expectancy_per_trade":(35*w-10*(n-w))/n if n else None})
 viable=[x for x in opts if x["events"]>=50]
 chosen=max(viable,key=lambda x:(x["net_expectancy_per_trade"],x["events"])) if viable else max(opts,key=lambda x:x["events"])
 # Tune thresholds per family on the chronological tuning segment; sparse families fall back to the global threshold.
 family_thresholds={"__default__":chosen["threshold"]}
 family_threshold_diagnostics={}
 p_by_event=pd.Series(p,index=tune.index)
 for fam in FAMILIES:
  mask=(tune["family"].to_numpy()==fam)
  fam_labels=tune.label.to_numpy()[mask]
  fam_p=p[mask]
  candidates=[]
  for th in (.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75):
   keep=fam_p>=th;n=int(keep.sum());w=int(fam_labels[keep].sum())
   if n>=50:
    candidates.append({"threshold":th,"events":n,"win_rate_pct":100*w/n,"net_expectancy_per_trade":(35*w-10*(n-w))/n})
  if candidates:
   best=max(candidates,key=lambda x:(x["net_expectancy_per_trade"],x["events"]))
   family_thresholds[fam]=best["threshold"];family_threshold_diagnostics[fam]=best
  else:
   family_thresholds[fam]=chosen["threshold"];family_threshold_diagnostics[fam]={"fallback":True,"available_events":int(mask.sum()),"threshold":chosen["threshold"]}
 holdout_start_idx=int(holdout.idx.min()) if len(holdout) else len(parts["train"][0])
 rep["train_model"]={"fit_events_raw":len(fit_raw),"fit_events_after_purge":len(fit),"threshold_tuning_events_raw":len(tune_raw),"threshold_tuning_events_after_purge":len(tune),"final_train_holdout_events":len(holdout),"threshold_candidates":opts,"chosen_threshold":chosen["threshold"],"chosen_tuning_metrics":chosen,"family_thresholds":family_thresholds,"family_threshold_diagnostics":family_threshold_diagnostics,"selection_protocol":"chronological purged 60% fit / 20% threshold tuning / 20% untouched TRAIN holdout; global fallback plus family-specific thresholds selected only on tuning events; TRAIN backtest restricted to final holdout tail"}
 for name in ("train","development"):
  d,a=parts[name];res=[]
  start_idx=holdout_start_idx if name=="train" else 0
  for fam in FAMILIES:
   if fam not in get_signals(d)[0]:continue
   m,led=backtest(d,a,fam,family_thresholds,start_idx=start_idx);res.append({"family":fam,"metrics":m,"trades":len(led)})
  res.sort(key=lambda x:(x["metrics"]["passes_all_user_targets"],x["metrics"]["tp_per_day"],x["metrics"]["profit_factor"] or 0),reverse=True)
  rep["splits"][name].update(passing_families=sum(x["metrics"]["passes_all_user_targets"] for x in res),backtests=res)
 rep["elapsed_seconds"]=round(time.monotonic()-start,2);p=OUT/"causal_ml_meta_label_report.json";p.write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False))
 print("CAUSAL_ML_FINAL "+json.dumps({"report":str(p),"elapsed_seconds":rep["elapsed_seconds"],"model":rep["train_model"],"splits":{k:{"events":v["events"],"passing":v.get("passing_families",0),"top":v.get("backtests",[])[:2]} for k,v in rep["splits"].items()}},sort_keys=True),flush=True)
if __name__=="__main__":asyncio.run(main())
