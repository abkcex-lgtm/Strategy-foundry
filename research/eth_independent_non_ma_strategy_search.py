"""Independent ETH-only non-MA strategy search; research-only, OOS untouched."""
import asyncio, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
     "reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
     "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only",
     "break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
ENTRIES=("donchian_break_5","donchian_break_10","donchian_break_20","rsi_reclaim","rsi_midline_cross",
         "bollinger_reentry","bollinger_breakout","stochastic_reversal","swing_failure_reclaim","atr_expansion_breakout")
FILTERS=("baseline","adx18_directional","rsi_momentum","volume_above_sma20","strong_candle_close",
         "atr_midrange","avoid_extended_entry","body_plus_adx22")
STOPS=("signal_candle","swing_3","atr_0.75","atr_1.0","atr_1.5")
COOLDOWNS=((0,1),(1,1),(2,1),(3,1),(1,2),(3,2))
def clean(m):
 return {k:(float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v) for k,v in m.items()}
def make_signals(df):
 c=df.close.astype(float); o=df.open.astype(float); h=df.high.astype(float); lo=df.low.astype(float); v=df.volume.astype(float)
 tr=pd.concat([h-lo,(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1)
 atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
 delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
 loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
 rs=gain/loss.replace(0,np.nan); rsi=100-100/(1+rs)
 lo_band=(c.rolling(20,min_periods=20).mean()-2*c.rolling(20,min_periods=20).std())
 hi_band=(c.rolling(20,min_periods=20).mean()+2*c.rolling(20,min_periods=20).std())
 # Stochastic oscillator, causal rolling extrema.
 ll=lo.rolling(14,min_periods=14).min(); hh=h.rolling(14,min_periods=14).max()
 k=100*(c-ll)/(hh-ll).replace(0,np.nan); d=k.rolling(3,min_periods=3).mean()
 sig={}
 for n in (5,10,20):
  ph=h.shift(1).rolling(n,min_periods=n).max(); pl=lo.shift(1).rolling(n,min_periods=n).min()
  sig[f"donchian_break_{n}"]=(c>ph,c<pl)
 sig["rsi_reclaim"]=((rsi.shift(1)<30)&(rsi>=30),(rsi.shift(1)>70)&(rsi<=70))
 sig["rsi_midline_cross"]=((rsi.shift(1)<50)&(rsi>=50),(rsi.shift(1)>50)&(rsi<=50))
 sig["bollinger_reentry"]=((c.shift(1)<lo_band.shift(1))&(c>=lo_band),(c.shift(1)>hi_band.shift(1))&(c<=hi_band))
 sig["bollinger_breakout"]=(c>hi_band.shift(1),c<lo_band.shift(1))
 sig["stochastic_reversal"]=((k.shift(1)<20)&(k>=20)&(k>d),(k.shift(1)>80)&(k<=80)&(k<d))
 extras,atr2=sm.signals(df)
 sig["swing_failure_reclaim"]=extras["swing_failure_reclaim"]
 sig["atr_expansion_breakout"]=extras["atr_expansion_breakout"]
 return sig,atr
async def main():
 started=time.monotonic(); sm.CONTRACT=normalize_contract(CFG)
 df,manifest=await get_data(); splits=split_reference_period(df); report={"schema":"eth_independent_non_ma_strategy_sweep_v1","data_manifest":manifest,
 "contract":CFG,"oos_evaluated":False,"ambiguity_note":"1m OHLC only; same-bar SL/TP ambiguity resolves conservatively to SL; exploratory, not final validation.",
 "design":"Donchian breakouts, RSI reclaim/midline, Bollinger re-entry/breakout, stochastic reversal, swing-failure reclaim, ATR expansion. No moving-average entry rule in the custom families; optional legacy directional filters are separately labeled.",
 "splits":{},"elapsed_seconds":None}
 allrows={}
 for split_name in ("train","development"):
  part=splits[split_name]; signals,atr=make_signals(part); filters=sm.confirmation_filters(part); rows=[]
  for ent in ENTRIES:
   rl,rs=signals[ent]
   for fn in FILTERS:
    lf,sf=filters[fn]
    ls=pd.Series(rl,index=part.index).fillna(False)&lf.fillna(False)
    ss=pd.Series(rs,index=part.index).fillna(False)&sf.fillna(False)
    for stop in STOPS:
     for cb,ct in COOLDOWNS:
      m,ledger=sm.run_combined(part,ls,ss,atr,stop,loss_cooldown_bars=cb,cooldown_trigger_losses=ct)
      m=clean(m); key=(ent,fn,stop,cb,ct)
      row={"entry":ent,"filter":fn,"stop":stop,"cooldown_bars":cb,"cooldown_trigger_losses":ct,"metrics":m,
           "ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()}
      rows.append(row); allrows[(split_name,)+key]=row
  # exact 11-loss baseline reference
  old,atr_old=sm.signals(part); oldf=sm.confirmation_filters(part)
  a,b=old["structure_break_after_pullback"]; fa,fb=oldf["ema21_slope5_atr"]
  bm,_=sm.run_combined(part,pd.Series(a&fa.fillna(False),index=part.index),pd.Series(b&fb.fillna(False),index=part.index),atr_old,"swing_5",loss_cooldown_bars=1,cooldown_trigger_losses=1)
  bm=clean(bm)
  eligible=[r for r in rows if r["metrics"]["max_consecutive_losses"]<=12 and r["metrics"]["tp_per_day"]>=3]
  eligible.sort(key=lambda r:(r["metrics"]["tp_per_day"],r["metrics"]["net_pnl"],r["metrics"]["profit_factor"] or 0),reverse=True)
  streak=[r for r in rows if r["metrics"]["max_consecutive_losses"]<=12]
  streak.sort(key=lambda r:(r["metrics"]["tp_per_day"],r["metrics"]["net_pnl"]),reverse=True)
  report["splits"][split_name]={"tested_candidates":len(rows),"baseline_11_loss_metrics":bm,"pass_both_conditions_count":len(eligible),
    "top_pass_both_conditions":eligible[:15],"top_streak_le_12":streak[:15]}
  print("NON_MA_SPLIT "+json.dumps({"split":split_name,"tested":len(rows),"baseline":bm,"passes":len(eligible),"top":eligible[:3]},sort_keys=True),flush=True)
 cross=[]
 for key in [(e,f,s,cb,ct) for e in ENTRIES for f in FILTERS for s in STOPS for cb,ct in COOLDOWNS]:
  tr=allrows[("train",)+key]; dv=allrows[("development",)+key]; tm=tr["metrics"]; dm=dv["metrics"]
  if tm["max_consecutive_losses"]<=12 and dm["max_consecutive_losses"]<=12 and tm["tp_per_day"]>=3 and dm["tp_per_day"]>=3:
   cross.append({"entry":key[0],"filter":key[1],"stop":key[2],"cooldown_bars":key[3],"cooldown_trigger_losses":key[4],
    "train":tm,"development":dm,"train_ledger_sha256":tr["ledger_sha256"],"development_ledger_sha256":dv["ledger_sha256"]})
 cross.sort(key=lambda x:(min(x["train"]["tp_per_day"],x["development"]["tp_per_day"]),min(x["train"]["net_pnl"],x["development"]["net_pnl"])),reverse=True)
 report["cross_split_candidates_meeting_both"]={"count":len(cross),"top":cross[:30]}
 report["elapsed_seconds"]=round(time.monotonic()-started,2)
 p=OUT/"eth_independent_non_ma_strategy_sweep_report.json"; p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
 print("NON_MA_FINAL "+json.dumps({"report":str(p),"elapsed_seconds":report["elapsed_seconds"],"cross_split_passes":len(cross),"top":cross[:10]},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
