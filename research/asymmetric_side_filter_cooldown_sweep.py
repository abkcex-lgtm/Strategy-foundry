"""Asymmetric long/short confirmation and cooldown search; preserve contract and leave OOS untouched."""
import asyncio, json, time, hashlib
from pathlib import Path
import numpy as np
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,
"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
FILTERS=("baseline","ema21_slope3","adx18_directional","rsi_momentum","strong_candle_close",
"adx22_directional","ema21_slope5_atr","rsi_tight_momentum","avoid_extended_entry",
"distance_to_ema_atr_0_75","distance_to_ema_atr_1_0","slope_plus_adx",
"body_plus_adx22","slope_adx_candle_stack","trend_quality_stack")
ENTRIES=("structure_break_after_pullback","structure_break_pb8_br10","structure_break_pb5_br10","range_break_retest")
STOPS=("swing_3","swing_5","atr_1.0")
COOLDOWNS=((0,2),(1,1),(2,1),(3,1),(1,2),(2,2))
def clean(m):
 return {k:(float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v) for k,v in m.items()}
def gates(m):
 return (m["win_rate_pct"]>=30 and (m["profit_factor"] or 0)>1 and m["tp_per_day"]>=3
 and m["trades_per_day"]>=10 and m["max_consecutive_losses"]<=12 and not m["capital_exhausted"]
 and all(v>=300 for v in m["monthly_trade_counts"].values()))
def score(m):
 # Hard requirements dominate ranking; no sparse strategy can rank as a success.
 return (gates(m), m["trades_per_day"]>=10, m["tp_per_day"]>=3,
         m["max_consecutive_losses"]<=12, m["win_rate_pct"]>=30,
         (m["profit_factor"] or 0)>1, m["win_rate_pct"],m["profit_factor"] or 0,
         m["tp_per_day"],m["trades_per_day"],m["net_pnl"])
def sig_hash(ledger):
 return hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
async def main():
 started=time.monotonic(); sm.CONTRACT=normalize_contract(CFG)
 df,manifest=await get_data(); splits=split_reference_period(df)
 report={"schema":"asymmetric_side_filter_cooldown_sweep_v1","data_manifest":manifest,"contract":CFG,
 "oos_evaluated":False,"oos_used_for_selection":False,
 "ambiguity_note":"1m OHLC same-bar SL/TP resolves conservatively as SL; this is research, not final 1s ambiguity validation.",
 "filters":list(FILTERS),"entries":list(ENTRIES),"stops":list(STOPS),"cooldowns":[{"bars":b,"trigger_losses":t} for b,t in COOLDOWNS],"splits":{}}
 cross={}
 for split in ("train","development"):
  part=splits[split]; signals,atr=sm.signals(part); filters=sm.confirmation_filters(part)
  # Exact 11-loss reference on each split for apples-to-apples comparison.
  base_long,base_short=signals["structure_break_after_pullback"]
  fl,fs=filters["ema21_slope5_atr"]
  bm,_=sm.run_combined(part,base_long&fl.fillna(False),base_short&fs.fillna(False),atr,"swing_5",loss_cooldown_bars=1,cooldown_trigger_losses=1)
  baseline=clean(bm); rows=[]
  for entry in ENTRIES:
   raw_l,raw_s=signals[entry]
   for lf in FILTERS:
    lfilter=filters[lf][0].reindex(part.index).fillna(False)
    long_sig=raw_l & lfilter
    for sf in FILTERS:
     sfilter=filters[sf][1].reindex(part.index).fillna(False)
     short_sig=raw_s & sfilter
     # If both sides signal simultaneously, engine conservatively skips the conflict.
     for stop in STOPS:
      for bars,trigger in COOLDOWNS:
       m,ledger=sm.run_combined(part,long_sig,short_sig,atr,stop,loss_cooldown_bars=bars,cooldown_trigger_losses=trigger)
       m=clean(m)
       row={"entry":entry,"long_filter":lf,"short_filter":sf,"stop":stop,"cooldown_bars":bars,"cooldown_trigger_losses":trigger,
            "metrics":m,"ledger_sha256":sig_hash(ledger)}
       rows.append(row); cross[(entry,lf,sf,stop,bars,trigger,split)]=row
  rows.sort(key=lambda r:score(r["metrics"]),reverse=True)
  passes=[r for r in rows if gates(r["metrics"])]
  freq_streak=[r for r in rows if r["metrics"]["trades_per_day"]>=7.5 and r["metrics"]["max_consecutive_losses"]<=12]
  report["splits"][split]={"tested":len(rows),"baseline_11_loss_metrics":baseline,"pass_all_gates":len(passes),
    "top_by_full_gate_proximity":rows[:20],"top_win_rate_with_frequency_and_streak":sorted(freq_streak,key=lambda r:(r["metrics"]["win_rate_pct"],r["metrics"]["profit_factor"] or 0,r["metrics"]["tp_per_day"]),reverse=True)[:15],
    "top_30pct_wr_candidates": [r for r in rows if r["metrics"]["win_rate_pct"]>=30 and r["metrics"]["trades_per_day"]>=7.5][:15]}
  print("ASYMMETRIC_SPLIT "+json.dumps({"split":split,"tested":len(rows),"baseline":baseline,"passes":len(passes),"top":rows[:3]},sort_keys=True),flush=True)
 # Matched configurations: require evidence across both Train and Development, not one lucky split.
 matched=[]
 for entry in ENTRIES:
  for lf in FILTERS:
   for sf in FILTERS:
    for stop in STOPS:
     for bars,trigger in COOLDOWNS:
      tr=cross[(entry,lf,sf,stop,bars,trigger,"train")]
      dv=cross[(entry,lf,sf,stop,bars,trigger,"development")]
      matched.append({"entry":entry,"long_filter":lf,"short_filter":sf,"stop":stop,"cooldown_bars":bars,"cooldown_trigger_losses":trigger,
        "train":tr["metrics"],"development":dv["metrics"],"train_pass":gates(tr["metrics"]),"development_pass":gates(dv["metrics"]),
        "train_ledger_sha256":tr["ledger_sha256"],"development_ledger_sha256":dv["ledger_sha256"]})
 matched.sort(key=lambda x:(x["train_pass"] and x["development_pass"],
  min(x["train"]["trades_per_day"],x["development"]["trades_per_day"])>=7.5,
  min(x["train"]["max_consecutive_losses"],x["development"]["max_consecutive_losses"])<=12,
  min(x["train"]["win_rate_pct"],x["development"]["win_rate_pct"])>=30,
  min(x["train"]["win_rate_pct"],x["development"]["win_rate_pct"]),
  min(x["train"]["profit_factor"] or 0,x["development"]["profit_factor"] or 0),
  min(x["train"]["tp_per_day"],x["development"]["tp_per_day"])),reverse=True)
 report["cross_split"]={"matched_configurations":len(matched),"both_splits_pass_all_gates":sum(x["train_pass"] and x["development_pass"] for x in matched),"top":matched[:30]}
 report["elapsed_seconds"]=round(time.monotonic()-started,2)
 path=OUT/"asymmetric_side_filter_cooldown_sweep.json"; path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
 print("ASYMMETRIC_FINAL "+json.dumps({"report":str(path),"elapsed_seconds":report["elapsed_seconds"],"both_pass":report["cross_split"]["both_splits_pass_all_gates"],"top":matched[:10]},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
