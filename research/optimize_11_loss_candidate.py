import asyncio,json,hashlib
from pathlib import Path
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data,split_reference_period
OUT=Path("research-results");OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
BASE=("structure_break_after_pullback","ema21_slope5_atr","swing_5",1,1)
FILTERS=("baseline","ema21_slope5_atr","strong_candle_close","adx18_directional","rsi_momentum","distance_to_ema_atr_1_0")
def short(m):
 k=("trades","wins","win_rate_pct","profit_factor","tp_per_day","trades_per_day","max_consecutive_losses","net_pnl","final_balance","total_costs","capital_exhausted","monthly_trade_counts")
 return {x:m.get(x) for x in k}
def good(m,b):
 return m["win_rate_pct"]>b["win_rate_pct"] and (m["profit_factor"] or 0)>=(b["profit_factor"] or 0) and m["tp_per_day"]>=b["tp_per_day"] and m["trades_per_day"]>=b["trades_per_day"] and m["max_consecutive_losses"]<=b["max_consecutive_losses"] and m["net_pnl"]>=b["net_pnl"] and not m["capital_exhausted"]
async def main():
 df,manifest=await get_data();splits=split_reference_period(df);sm.CONTRACT=normalize_contract(CFG)
 entries=list(sm.ENTRIES);stops=list(sm.STOPS)
 if BASE[0] not in entries:entries.append(BASE[0])
 if BASE[2] not in stops:stops.append(BASE[2])
 out={"schema":"optimize_11_loss_v2","data":manifest,"oos_evaluated":False,"ambiguity_policy":"Same-bar TP/SL is counted as SL on 1m OHLC; 1s resolution not implemented.","baseline_spec":BASE,"splits":{},"variants":[]}
 for sn in ("train","development"):
  part=splits[sn];signals,atr=sm.signals(part);filters=sm.confirmation_filters(part);res=[]
  for en in entries:
   if en not in signals:continue
   rl,rs=signals[en]
   for fn in FILTERS:
    fl,fs=filters[fn];ls=rl&fl.fillna(False);ss=rs&fs.fillna(False)
    for st in stops:
     for cb,ct in ((1,1),(2,1),(3,2)):
      m,ledger=sm.run_combined(part,ls,ss,atr,st,loss_cooldown_bars=cb,cooldown_trigger_losses=ct)
      x={"split":sn,"entry":en,"filter":fn,"stop":st,"cooldown_bars":cb,"cooldown_trigger_losses":ct,"metrics":short(m),"ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()}
      res.append(x);out["variants"].append(x)
  out["splits"][sn]={"count":len(res),"top_by_win_rate":sorted(res,key=lambda x:(x["metrics"]["win_rate_pct"],x["metrics"]["profit_factor"] or 0,x["metrics"]["tp_per_day"]),reverse=True)[:12]}
  print("SPLIT_DONE",sn,len(res),flush=True)
 base={}
 for sn in ("train","development"):
  z=[x for x in out["variants"] if x["split"]==sn and (x["entry"],x["filter"],x["stop"],x["cooldown_bars"],x["cooldown_trigger_losses"])==BASE]
  if z:base[sn]=z[0]
 out["baseline"]={sn:x["metrics"] for sn,x in base.items()}
 eligible=[]
 if len(base)==2:
  for x in out["variants"]:
   if x["split"]!="development":continue
   key=(x["entry"],x["filter"],x["stop"],x["cooldown_bars"],x["cooldown_trigger_losses"])
   t=[y for y in out["variants"] if y["split"]=="train" and (y["entry"],y["filter"],y["stop"],y["cooldown_bars"],y["cooldown_trigger_losses"])==key]
   if t and good(x["metrics"],base["development"]["metrics"]) and good(t[0]["metrics"],base["train"]["metrics"]):
    eligible.append({"spec":key,"development":x["metrics"],"train":t[0]["metrics"]})
 out["eligible_no_regression_both_splits"]=sorted(eligible,key=lambda x:(x["development"]["win_rate_pct"],x["development"]["profit_factor"] or 0,x["development"]["tp_per_day"]),reverse=True)[:20]
 out["note"]="Recommend only candidates that improve WR and do not regress PF, TP/day, trades/day, max loss streak or net PnL on both train and development. OOS untouched."
 p=OUT/"optimize_11_loss_report.json";p.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))
 print("FINAL",json.dumps({"report":str(p),"variants":len(out["variants"]),"eligible":len(eligible),"top":out["eligible_no_regression_both_splits"][:3]},sort_keys=True),flush=True)
if __name__=="__main__":asyncio.run(main())
