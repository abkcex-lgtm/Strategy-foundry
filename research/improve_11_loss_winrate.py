"""Research sweep: improve the prior 11-loss candidate without changing its risk/reward contract.
Real Binance ETHUSDT 1m data; Train + Development only, OOS untouched.
Intrabar SL/TP ambiguity is fail-closed using the current 1m OHLC engine (counted as SL), not 1s-resolved.
"""
import asyncio, json, hashlib
from pathlib import Path
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period
OUT=Path('research-results'); OUT.mkdir(exist_ok=True)
BASE={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","min_win_rate_pct":40,"min_profit_factor":1,"min_tp_per_day":3,"max_consecutive_losses":12,"use_min_win_rate":True,"use_min_profit_factor":True,"use_min_tp_per_day":True,"use_max_consecutive_losses":True,"min_trades_per_day":10,"min_trades_per_month":300,"break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
CANDIDATE={"entry_method":"structure_break_after_pullback","confirmation_filter":"ema21_slope5_atr","stop_method":"swing_5","cooldown_bars":1,"cooldown_trigger_losses":1,"side":"LONG+SHORT_SHARED_SINGLE_POSITION"}
def metrics_short(m):
 keys=('trades','wins','win_rate_pct','profit_factor','tp_per_day','trades_per_day','max_consecutive_losses','net_pnl','final_balance','total_costs','capital_exhausted','unresolved_ohlc_bars_count','monthly_trade_counts')
 return {k:m.get(k) for k in keys}
def rank(m):
 # Prefer higher WR only among variants that do not regress baseline on PF, TP/day, trades/day, streak, or PnL.
 return (m['win_rate_pct'],m['profit_factor'] or 0,m['tp_per_day'],-m['max_consecutive_losses'],m['net_pnl'])
async def main():
 df,manifest=await get_data(); splits=split_reference_period(df)
 out={"schema":"improve_11_loss_candidate_winrate_v1","data":manifest,"candidate":CANDIDATE,"contract":{"initial_balance":1000,"net_risk_usd":10,"net_reward_usd":35,"commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"single_shared_position":True,"next_candle_open":True,"SL_TP_only":True},"oos_evaluated":False,"ambiguity_policy":"Current 1m OHLC resolver fails closed: simultaneous SL/TP in a candle is counted as SL; 1s execution resolution is not implemented in this sweep.","splits":{},"variants":[]}
 for split_name in ('train','development'):
  part=splits[split_name]; sigs,atr=sm.signals(part); filters=sm.confirmation_filters(part); raw_l,raw_s=sigs[CANDIDATE['entry_method']]
  names=list(filters.keys())
  if CANDIDATE['confirmation_filter'] not in names: names.insert(0,CANDIDATE['confirmation_filter'])
  # Baseline and one-filter-at-a-time candidates: all other rules remain fixed.
  splitres=[]
  for name in names:
   lf,sf=filters[name]; ls=raw_l & lf.fillna(False); ss=raw_s & sf.fillna(False)
   for cool,trigger in ((1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(5,2)):
    contract=normalize_contract(BASE); sm.CONTRACT=contract
    m,ledger=sm.run_combined(part,ls,ss,atr,CANDIDATE['stop_method'],loss_cooldown_bars=cool,cooldown_trigger_losses=trigger)
    item={"split":split_name,"filter":name,"cooldown_bars":cool,"cooldown_trigger_losses":trigger,"metrics":metrics_short(m),"ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
    splitres.append(item); out['variants'].append(item)
    print('WINRATE_SWEEP '+json.dumps(item,sort_keys=True),flush=True)
  out['splits'][split_name]={"variant_count":len(splitres),"top_by_win_rate":sorted(splitres,key=lambda x:rank(x['metrics']),reverse=True)[:10]}
 # Select only Development candidates with no regression on baseline PF, TP/day, trades/day, max streak and net PnL; Train must also be inspected before recommending.
 dev=[x for x in out['variants'] if x['split']=='development']
 base=next(x for x in dev if x['filter']==CANDIDATE['confirmation_filter'] and x['cooldown_bars']==1 and x['cooldown_trigger_losses']==1)
 bm=base['metrics']
 eligible=[]
 for x in dev:
  m=x['metrics']
  if (m['win_rate_pct']>bm['win_rate_pct'] and (m['profit_factor'] or 0)>=bm['profit_factor'] and m['tp_per_day']>=bm['tp_per_day'] and m['trades_per_day']>=bm['trades_per_day'] and m['max_consecutive_losses']<=bm['max_consecutive_losses'] and m['net_pnl']>=bm['net_pnl']): eligible.append(x)
 out['baseline_development']=base; out['eligible_no_regression_development']=sorted(eligible,key=lambda x:rank(x['metrics']),reverse=True)[:20]
 out['note']='No variant is recommended unless its Train metrics are also checked against the same baseline; OOS remains untouched.'
 p=OUT/'improve_11_loss_winrate_report.json'; p.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))
 print('WINRATE_FINAL '+json.dumps({'report':str(p),'baseline_development':base,'eligible_count':len(eligible),'top_eligible':out['eligible_no_regression_development'][:5]},sort_keys=True),flush=True)
if __name__=='__main__': asyncio.run(main())