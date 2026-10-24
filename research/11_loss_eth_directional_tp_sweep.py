"""ETH-only follow-up: test LONG/SHORT selection and entry quality for the 11-loss setup.
Research-only; no production deployment and no OOS selection.
"""
import asyncio, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT=Path('research-results'); OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
ENTRIES=('structure_break_after_pullback','structure_break_pb3_br5','structure_break_pb3_br10','structure_break_pb3_br15','structure_break_pb5_br10','structure_break_pb5_br15','structure_break_pb5_br20','structure_break_pb8_br15')
FILTERS=('baseline','ema21_slope3','ema21_slope5_atr','adx18_directional','rsi_momentum','volume_above_sma20','strong_candle_close','adx22_directional','rsi_tight_momentum','avoid_extended_entry','slope_plus_adx','body_plus_adx22')
STOPS=('swing_3','swing_5')
COOLDOWNS=((0,1),(1,1),(1,2))
SIDES=('BOTH','LONG_ONLY','SHORT_ONLY')

def clean(m):
 return {k:(float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v) for k,v in m.items()}

def rank_tuple(m):
 return (m['tp_per_day'],m['win_rate_pct'],m['profit_factor'] or 0,m['trades_per_day'])

async def main():
 started=time.monotonic(); sm.CONTRACT=normalize_contract(CFG)
 df,manifest=await get_data(); splits=split_reference_period(df)
 report={'schema':'eth_11_loss_directional_tp_sweep_v1','data_manifest':manifest,'contract':CFG,'oos_evaluated':False,'ambiguity_note':'1m OHLC only; ambiguous same-bar stop/target is conservatively counted as SL. Exploratory, not final validation.','selection_rule':'Report all; shortlist requires max consecutive losses <=12 on both train and development, no capital exhaustion on either, and at least 70% of baseline trade count in each split.','splits':{},'elapsed_seconds':None}
 all_results={}
 for split_name in ('train','development'):
  part=splits[split_name]; sigs,atr=sm.signals(part); filters=sm.confirmation_filters(part); rows=[]
  for ent in ENTRIES:
   raw_l,raw_s=sigs[ent]
   for fname in FILTERS:
    lf,sf=filters[fname]
    ls=(raw_l & lf.fillna(False)).astype(bool); ss=(raw_s & sf.fillna(False)).astype(bool)
    for side_mode in SIDES:
     if side_mode=='LONG_ONLY': ss=pd.Series(False,index=part.index)
     elif side_mode=='SHORT_ONLY': ls=pd.Series(False,index=part.index)
     for stop in STOPS:
      for cb,ct in COOLDOWNS:
       m,ledger=sm.run_combined(part,ls,ss,atr,stop,loss_cooldown_bars=cb,cooldown_trigger_losses=ct)
       m=clean(m); item={'split':split_name,'entry':ent,'filter':fname,'side_mode':side_mode,'stop':stop,'cooldown_bars':cb,'cooldown_trigger_losses':ct,'metrics':m,'ledger_sha256':hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
       rows.append(item); all_results[(ent,fname,side_mode,stop,cb,ct,split_name)]=item
  # exact baseline reference using current known setup
  bl,bs=sigs['structure_break_after_pullback']; fl,fs=filters['ema21_slope5_atr']
  bm,blg=sm.run_combined(part,pd.Series(bl & fl.fillna(False),index=part.index),pd.Series(bs & fs.fillna(False),index=part.index),atr,'swing_5',loss_cooldown_bars=1,cooldown_trigger_losses=1)
  bm=clean(bm)
  report['splits'][split_name]={'tested_candidates':len(rows),'baseline_metrics':bm,'baseline_ledger_sha256':hashlib.sha256(json.dumps(blg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'top_tp_all':sorted(rows,key=lambda x:rank_tuple(x['metrics']),reverse=True)[:15],'top_tp_with_streak12':sorted([x for x in rows if x['metrics']['max_consecutive_losses']<=12],key=lambda x:rank_tuple(x['metrics']),reverse=True)[:15],'top_wr_with_streak12':sorted([x for x in rows if x['metrics']['max_consecutive_losses']<=12],key=lambda x:(x['metrics']['win_rate_pct'],x['metrics']['tp_per_day'],x['metrics']['profit_factor'] or 0),reverse=True)[:15]}
  print('DIRECTION_SWEEP '+json.dumps({'split':split_name,'tested':len(rows),'baseline':bm,'top_tp':report['splits'][split_name]['top_tp_with_streak12'][:3]},sort_keys=True),flush=True)
 # strict cross-split shortlist, same parameter tuple and <=12 streak both; no capital exhaustion; retain >=70% trades of baseline
 base_train=report['splits']['train']['baseline_metrics']; base_dev=report['splits']['development']['baseline_metrics']; cross=[]
 keys=[k[:-1] for k in all_results if k[-1]=='development']
 for k in keys:
  tr=all_results[k+('train',)]; dv=all_results[k+('development',)]; tm=tr['metrics']; dm=dv['metrics']
  if tm['max_consecutive_losses']<=12 and dm['max_consecutive_losses']<=12 and not tm['capital_exhausted'] and not dm['capital_exhausted'] and tm['trades']>=.70*base_train['trades'] and dm['trades']>=.70*base_dev['trades']:
   cross.append({'entry':k[0],'filter':k[1],'side_mode':k[2],'stop':k[3],'cooldown_bars':k[4],'cooldown_trigger_losses':k[5],'train':tm,'development':dm,'train_ledger_sha256':tr['ledger_sha256'],'development_ledger_sha256':dv['ledger_sha256']})
 cross.sort(key=lambda x:(min(x['train']['tp_per_day'],x['development']['tp_per_day']),min(x['train']['win_rate_pct'],x['development']['win_rate_pct']),min(x['train']['profit_factor'] or 0,x['development']['profit_factor'] or 0)),reverse=True)
 report['cross_split_shortlist']={'count':len(cross),'top':cross[:30]}
 report['elapsed_seconds']=round(time.monotonic()-started,2)
 p=OUT/'11_loss_eth_directional_tp_sweep_report.json'; p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
 print('DIRECTION_SWEEP_FINAL '+json.dumps({'elapsed_seconds':report['elapsed_seconds'],'cross_split_shortlist_count':len(cross),'top':cross[:10]},sort_keys=True),flush=True)
if __name__=='__main__': asyncio.run(main())
