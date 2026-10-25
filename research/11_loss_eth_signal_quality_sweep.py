"""ETH-only quality sweep for the historical 11-loss candidate. Research-only; never evaluates OOS for selection."""
import asyncio, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period, frame_hash

OUT=Path('research-results'); OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
BASE_ENTRY='structure_break_after_pullback'; BASE_FILTER='ema21_slope5_atr'; BASE_STOP='swing_5'; BASE_COOLDOWN=(1,1)
FILTERS=('baseline','ema21_slope3','adx18_directional','rsi_momentum','volume_above_sma20','atr_midrange','slope_plus_adx','strong_candle_close','adx22_directional','ema21_slope5_atr','rsi_tight_momentum','avoid_extended_entry','distance_to_ema_atr_0_75','distance_to_ema_atr_1_0','distance_to_ema_atr_1_5','body_plus_adx22','slope_adx_candle_stack','trend_quality_stack')
ENTRIES=('structure_break_after_pullback','structure_break_pb3_br10','structure_break_pb5_br10','structure_break_pb5_br15','structure_break_pb8_br15','structure_break_pb3_br5','structure_break_pb3_br15','structure_break_pb5_br20','structure_break_pb8_br10','structure_break_pb8_br20','structure_break_pb10_br15','structure_break_pb10_br20')
STOPS=('swing_3','swing_5')
COOLDOWNS=((0,1),(1,1),(2,1),(1,2),(2,2))

def serial_metrics(m):
    return {k:(float(v) if isinstance(v,(np.floating,)) else int(v) if isinstance(v,(np.integer,)) else v) for k,v in m.items()}

async def main():
    started=time.monotonic(); sm.CONTRACT=normalize_contract(CFG)
    df,manifest=await get_data(); splits=split_reference_period(df)
    report={'schema':'eth_11_loss_signal_quality_sweep_v2_directional_diagnostic','data_manifest':manifest,'baseline':{'entry':BASE_ENTRY,'filter':BASE_FILTER,'stop':BASE_STOP,'cooldown_bars':1,'cooldown_trigger_losses':1},'contract':CFG,'oos_evaluated':False,'directional_diagnostic':True,'direction_note':'Each candidate is evaluated as combined long+short, long-only, and short-only. Single-direction rows are diagnostics only; final deployment still requires user approval and all fixed constraints.','ambiguity_note':'1m OHLC only; unresolved same-bar SL/TP is counted as SL; 1s resolution not implemented, so exploratory only.','splits':{},'elapsed_seconds':None}
    all_rows=[]
    for split_name in ('train','development'):
        part=splits[split_name]; signals,atr=sm.signals(part); filters=sm.confirmation_filters(part)
        rows=[]
        for entry_name in ENTRIES:
            raw_l,raw_s=signals[entry_name]
            for filter_name in FILTERS:
                lf,sf=filters[filter_name]; long_sig=raw_l & lf.fillna(False); short_sig=raw_s & sf.fillna(False)
                for direction in ('both','long_only','short_only'):
                    if direction=='long_only': short_sig_test=np.zeros(len(part),dtype=bool); long_sig_test=long_sig
                    elif direction=='short_only': long_sig_test=np.zeros(len(part),dtype=bool); short_sig_test=short_sig
                    else: long_sig_test=long_sig; short_sig_test=short_sig
                    for stop in STOPS:
                        for cooldown_bars,cooldown_trigger in COOLDOWNS:
                            metrics,ledger=sm.run_combined(part,pd.Series(long_sig_test,index=part.index),pd.Series(short_sig_test,index=part.index),atr,stop,loss_cooldown_bars=cooldown_bars,cooldown_trigger_losses=cooldown_trigger)
                            metrics=serial_metrics(metrics)
                            item={'split':split_name,'entry':entry_name,'filter':filter_name,'direction':direction,'stop':stop,'cooldown_bars':cooldown_bars,'cooldown_trigger_losses':cooldown_trigger,'metrics':metrics,'ledger_sha256':hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
                            rows.append(item); all_rows.append(item)
        # baseline exact replay
        bl,bs=signals[BASE_ENTRY]; fl,fs=filters[BASE_FILTER]
        bm,blg=sm.run_combined(part,pd.Series(bl & fl.fillna(False),index=part.index),pd.Series(bs & fs.fillna(False),index=part.index),atr,BASE_STOP,loss_cooldown_bars=BASE_COOLDOWN[0],cooldown_trigger_losses=BASE_COOLDOWN[1])
        # Candidate ranking is constrained: never accept a worse streak than the split baseline;
        # prioritize WR, then PF, with trade frequency as a tie-break, but report all candidates.
        bm=serial_metrics(bm)
        eligible=[x for x in rows if x['metrics']['max_consecutive_losses']<=bm['max_consecutive_losses'] and x['metrics']['trades']>=max(30,int(0.50*bm['trades']))]
        eligible.sort(key=lambda x:(x['metrics']['win_rate_pct'],x['metrics']['profit_factor'] or 0,x['metrics']['trades']),reverse=True)
        report['splits'][split_name]={'baseline_metrics':bm,'baseline_ledger_sha256':hashlib.sha256(json.dumps(blg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'tested_candidates':len(rows),'eligible_count':len(eligible),'top_eligible':eligible[:20],'best_by_win_rate_all':sorted(rows,key=lambda x:x['metrics']['win_rate_pct'],reverse=True)[:10],'best_by_profit_factor_all':sorted(rows,key=lambda x:(x['metrics']['profit_factor'] or 0),reverse=True)[:10]}
        print('ETH_QUALITY_SWEEP '+json.dumps({'split':split_name,'baseline':bm,'tested':len(rows),'eligible':len(eligible),'top':eligible[:5]},sort_keys=True),flush=True)
    # Cross-split candidates must preserve each split's max loss streak; no OOS touched.
    dev={ (x['entry'],x['filter'],x.get('direction','both'),x['stop'],x['cooldown_bars'],x['cooldown_trigger_losses']):x for x in all_rows if x['split']=='development'}
    trn={ (x['entry'],x['filter'],x.get('direction','both'),x['stop'],x['cooldown_bars'],x['cooldown_trigger_losses']):x for x in all_rows if x['split']=='train'}
    cross=[]
    for k,d in dev.items():
        t=trn[k]
        if d['metrics']['max_consecutive_losses']<=report['splits']['development']['baseline_metrics']['max_consecutive_losses'] and t['metrics']['max_consecutive_losses']<=report['splits']['train']['baseline_metrics']['max_consecutive_losses']:
            cross.append({'entry':k[0],'filter':k[1],'direction':k[2],'stop':k[3],'cooldown_bars':k[4],'cooldown_trigger_losses':k[5],'train':t['metrics'],'development':d['metrics'],'train_ledger_sha256':t['ledger_sha256'],'development_ledger_sha256':d['ledger_sha256']})
    cross.sort(key=lambda x:(min(x['train']['win_rate_pct'],x['development']['win_rate_pct']),min(x['train']['profit_factor'] or 0,x['development']['profit_factor'] or 0),min(x['train']['trades'],x['development']['trades'])),reverse=True)
    report['cross_split_candidates_preserving_streak']={'count':len(cross),'top':cross[:30]}
    report['elapsed_seconds']=round(time.monotonic()-started,2)
    path=OUT/'11_loss_eth_signal_quality_sweep_report.json'; path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    print('ETH_QUALITY_SWEEP_FINAL '+json.dumps({'report':str(path),'elapsed_seconds':report['elapsed_seconds'],'cross_split_candidates':len(cross),'top':cross[:10]},sort_keys=True),flush=True)
if __name__=='__main__': asyncio.run(main())
