from __future__ import annotations
import hashlib, json, math
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from .data import validate_ohlcv, sha256_bytes, DataIntegrityError

CONTRACT={'initial_balance':1000.0,'risk_usd':10.0,'reward_risk':3.5,'fee_slippage_per_side':0.0006,'max_positions':1,'entry_model':'next_candle_open','exit_model':'sl_tp_only','max_consecutive_losses':13,'min_win_rate_pct':40.0,'min_profit_factor':1.0,'min_tp_per_day':3.0}

@dataclass(frozen=True)
class StrategySpec:
    name: str='EMA trend crossover'
    fast: int=9
    slow: int=36
    atr_period: int=14
    stop_atr: float=1.5
    direction: str='both'
    seed: int=42
    def validate(self):
        if not (2 <= self.fast < self.slow <= 500): raise ValueError('Require 2 <= fast < slow <= 500')
        if not (2 <= self.atr_period <= 200): raise ValueError('atr_period must be 2..200')
        if not (0.2 <= self.stop_atr <= 10): raise ValueError('stop_atr must be 0.2..10')
        if self.direction not in {'both','long','short'}: raise ValueError('Invalid direction')

def indicators(df, spec):
    x=df.copy(); c=x.close.astype(float); prev=c.shift(1)
    x['ema_fast']=c.ewm(span=spec.fast,adjust=False,min_periods=spec.fast).mean()
    x['ema_slow']=c.ewm(span=spec.slow,adjust=False,min_periods=spec.slow).mean()
    tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x['atr']=tr.ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
    x['long_signal']=(x.ema_fast>x.ema_slow)&(x.ema_fast.shift(1)<=x.ema_slow.shift(1))
    x['short_signal']=(x.ema_fast<x.ema_slow)&(x.ema_fast.shift(1)>=x.ema_slow.shift(1))
    return x

def _qty(entry, stop_distance, risk, fee):
    estimated_loss=stop_distance+(entry+max(entry-stop_distance,1e-12))*fee
    return risk/estimated_loss if estimated_loss>0 else 0.0

def backtest(frame, spec=None, contract=None):
    spec=StrategySpec(**spec) if isinstance(spec,dict) else (spec or StrategySpec()); spec.validate()
    cfg=dict(CONTRACT); cfg.update(contract or {})
    df=frame.copy().reset_index(drop=True)
    if len(df)>1:
        diffs=df.open_time.diff().dropna(); interval=int(diffs.mode().iloc[0]) if len(diffs) else None
        report=validate_ohlcv(df,interval)
    else: report=validate_ohlcv(df,None)
    if len(df)<max(spec.slow,spec.atr_period)+3: raise DataIntegrityError('Not enough candles for warmup')
    data_hash=sha256_bytes(df.to_csv(index=False,float_format='%.12g').encode())
    x=indicators(df,spec); balance=float(cfg['initial_balance']); ledger=[]; streak=0; max_streak=0
    i=max(spec.slow,spec.atr_period)
    while i < len(x)-1:
        row=x.iloc[i]
        if not np.isfinite(row.atr) or row.atr<=0: i+=1; continue
        side='LONG' if bool(row.long_signal) else ('SHORT' if bool(row.short_signal) else None)
        if side is None or (spec.direction=='long' and side=='SHORT') or (spec.direction=='short' and side=='LONG'): i+=1; continue
        entry_i=i+1; entry=float(x.iloc[entry_i].open); dist=float(row.atr*spec.stop_atr)
        stop=entry-dist if side=='LONG' else entry+dist
        target=entry+dist*float(cfg['reward_risk']) if side=='LONG' else entry-dist*float(cfg['reward_risk'])
        qty=_qty(entry,dist,float(cfg['risk_usd']),float(cfg['fee_slippage_per_side']))
        if qty<=0: i+=1; continue
        exit_i=exit_price=None; reason=None
        for j in range(entry_i,len(x)):
            c=x.iloc[j]
            sl= float(c.low)<=stop if side=='LONG' else float(c.high)>=stop
            tp= float(c.high)>=target if side=='LONG' else float(c.low)<=target
            if sl and tp: raise DataIntegrityError(f'Unresolved same-candle SL/TP ambiguity at row {j}; finer data required')
            if sl: exit_i=j; exit_price=stop; reason='SL'; break
            if tp: exit_i=j; exit_price=target; reason='TP'; break
        if exit_i is None: break
        gross=(exit_price-entry)*qty*(1 if side=='LONG' else -1)
        entry_cost=entry*qty*float(cfg['fee_slippage_per_side']); exit_cost=exit_price*qty*float(cfg['fee_slippage_per_side'])
        net=gross-entry_cost-exit_cost; balance+=net
        if reason=='TP': streak=0
        else: streak+=1; max_streak=max(max_streak,streak)
        ledger.append({'trade_id':len(ledger)+1,'signal_time':int(row.open_time),'entry_time':int(x.iloc[entry_i].open_time),'exit_time':int(x.iloc[exit_i].open_time),'direction':side,'signal_price':float(row.close),'entry_price':entry,'stop_loss':stop,'take_profit':target,'exit_price':float(exit_price),'exit_reason':reason,'quantity':qty,'gross_pnl':gross,'entry_cost':entry_cost,'exit_cost':exit_cost,'fees_slippage':entry_cost+exit_cost,'net_pnl':net,'balance_after':balance,'loss_streak_after':streak})
        i=exit_i+1
    wins=[t for t in ledger if t['exit_reason']=='TP' and t['net_pnl']>0]
    gp=sum(max(0,t['net_pnl']) for t in ledger); gl=abs(sum(min(0,t['net_pnl']) for t in ledger))
    span_days=max((int(df.open_time.iloc[-1])-int(df.open_time.iloc[0]))/86400000,1/1440)
    wr=100*len(wins)/len(ledger) if ledger else 0.0; pf=gp/gl if gl else (math.inf if gp else 0.0); tp_day=len(wins)/span_days
    passes=wr>cfg['min_win_rate_pct'] and pf>cfg['min_profit_factor'] and tp_day>=cfg['min_tp_per_day'] and max_streak<=cfg['max_consecutive_losses']
    metrics={'trades':len(ledger),'wins':len(wins),'losses':sum(t['net_pnl']<=0 for t in ledger),'win_rate_pct':wr,'profit_factor':pf if math.isfinite(pf) else None,'avg_net_tp_per_day':tp_day,'max_consecutive_losses':max_streak,'net_pnl':sum(t['net_pnl'] for t in ledger),'final_balance':balance,'total_costs':sum(t['fees_slippage'] for t in ledger),'data_hash_sha256':data_hash,'rows':len(df),'span_days':span_days,'passes_all_targets':bool(passes)}
    run_hash=sha256_bytes(json.dumps({'spec':asdict(spec),'contract':cfg,'data_hash':data_hash},sort_keys=True).encode())
    return {'strategy':asdict(spec),'contract':cfg,'metrics':metrics,'ledger':ledger,'data_report':report,'run_hash_sha256':run_hash}

def search_candidates(frame, max_candidates=500, seed=42):
    if not 1<=max_candidates<=10000: raise ValueError('max_candidates must be 1..10000 in this starter API')
    rng=np.random.default_rng(seed); results=[]; errors=0
    for _ in range(max_candidates):
        fast=int(rng.integers(4,40)); slow=int(rng.integers(max(fast+2,20),121)); period=int(rng.integers(7,31)); mult=float(rng.choice([0.8,1.0,1.2,1.5,2.0,2.5]))
        spec=StrategySpec(f'EMA {fast}/{slow} ATR {period}x{mult:g}',fast,slow,period,mult,'both',seed)
        try:
            r=backtest(frame,spec); m=r['metrics']
            dist=max(0,40.0001-m['win_rate_pct'])/40+max(0,1.0001-(m['profit_factor'] or 0))+max(0,3-m['avg_net_tp_per_day'])/3+max(0,m['max_consecutive_losses']-13)/13
            results.append({'strategy':r['strategy'],'metrics':m,'distance':dist,'run_hash_sha256':r['run_hash_sha256']})
        except (ValueError,DataIntegrityError,IndexError): errors+=1
    results.sort(key=lambda r:(not r['metrics']['passes_all_targets'],r['distance'],r['metrics']['max_consecutive_losses'],-r['metrics']['win_rate_pct'],-r['metrics']['avg_net_tp_per_day'],-(r['metrics']['profit_factor'] or 0)))
    passing=[r for r in results if r['metrics']['passes_all_targets']]
    selected=passing if passing else results[:5]
    return {'requested_candidates':max_candidates,'evaluated_candidates':len(results),'failed_candidates':errors,'seed':seed,'all_passing':bool(passing),'results':selected,'warning':'Bounded random search only; run on TRAIN data. No 500k+ or OOS certification.'}
