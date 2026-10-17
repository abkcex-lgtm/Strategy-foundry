'''Standalone replay: generated per run; does not import Strategy Foundry.'''
import sys, json, hashlib, math
import numpy as np
import pandas as pd
SPEC = __SPEC__
CFG = __CONTRACT__
EXPECTED_DATA_SHA256 = '__DATA_HASH__'
EXPECTED_RUN_SHA256 = '__RUN_HASH__'
def digest(b): return hashlib.sha256(b).hexdigest()
def validate(df):
    required=['open_time','open','high','low','close','volume']
    if any(c not in df for c in required): raise ValueError('Missing OHLCV columns')
    if df.empty or df[required].isna().any().any(): raise ValueError('Empty/missing OHLCV data')
    if df.open_time.duplicated().any() or not df.open_time.is_monotonic_increasing: raise ValueError('Duplicate or out-of-order timestamps')
    if (df[['open','high','low','close']]<=0).any().any() or (df.volume<0).any(): raise ValueError('Invalid price/volume')
    if (df.high<df[['open','close','low']].max(axis=1)).any() or (df.low>df[['open','close','high']].min(axis=1)).any(): raise ValueError('Invalid OHLC')
def risk_budget(balance):
    if CFG['risk_mode']=='fixed_usd': return float(CFG['risk_usd'])
    base=balance if (CFG['risk_mode']=='percent_equity' or CFG['compound']) else float(CFG['initial_balance'])
    return base*float(CFG['risk_pct'])/100
def size_target(entry,stop,side,risk):
    rate=(float(CFG['commission_pct_per_side'])+float(CFG['slippage_pct_per_side']))/100
    dist=entry-stop if side=='LONG' else stop-entry
    if dist<=0: raise ValueError('Invalid stop placement')
    qty=risk/(dist+entry*rate+stop*rate)
    if CFG['target_mode']=='gross_rr':
        target=entry+dist*float(CFG['reward_risk']) if side=='LONG' else entry-dist*float(CFG['reward_risk'])
    else:
        reward=risk*float(CFG['reward_risk'])
        target=(entry*(1+rate)+reward/qty)/(1-rate) if side=='LONG' else (entry*(1-rate)-reward/qty)/(1+rate)
    if target<=0: raise ValueError('Non-positive target')
    return qty,target,rate
def replay(df):
    df=df.copy().reset_index(drop=True); validate(df)
    data_hash=digest(df.to_csv(index=False,float_format='%.12g').encode())
    s=SPEC; x=df.copy(); c=x.close.astype(float); prev=c.shift(1)
    x['ema_fast']=c.ewm(span=s['fast'],adjust=False,min_periods=s['fast']).mean()
    x['ema_slow']=c.ewm(span=s['slow'],adjust=False,min_periods=s['slow']).mean()
    tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x['atr']=tr.ewm(alpha=1/s['atr_period'],adjust=False,min_periods=s['atr_period']).mean()
    family=s.get('family','ema_crossover')
    long=(x.ema_fast>x.ema_slow)&(x.ema_fast.shift(1)<=x.ema_slow.shift(1))
    short=(x.ema_fast<x.ema_slow)&(x.ema_fast.shift(1)>=x.ema_slow.shift(1))
    if family in ('channel_breakout','volume_breakout'):
        upper=x.high.rolling(s['slow'],min_periods=s['slow']).max().shift(1)
        lower=x.low.rolling(s['slow'],min_periods=s['slow']).min().shift(1)
        long=(c>upper)&(c.shift(1)<=upper.shift(1))
        short=(c<lower)&(c.shift(1)>=lower.shift(1))
        if family=='volume_breakout':
            baseline=x.volume.rolling(s['fast'],min_periods=s['fast']).mean().shift(1)
            active=x.volume>=baseline*1.5
            long=long&active
            short=short&active
    elif family=='rsi_reversion':
        delta=c.diff()
        gain=delta.clip(lower=0).ewm(alpha=1/s['fast'],adjust=False,min_periods=s['fast']).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/s['fast'],adjust=False,min_periods=s['fast']).mean()
        rs=gain/loss.replace(0,np.nan)
        rsi=100-100/(1+rs)
        long=(rsi.shift(1)<=30)&(rsi>30)
        short=(rsi.shift(1)>=70)&(rsi<70)
    elif family=='ema_pullback':
        up=x.ema_fast>x.ema_slow; down=x.ema_fast<x.ema_slow
        long=up&(c>x.ema_fast)&(c.shift(1)<=x.ema_fast.shift(1))
        short=down&(c<x.ema_fast)&(c.shift(1)>=x.ema_fast.shift(1))
    elif family=='macd_crossover':
        macd=c.ewm(span=s['fast'],adjust=False,min_periods=s['fast']).mean()-c.ewm(span=s['slow'],adjust=False,min_periods=s['slow']).mean()
        sig=macd.ewm(span=max(2,s['fast']//2),adjust=False,min_periods=max(2,s['fast']//2)).mean()
        long=(macd>sig)&(macd.shift(1)<=sig.shift(1)); short=(macd<sig)&(macd.shift(1)>=sig.shift(1))
    elif family=='bollinger_reversion':
        mid=c.rolling(s['slow'],min_periods=s['slow']).mean(); dev=c.rolling(s['slow'],min_periods=s['slow']).std(ddof=0)
        upper=mid+2*dev; lower=mid-2*dev
        long=(c>lower)&(c.shift(1)<=lower.shift(1)); short=(c<upper)&(c.shift(1)>=upper.shift(1))
    elif family=='inside_bar_breakout':
        inside=(x.high.shift(1)<x.high.shift(2))&(x.low.shift(1)>x.low.shift(2))
        long=inside&(c>x.high.shift(1))&(c.shift(1)<=x.high.shift(1))
        short=inside&(c<x.low.shift(1))&(c.shift(1)>=x.low.shift(1))
    elif family=='engulfing_reversal':
        po=x.open.shift(1); pc=c.shift(1)
        long=(c>x.open)&(pc<po)&(x.open<=pc)&(c>=po)
        short=(c<x.open)&(pc>po)&(x.open>=pc)&(c<=po)
    elif family=='momentum_breakout':
        upper=x.high.rolling(s['slow'],min_periods=s['slow']).max().shift(1); lower=x.low.rolling(s['slow'],min_periods=s['slow']).min().shift(1)
        long=(c>c.shift(s['fast']))&(c>upper)&(c.shift(1)<=upper.shift(1))
        short=(c<c.shift(s['fast']))&(c<lower)&(c.shift(1)>=lower.shift(1))
    elif family=='ema_pullback':
        up=x.ema_fast>x.ema_slow; down=x.ema_fast<x.ema_slow
        long=up&(c>x.ema_fast)&(c.shift(1)<=x.ema_fast.shift(1))
        short=down&(c<x.ema_fast)&(c.shift(1)>=x.ema_fast.shift(1))
    elif family=='macd_crossover':
        macd=c.ewm(span=s['fast'],adjust=False,min_periods=s['fast']).mean()-c.ewm(span=s['slow'],adjust=False,min_periods=s['slow']).mean()
        sig=macd.ewm(span=max(2,s['fast']//2),adjust=False,min_periods=max(2,s['fast']//2)).mean()
        long=(macd>sig)&(macd.shift(1)<=sig.shift(1)); short=(macd<sig)&(macd.shift(1)>=sig.shift(1))
    elif family=='bollinger_reversion':
        mid=c.rolling(s['slow'],min_periods=s['slow']).mean(); dev=c.rolling(s['slow'],min_periods=s['slow']).std(ddof=0)
        upper=mid+2*dev; lower=mid-2*dev
        long=(c>lower)&(c.shift(1)<=lower.shift(1)); short=(c<upper)&(c.shift(1)>=upper.shift(1))
    elif family=='inside_bar_breakout':
        inside=(x.high.shift(1)<x.high.shift(2))&(x.low.shift(1)>x.low.shift(2))
        long=inside&(c>x.high.shift(1))&(c.shift(1)<=x.high.shift(1))
        short=inside&(c<x.low.shift(1))&(c.shift(1)>=x.low.shift(1))
    elif family=='engulfing_reversal':
        po=x.open.shift(1); pc=c.shift(1)
        long=(c>x.open)&(pc<po)&(x.open<=pc)&(c>=po)
        short=(c<x.open)&(pc>po)&(x.open>=pc)&(c<=po)
    elif family=='momentum_breakout':
        upper=x.high.rolling(s['slow'],min_periods=s['slow']).max().shift(1); lower=x.low.rolling(s['slow'],min_periods=s['slow']).min().shift(1)
        long=(c>c.shift(s['fast']))&(c>upper)&(c.shift(1)<=upper.shift(1))
        short=(c<c.shift(s['fast']))&(c<lower)&(c.shift(1)>=lower.shift(1))
    x['long_signal']=long.fillna(False)
    x['short_signal']=short.fillna(False)
    balance=float(CFG['initial_balance']); ledger=[]; streak=0; max_streak=0; i=max(s['slow'],s['atr_period'])
    while i<len(x)-1:
        row=x.iloc[i]
        if not np.isfinite(row.atr) or row.atr<=0: i+=1; continue
        side='LONG' if bool(row.long_signal) else ('SHORT' if bool(row.short_signal) else None)
        if side is None or (s['direction']=='long' and side=='SHORT') or (s['direction']=='short' and side=='LONG'): i+=1; continue
        entry_i=i+1; entry=float(x.iloc[entry_i].open); dist=float(row.atr*s['stop_atr'])
        stop=entry-dist if side=='LONG' else entry+dist
        risk=risk_budget(balance if (CFG['compound'] or CFG['risk_mode']=='percent_equity') else CFG['initial_balance'])
        if risk<=0 or (CFG['risk_mode']=='percent_equity' and risk>=balance): i+=1; continue
        try: qty,target,rate=size_target(entry,stop,side,risk)
        except ValueError: i+=1; continue
        exit_i=exit_price=reason=None
        for j in range(entry_i,len(x)):
            bar=x.iloc[j]; sl=float(bar.low)<=stop if side=='LONG' else float(bar.high)>=stop; tp=float(bar.high)>=target if side=='LONG' else float(bar.low)<=target
            if sl and tp: raise ValueError('Unresolved same-candle SL/TP ambiguity at row '+str(j))
            if sl: exit_i=j; exit_price=stop; reason='SL'; break
            if tp: exit_i=j; exit_price=target; reason='TP'; break
        if exit_i is None: break
        gross=(exit_price-entry)*qty*(1 if side=='LONG' else -1); entry_cost=entry*qty*rate; exit_cost=exit_price*qty*rate; net=gross-entry_cost-exit_cost; balance+=net
        if reason=='TP': streak=0
        else: streak+=1; max_streak=max(max_streak,streak)
        ledger.append({'trade_id':len(ledger)+1,'signal_time':int(row.open_time),'entry_time':int(x.iloc[entry_i].open_time),'exit_time':int(x.iloc[exit_i].open_time),'direction':side,'signal_price':float(row.close),'entry_price':entry,'stop_loss':stop,'take_profit':target,'exit_price':float(exit_price),'exit_reason':reason,'quantity':qty,'risk_budget':risk,'gross_pnl':gross,'entry_cost':entry_cost,'exit_cost':exit_cost,'fees_slippage':entry_cost+exit_cost,'net_pnl':net,'balance_after':balance,'loss_streak_after':streak})
        i=exit_i+1
    wins=[t for t in ledger if t['exit_reason']=='TP' and t['net_pnl']>0]; gp=sum(max(0,t['net_pnl']) for t in ledger); gl=abs(sum(min(0,t['net_pnl']) for t in ledger))
    span=max((int(df.open_time.iloc[-1])-int(df.open_time.iloc[0]))/86400000,1/1440)
    wr=100*len(wins)/len(ledger) if ledger else 0.0; pf=gp/gl if gl else (math.inf if gp else 0.0); tpday=len(wins)/span
    metrics={'trades':len(ledger),'wins':len(wins),'losses':sum(t['net_pnl']<=0 for t in ledger),'win_rate_pct':wr,'profit_factor':pf if math.isfinite(pf) else None,'avg_net_tp_per_day':tpday,'max_consecutive_losses':max_streak,'net_pnl':sum(t['net_pnl'] for t in ledger),'final_balance':balance,'total_costs':sum(t['fees_slippage'] for t in ledger),'data_hash_sha256':data_hash,'rows':len(df),'span_days':span,'passes_all_targets':bool((not CFG.get('use_min_win_rate',True) or wr>CFG['min_win_rate_pct']) and (not CFG.get('use_min_profit_factor',True) or pf>CFG['min_profit_factor']) and (not CFG.get('use_min_tp_per_day',True) or tpday>=CFG['min_tp_per_day']) and (not CFG.get('use_max_consecutive_losses',True) or max_streak<=CFG['max_consecutive_losses']))}
    run_hash=digest(json.dumps({'spec':s,'contract':CFG,'data_hash':data_hash},sort_keys=True).encode())
    return {'metrics':metrics,'ledger':ledger,'run_hash_sha256':run_hash}
if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('Usage: python replay.py EXACT_SOURCE.csv')
    result=replay(pd.read_csv(sys.argv[1]))
    if result['metrics']['data_hash_sha256']!=EXPECTED_DATA_SHA256: raise SystemExit('DATA HASH MISMATCH')
    if result['run_hash_sha256']!=EXPECTED_RUN_SHA256: raise SystemExit('RUN HASH MISMATCH')
    print(json.dumps(result,indent=2,ensure_ascii=False))
