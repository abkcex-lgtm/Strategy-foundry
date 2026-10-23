"""Focused EMA 9/21 + post-cross 1-2-3 test. Real Binance USD-M candles; no OOS selection."""
import asyncio, json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from app.engine import normalize_contract, _size_and_target, _resolve_bar_exit
from app.data import DataIntegrityError
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CONTRACT=normalize_contract({"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","min_win_rate_pct":40,"min_profit_factor":1,"min_tp_per_day":3,"max_consecutive_losses":12,"use_min_win_rate":True,"use_min_profit_factor":True,"use_min_tp_per_day":True,"use_max_consecutive_losses":True,"min_trades_per_day":10,"min_trades_per_month":300,"break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False})
STOPS=("signal_candle","swing_3","swing_5","atr_1.0","atr_1.5","atr_2.0")
ENTRIES=("breakout_123","pullback_reclaim","higher_low_breakout")

def signals(df):
    c=df.close.astype(float); o=df.open.astype(float); h=df.high.astype(float); lo=df.low.astype(float)
    e9=c.ewm(span=9,adjust=False,min_periods=9).mean()
    e21=c.ewm(span=21,adjust=False,min_periods=21).mean()
    up=(e9>e21)&(e9.shift(1)<=e21.shift(1))
    dn=(e9<e21)&(e9.shift(1)>=e21.shift(1))
    recent_up=up.rolling(31,min_periods=1).max().shift(1).fillna(0).astype(bool)
    recent_dn=dn.rolling(31,min_periods=1).max().shift(1).fillna(0).astype(bool)
    # Cross bar itself is excluded: setup must form after the cross and within 30 bars.
    up_age=up.astype(int).rolling(31,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age=dn.astype(int).rolling(31,min_periods=1).sum().shift(1).fillna(0)>0
    three_up=(c>c.shift(1))&(c.shift(1)>c.shift(2))&(c.shift(2)>c.shift(3))
    three_dn=(c<c.shift(1))&(c.shift(1)<c.shift(2))&(c.shift(2)<c.shift(3))
    prev3h=h.shift(1).rolling(3,min_periods=3).max()
    prev3l=lo.shift(1).rolling(3,min_periods=3).min()
    result={}
    result["breakout_123"]=(up_age&(c>e21)&three_up&(c>prev3h), dn_age&(c<e21)&three_dn&(c<prev3l))
    result["pullback_reclaim"]=(up_age&(lo<=e21)&(c>e21)&(c>o), dn_age&(h>=e21)&(c<e21)&(c<o))
    result["higher_low_breakout"]=(up_age&(lo.shift(1)<lo.shift(2))&(lo>lo.shift(1))&(c>h.shift(1))&(c>e21),
                                   dn_age&(h.shift(1)>h.shift(2))&(h<h.shift(1))&(c<lo.shift(1))&(c<e21))
    atr=pd.concat([h-lo,(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    return result,atr

def run_one(df, sig, atr, side, stopmode):
    op=df.open.to_numpy(float); hi=df.high.to_numpy(float); lo=df.low.to_numpy(float); cl=df.close.to_numpy(float); ts=df.open_time.to_numpy(np.int64)
    mask=sig.fillna(False).to_numpy(bool); positions=np.flatnonzero(mask); ledger=[]; bal=1000.; streak=0; maxstreak=0; capital=False; unresolved=0
    i0=max(21,14)
    positions=positions[positions>=i0]
    cursor=0
    while cursor<len(positions):
        i=int(positions[cursor]); cursor+=1
        if i+1>=len(df): break
        entry_i=i+1; entry=float(op[entry_i]); side_long=side=="LONG"
        if stopmode=="signal_candle": stop=float(lo[i] if side_long else hi[i])
        elif stopmode=="swing_3": stop=float(np.min(lo[max(0,i-2):i+1]) if side_long else np.max(hi[max(0,i-2):i+1]))
        elif stopmode=="swing_5": stop=float(np.min(lo[max(0,i-4):i+1]) if side_long else np.max(hi[max(0,i-4):i+1]))
        else:
            mult=float(stopmode.split("_")[1]); a=float(atr.iloc[i])
            if not np.isfinite(a) or a<=0: continue
            stop=entry-a*mult if side_long else entry+a*mult
        if (side_long and stop>=entry) or ((not side_long) and stop<=entry): continue
        if bal<=10+1e-9: capital=True; break
        try: qty,tp,er,xr=_size_and_target(entry,stop,side,10.,CONTRACT)
        except ValueError: continue
        reason=price=exit_t=None; exit_i=None
        for j in range(entry_i,len(df)):
            try: resolution=_resolve_bar_exit(side,float(op[j]),float(hi[j]),float(lo[j]),stop,tp)
            except DataIntegrityError:
                # Strict conservative rule: unresolved same-bar SL+TP becomes a stop.
                resolution=("SL",float(stop)); unresolved+=1
            if resolution is not None:
                reason,price=resolution; exit_t=int(ts[j]); exit_i=j; break
        if exit_i is None: break
        gross=(price-entry)*qty*(1 if side_long else -1); ec=entry*qty*er; xc=price*qty*xr; net=gross-ec-xc; bal+=net
        if reason=="TP": streak=0
        else: streak+=1; maxstreak=max(maxstreak,streak)
        ledger.append({"entry_time":int(ts[entry_i]),"exit_time":exit_t,"side":side,"entry":entry,"stop":stop,"target":tp,"exit":price,"reason":reason,"net":net,"costs":ec+xc,"streak":streak})
        cursor=int(np.searchsorted(positions,exit_i+1,side="left"))
    wins=sum(t["reason"]=="TP" and t["net"]>0 for t in ledger); gp=sum(max(0,t["net"]) for t in ledger); gl=abs(sum(min(0,t["net"]) for t in ledger))
    span=max((int(ts[-1])-int(ts[0]))/86400000,1/1440)
    months={}
    for t in ledger:
        month=pd.to_datetime(t["entry_time"],unit="ms",utc=True).strftime("%Y-%m"); months[month]=months.get(month,0)+1
    metrics={"trades":len(ledger),"wins":wins,"win_rate_pct":100*wins/len(ledger) if ledger else 0.,"profit_factor":gp/gl if gl else (None if gp else 0.),"tp_per_day":wins/span,"trades_per_day":len(ledger)/span,"max_consecutive_losses":maxstreak,"net_pnl":sum(t["net"] for t in ledger),"final_balance":bal,"total_costs":sum(t["costs"] for t in ledger),"monthly_trade_counts":months,"capital_exhausted":capital,"unresolved_ohlc_bars_count":unresolved}
    metrics["passes_all_user_targets"]=(metrics["win_rate_pct"]>40 and (metrics["profit_factor"] or 0)>1 and metrics["tp_per_day"]>=3 and metrics["trades_per_day"]>=10 and maxstreak<=12 and not capital and bool(months) and all(v>=300 for v in months.values()))
    return metrics,ledger

async def main():
    df,manifest=await get_data()
    splits=split_reference_period(df)
    results=[]
    for split in ("train","development"):
        part=splits[split]; sigs,atr=signals(part)
        for ent in ENTRIES:
            for side,idx in (("LONG",0),("SHORT",1)):
                for stop in STOPS:
                    m,ledger=run_one(part,sigs[ent][idx],atr,side,stop)
                    results.append({"split":split,"entry_method":ent,"side":side,"stop_method":stop,"metrics":m,
                                    "ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":")).encode()).hexdigest()})
                    print("EMA123_RESULT "+json.dumps(results[-1],sort_keys=True),flush=True)
    def rank(x):
        m=x["metrics"]
        # No target relaxed; rank diagnostic misses by closest overall gates, then PnL.
        gaps=sum([max(0,40.0001-m["win_rate_pct"])/40, max(0,1.0001-(m["profit_factor"] or 0)),max(0,3-m["tp_per_day"])/3,max(0,10-m["trades_per_day"])/10,max(0,m["max_consecutive_losses"]-12)/12,1 if m["capital_exhausted"] else 0])
        return (not m["passes_all_user_targets"],gaps,-m["net_pnl"],-m["trades"])
    output={"schema":"ema9_21_post_cross_123_v1","data":manifest,"data_scope":"Binance USD-M ETHUSDT 1m Apr-Sep 2026; OOS not evaluated","split_rows":{"train":len(splits["train"]),"development":len(splits["development"]),"oos":len(splits["oos"])},"fixed_contract":CONTRACT,"entry_methods":list(ENTRIES),"stop_methods":list(STOPS),"result_count":len(results),"validated_count":sum(x["metrics"]["passes_all_user_targets"] for x in results),"top_by_split":{},"results":results}
    for split in ("train","development"):
        subset=sorted([x for x in results if x["split"]==split],key=rank)
        output["top_by_split"][split]=subset[:10]
    (OUT/"ema_9_21_123_report.json").write_text(json.dumps(output,indent=2,sort_keys=True))
    print("EMA123_FINAL "+json.dumps({"report":"research-results/ema_9_21_123_report.json","result_count":len(results),"validated_count":output["validated_count"],"top_by_split":{k:[{"entry_method":x["entry_method"],"side":x["side"],"stop_method":x["stop_method"],"metrics":x["metrics"]} for x in v[:5]] for k,v in output["top_by_split"].items()}},sort_keys=True),flush=True)

if __name__=="__main__": asyncio.run(main())
