"""Compare the exact ETH 11-loss baseline with real-cost break-even at +1.5R net.
Research-only; train/development only; OOS is never used for selection.
"""
import asyncio, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import research.ema_9_21_123 as sm
from app.engine import normalize_contract, _size_and_target, _resolve_bar_exit
from app.data import DataIntegrityError
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
ENTRY="structure_break_after_pullback"; FILTER="ema21_slope5_atr"; STOP="swing_5"; COOLDOWN_BARS=1; COOLDOWN_TRIGGER=1
TRIGGER_NET_R=1.5

def metrics(ledger, ts, bal, capital, unresolved):
    wins=sum(t["reason"]=="TP" and t["net"]>0 for t in ledger)
    gp=sum(max(0,t["net"]) for t in ledger); gl=abs(sum(min(0,t["net"]) for t in ledger))
    span=max((int(ts[-1])-int(ts[0]))/86400000,1/1440)
    months={}
    for t in ledger:
        month=pd.to_datetime(t["entry_time"],unit="ms",utc=True).strftime("%Y-%m")
        months[month]=months.get(month,0)+1
    n=len(ledger)
    return {"trades":n,"wins":wins,"win_rate_pct":100*wins/n if n else 0.,
      "profit_factor":gp/gl if gl else (None if gp else 0.),"tp_per_day":wins/span,
      "trades_per_day":n/span,"max_consecutive_losses":max((t["loss_streak"] for t in ledger),default=0),
      "net_pnl":sum(t["net"] for t in ledger),"final_balance":bal,
      "total_costs":sum(t["costs"] for t in ledger),"monthly_trade_counts":months,
      "capital_exhausted":capital,"unresolved_ohlc_bars_count":unresolved,
      "break_even_stops":sum(t.get("break_even_activated",False) for t in ledger),
      "break_even_exits":sum(t["reason"]=="BE" for t in ledger)}

def simulate(df, long_sig, short_sig, atr, use_be):
    op=df.open.to_numpy(float); hi=df.high.to_numpy(float); lo=df.low.to_numpy(float); ts=df.open_time.to_numpy(np.int64)
    lm=long_sig.fillna(False).to_numpy(bool); sh=short_sig.fillna(False).to_numpy(bool)
    sigidx=np.flatnonzero(lm|sh); sigidx=sigidx[sigidx>=21]
    ledger=[]; bal=1000.; streak=0; maxstreak=0; capital=False; unresolved=0; conflicts=0
    cursor=0; cooldown_until=-1
    while cursor<len(sigidx):
        i=int(sigidx[cursor]); cursor+=1
        if i<=cooldown_until: continue
        is_long=bool(lm[i]); is_short=bool(sh[i])
        if is_long and is_short: conflicts+=1; continue
        side="LONG" if is_long else "SHORT"; long=side=="LONG"
        if i+1>=len(df): break
        ei=i+1; entry=float(op[ei])
        stop=float(np.min(lo[max(0,i-4):i+1]) if long else np.max(hi[max(0,i-4):i+1]))
        if (long and stop>=entry) or ((not long) and stop<=entry): continue
        if bal<=10+1e-9: capital=True; break
        try: qty,tp,er,xr=_size_and_target(entry,stop,side,10.,sm.CONTRACT)
        except ValueError: continue
        # Net-real +1.5R trigger and cost-neutral stop include entry/exit costs.
        if long:
            trigger=(entry*(1+er)+TRIGGER_NET_R*10/qty)/(1-xr)
            be_stop=entry*(1+er)/(1-xr)
        else:
            trigger=(entry*(1-er)-TRIGGER_NET_R*10/qty)/(1+xr)
            be_stop=entry*(1-er)/(1+xr)
        active_stop=stop; be_active=False; reason=price=exit_t=None; exit_i=None
        for j in range(ei,len(df)):
            try: res=_resolve_bar_exit(side,float(op[j]),float(hi[j]),float(lo[j]),active_stop,tp)
            except DataIntegrityError:
                res=("SL",float(active_stop)); unresolved+=1
            if res is not None:
                reason,price=res; exit_t=int(ts[j]); exit_i=j
                # A stop moved to the true-cost breakeven is not a loss unless slippage/gap changes realized net.
                if reason=="SL" and be_active:
                    reason="BE"
                break
            if use_be and not be_active:
                touched=(hi[j]>=trigger) if long else (lo[j]<=trigger)
                if touched:
                    # Stop change becomes active next bar; 1m OHLC cannot prove intrabar ordering.
                    active_stop=be_stop; be_active=True
        if exit_i is None: break
        gross=(price-entry)*qty*(1 if long else -1); ec=entry*qty*er; xc=price*qty*xr; net=gross-ec-xc; bal+=net
        # classify a cost-neutral exit as BE only if the realized net is near zero;
        # gap-throughs are recorded by their actual realized PnL.
        if reason=="BE" and net < -0.05: reason="SL"
        if reason=="BE" and net > 0.05: reason="BE"
        if reason=="TP": streak=0
        elif net < -0.05:
            streak+=1; maxstreak=max(maxstreak,streak)
            if COOLDOWN_BARS and streak>=COOLDOWN_TRIGGER: cooldown_until=max(cooldown_until,exit_i+COOLDOWN_BARS)
        # Small positive/negative breakeven fills don't reset the streak; TP alone resets it.
        ledger.append({"entry_time":int(ts[ei]),"signal_time":int(ts[i]),"exit_time":exit_t,"side":side,
          "entry":entry,"initial_stop":stop,"effective_stop":active_stop,"breakeven_price":be_stop,
          "trigger_price":trigger,"target":tp,"exit":price,"reason":reason,"net":net,"costs":ec+xc,
          "loss_streak":streak,"break_even_activated":be_active})
        cursor=int(np.searchsorted(sigidx,exit_i+1,side="left"))
    m=metrics(ledger,ts,bal,capital,unresolved)
    m["simultaneous_signal_conflicts_skipped"]=conflicts
    return m,ledger

async def main():
    started=time.monotonic(); sm.CONTRACT=normalize_contract(CFG)
    df,manifest=await get_data(); splits=split_reference_period(df)
    report={"schema":"eth_11_loss_real_cost_break_even_15R_v1","data_manifest":manifest,"contract":CFG,
      "candidate":{"entry":ENTRY,"filter":FILTER,"stop":STOP,"cooldown_bars":COOLDOWN_BARS,
        "cooldown_trigger_losses":COOLDOWN_TRIGGER,"trigger":"net unrealized profit after entry and estimated exit costs >= $15",
        "stop":"exact cost-neutral exit price including entry and exit costs",
        "ambiguity":"1m OHLC only; ambiguous stop/target resolves conservatively to stop; after trigger stop change takes effect next candle because intrabar order is unknown",
        "oos_evaluated":False},"splits":{}}
    for name in ("train","development"):
        part=splits[name]; sigs,atr=sm.signals(part); filters=sm.confirmation_filters(part)
        l,s=sigs[ENTRY]; lf,sf=filters[FILTER]
        ls=pd.Series(l & lf.fillna(False),index=part.index); ss=pd.Series(s & sf.fillna(False),index=part.index)
        base,baseledger=simulate(part,ls,ss,atr,False)
        be,beledger=simulate(part,ls,ss,atr,True)
        report["splits"][name]={"baseline":base,"risk_free_1_5R":be,
          "delta":{"trades":be["trades"]-base["trades"],"wins":be["wins"]-base["wins"],
            "win_rate_pct":be["win_rate_pct"]-base["win_rate_pct"],"profit_factor":(be["profit_factor"] or 0)-(base["profit_factor"] or 0),
            "tp_per_day":be["tp_per_day"]-base["tp_per_day"],"trades_per_day":be["trades_per_day"]-base["trades_per_day"],
            "max_consecutive_losses":be["max_consecutive_losses"]-base["max_consecutive_losses"],
            "net_pnl":be["net_pnl"]-base["net_pnl"],"total_costs":be["total_costs"]-base["total_costs"],
            "final_balance":be["final_balance"]-base["final_balance"]},
          "baseline_ledger_sha256":hashlib.sha256(json.dumps(baseledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
          "risk_free_ledger_sha256":hashlib.sha256(json.dumps(beledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()}
    report["elapsed_seconds"]=round(time.monotonic()-started,2)
    p=OUT/"11_loss_eth_real_cost_break_even_report.json"; p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    print("REAL_COST_BE_FINAL "+json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
