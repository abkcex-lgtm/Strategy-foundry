"""Multi-timeframe trend + pullback continuation hypothesis, compared with historical 11-loss reference.
Uses real Binance ETHUSDT 1m candles; Train and Development only. OOS is never evaluated.
"""
import asyncio, hashlib, json, time, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
import research.ema_9_21_123 as sm
from research.run_factory import get_data, split_reference_period
from app.engine import normalize_contract

sm.CONTRACT=normalize_contract({"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False})
OUT=ROOT/"research-results"; OUT.mkdir(exist_ok=True)

def new_idea(df):
    c=df.close.astype(float); o=df.open.astype(float); h=df.high.astype(float); lo=df.low.astype(float)
    e9=c.ewm(span=9,adjust=False,min_periods=9).mean()
    e21=c.ewm(span=21,adjust=False,min_periods=21).mean()
    tr=pd.concat([h-lo,(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    # Causal 15-minute regime: only completed 15m bars can influence 1m signals.
    tmp=pd.DataFrame({"open_time":pd.to_datetime(df.open_time,unit="ms",utc=True),"open":o.to_numpy(),"high":h.to_numpy(),"low":lo.to_numpy(),"close":c.to_numpy()}).set_index("open_time")
    htf=tmp.resample("15min",label="right",closed="left").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    htf_fast=htf.close.ewm(span=20,adjust=False,min_periods=20).mean()
    htf_slow=htf.close.ewm(span=50,adjust=False,min_periods=50).mean()
    # htf values labeled at the close timestamp, forward-fill only onto 1m bars at/after that timestamp.
    t=pd.to_datetime(df.open_time,unit="ms",utc=True)
    regime_up=pd.Series(htf_fast-htf_slow,index=htf.index).reindex(t,method="ffill").to_numpy()
    htf_up=pd.Series(regime_up>0,index=df.index)
    htf_dn=pd.Series(regime_up<0,index=df.index)
    # Require a pullback touch in the previous 5 completed bars, then reclaim with a 1m micro-breakout.
    touched_up=(lo<=e21).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)
    touched_dn=(h>=e21).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)
    prior_high=h.shift(1).rolling(2,min_periods=2).max()
    prior_low=lo.shift(1).rolling(2,min_periods=2).min()
    rng=(h-lo).replace(0,np.nan)
    body=(c-o).abs()/rng
    close_pos=(c-lo)/rng
    long=(htf_up&(e9>e21)&(e21>e21.shift(3))&touched_up&(c>o)&(c>prior_high)&(c>e21)&(body>=0.45)&(close_pos>=0.7))
    short=(htf_dn&(e9<e21)&(e21<e21.shift(3))&touched_dn&(c<o)&(c<prior_low)&(c<e21)&(body>=0.45)&(close_pos<=0.3))
    return long.fillna(False),short.fillna(False),atr

def brief(m):
    return {k:m.get(k) for k in ("trades","wins","win_rate_pct","profit_factor","tp_per_day","trades_per_day","max_consecutive_losses","net_pnl","final_balance","total_costs","monthly_trade_counts","capital_exhausted","passes_all_user_targets")}

async def main():
    t0=time.monotonic(); df,manifest=await get_data(); splits=split_reference_period(df)
    report={"strategy":"15m EMA20/50 regime + 1m EMA9/21 slope + 5-bar EMA21 pullback + 2-bar breakout and strong close","oos_evaluated":False,"oos_used_for_selection":False,"data_manifest":manifest,"splits":{}}
    for name in ("train","development"):
        part=splits[name]
        sigs,base_atr=sm.signals(part); filters=sm.confirmation_filters(part)
        bl,bs=sigs["structure_break_after_pullback"]; lf,sf=filters["ema21_slope5_atr"]
        base_m,base_ledger=sm.run_combined(part,bl&lf.fillna(False),bs&sf.fillna(False),base_atr,"swing_5",loss_cooldown_bars=1,cooldown_trigger_losses=1)
        nl,ns,atr=new_idea(part)
        new_m,new_ledger=sm.run_combined(part,nl,ns,atr,"swing_3")
        report["splits"][name]={"reference_11_loss_baseline":{"metrics":brief(base_m),"trades":len(base_ledger),"ledger_sha256":hashlib.sha256(json.dumps(base_ledger,sort_keys=True,separators=(",",":")).encode()).hexdigest()},
          "multitimeframe_pullback":{"metrics":brief(new_m),"trades":len(new_ledger),"long_signals":int(nl.sum()),"short_signals":int(ns.sum()),"ledger_sha256":hashlib.sha256(json.dumps(new_ledger,sort_keys=True,separators=(",",":")).encode()).hexdigest()}}
        OUT.joinpath("multitimeframe_pullback_report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
        print("MTF_SPLIT "+json.dumps({"split":name,"reference":brief(base_m),"new_idea":brief(new_m),"long_signals":int(nl.sum()),"short_signals":int(ns.sum())},sort_keys=True),flush=True)
    report["elapsed_seconds"]=round(time.monotonic()-t0,2)
    OUT.joinpath("multitimeframe_pullback_report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    print("MTF_FINAL "+json.dumps({"elapsed_seconds":report["elapsed_seconds"],"oos_evaluated":False,"splits":{k:{"reference":v["reference_11_loss_baseline"]["metrics"],"new_idea":v["multitimeframe_pullback"]["metrics"]} for k,v in report["splits"].items()}},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
