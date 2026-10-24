"""Test session, volatility, volume and funding-window filters on the 11-loss candidate.
Real costs retained. Train + development only; OOS untouched.
"""
import asyncio, json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
BASE=("structure_break_after_pullback","ema21_slope5_atr","swing_5",1,1)

def summarize(m):
    keys=("trades","wins","win_rate_pct","profit_factor","tp_per_day","trades_per_day","max_consecutive_losses","net_pnl","final_balance","total_costs","capital_exhausted","monthly_trade_counts","unresolved_ohlc_bars_count")
    return {k:m.get(k) for k in keys}

def masks(df, atr):
    ts=pd.to_datetime(df.open_time.astype("int64"),unit="ms",utc=True)
    hour=ts.dt.hour
    dow=ts.dt.dayofweek
    c=df.close.astype(float); v=df.volume.astype(float)
    atr_pct=atr/c
    rank=atr_pct.rolling(240,min_periods=60).rank(pct=True)
    volmean=v.rolling(20,min_periods=20).mean()
    funding_distance=np.minimum.reduce([np.abs((hour*60+ts.dt.minute).to_numpy()-0),np.abs((hour*60+ts.dt.minute).to_numpy()-480),np.abs((hour*60+ts.dt.minute).to_numpy()-960)])
    base=pd.Series(True,index=df.index)
    out={"no_filter":base}
    # UTC market-activity windows (crypto trades 24/7; these are hypotheses, not market closures).
    out["london_0700_1600_utc"]=((hour>=7)&(hour<16))
    out["new_york_1300_2200_utc"]=((hour>=13)&(hour<22))
    out["london_ny_overlap_1300_1600_utc"]=((hour>=13)&(hour<16))
    out["asia_0000_0800_utc"]=(hour<8)
    out["avoid_low_activity_0000_0600_utc"]=((hour>=6)&(hour<24))
    out["weekdays_only"]=(dow<5)
    out["exclude_weekend"]=(dow<5)
    out["avoid_funding_plusminus_10m"]=pd.Series(funding_distance>10,index=df.index)
    out["avoid_funding_plusminus_20m"]=pd.Series(funding_distance>20,index=df.index)
    out["atr_middle_20_80_pct"]=(rank>=0.2)&(rank<=0.8)
    out["atr_above_20_pct"]=(rank>=0.2)
    out["volume_above_sma20"]=(v>volmean)
    out["volume_above_1_2_sma20"]=(v>1.2*volmean)
    out["london_or_ny"]=(hour.between(7,21))
    out["ny_plus_atr_middle"]=(hour>=13)&(hour<22)&(rank>=0.2)&(rank<=0.8)
    out["london_ny_plus_volume"]=(hour>=7)&(hour<22)&(v>volmean)
    out["weekdays_plus_atr_middle"]=(dow<5)&(rank>=0.2)&(rank<=0.8)
    out["avoid_extreme_atr"]=(rank<=0.9)
    return {k:v.fillna(False).astype(bool) for k,v in out.items()}

def targets(m):
    return (float(m.get("win_rate_pct") or 0)>40 and float(m.get("profit_factor") or 0)>1 and float(m.get("tp_per_day") or 0)>=3 and float(m.get("trades_per_day") or 0)>=10 and int(m.get("max_consecutive_losses") or 0)<=12 and not m.get("capital_exhausted") and bool(m.get("monthly_trade_counts")) and all(int(v)>=300 for v in m["monthly_trade_counts"].values()))

async def main():
    start=time.monotonic()
    df,manifest=await get_data(); splits=split_reference_period(df); sm.CONTRACT=normalize_contract(CFG)
    out={"schema":"11_loss_session_filter_sweep_v1","data":manifest,"baseline_spec":BASE,"oos_evaluated":False,
         "costs":{"commission_pct_per_side":0.04,"slippage_pct_per_side":0.02},"news_filter":"NOT TESTED: no verified historical timestamped news-event dataset is wired into this research runner; no synthetic news labels used.",
         "ambiguity_note":"Same-bar TP/SL resolved conservatively as SL from 1m OHLC; 1-second resolution not implemented. Not final validation.","splits":{}}
    per_split={}
    for sn in ("train","development"):
        part=splits[sn]
        signals,atr=sm.signals(part); filters=sm.confirmation_filters(part)
        raw_l,raw_s=signals[BASE[0]]; fl,fs=filters[BASE[1]]
        long0=raw_l & fl.fillna(False); short0=raw_s & fs.fillna(False)
        filt_masks=masks(part,atr)
        rows=[]
        for name,mask in filt_masks.items():
            # The baseline is exactly the established candidate. All other variants only gate its signal times.
            ls=long0 & mask; ss=short0 & mask
            m,ledger=sm.run_combined(part,ls,ss,atr,BASE[2],loss_cooldown_bars=BASE[3],cooldown_trigger_losses=BASE[4])
            rows.append({"filter":name,"metrics":summarize(m),"ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),"passes_all_targets":targets(m)})
        per_split[sn]={r["filter"]:r for r in rows}
        baseline=per_split[sn]["no_filter"]["metrics"]
        ranked=sorted(rows,key=lambda r:(r["metrics"]["win_rate_pct"]>=25, r["metrics"]["win_rate_pct"],r["metrics"]["trades"]>=300,r["metrics"]["profit_factor"] or 0,r["metrics"]["tp_per_day"],-r["metrics"]["max_consecutive_losses"]),reverse=True)
        out["splits"][sn]={"baseline":baseline,"variants":rows,"best_by_win_rate":ranked[:8]}
        print("FILTER_SPLIT "+json.dumps({"split":sn,"baseline":baseline,"best":ranked[:5]},sort_keys=True),flush=True)
    combined=[]
    for name in per_split["development"]:
        d=per_split["development"][name]["metrics"]; t=per_split["train"][name]["metrics"]
        combined.append({"filter":name,"development":d,"train":t,"passes_all_targets_both_splits":targets(d) and targets(t),
          "win_rate_at_least_25_both_splits":d["win_rate_pct"]>=25 and t["win_rate_pct"]>=25,
          "no_regression_vs_baseline_both_splits":all(
            per_split[s][name]["metrics"]["win_rate_pct"]>=per_split[s]["no_filter"]["metrics"]["win_rate_pct"] and
            (per_split[s][name]["metrics"]["profit_factor"] or 0)>=(per_split[s]["no_filter"]["metrics"]["profit_factor"] or 0) and
            per_split[s][name]["metrics"]["tp_per_day"]>=per_split[s]["no_filter"]["metrics"]["tp_per_day"] and
            per_split[s][name]["metrics"]["trades_per_day"]>=per_split[s]["no_filter"]["metrics"]["trades_per_day"] and
            per_split[s][name]["metrics"]["max_consecutive_losses"]<=per_split[s]["no_filter"]["metrics"]["max_consecutive_losses"] and
            per_split[s][name]["metrics"]["net_pnl"]>=per_split[s]["no_filter"]["metrics"]["net_pnl"]
            for s in ("train","development"))})
    out["cross_split"]=sorted(combined,key=lambda x:(x["passes_all_targets_both_splits"],x["win_rate_at_least_25_both_splits"],x["no_regression_vs_baseline_both_splits"],min(x["development"]["win_rate_pct"],x["train"]["win_rate_pct"]),min(x["development"]["trades"],x["train"]["trades"])),reverse=True)
    out["elapsed_seconds"]=round(time.monotonic()-start,2)
    p=OUT/"11_loss_session_filter_report.json";p.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))
    print("FILTER_FINAL "+json.dumps({"report":str(p),"elapsed_seconds":out["elapsed_seconds"],"cross_split_top":out["cross_split"][:8]},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
