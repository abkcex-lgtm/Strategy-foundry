"""Shared-one-position portfolio test for the historical 11-loss ETHUSDT candidate.
Research only. Real costs, no OOS evaluation. All symbols compete for one $1,000 account and one position.
"""
import asyncio, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import research.ema_9_21_123 as sm
from app.data import download_binance_month, validate_ohlcv, DataIntegrityError
from app.engine import normalize_contract, _size_and_target, _resolve_bar_exit
from research.run_factory import frame_hash

SYMBOLS=["ETHUSDT","BTCUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","LINKUSDT","AVAXUSDT","LTCUSDT"]
MONTHS=[4,5,6,7,8,9]; YEAR=2026
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
     "reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
     "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only",
     "break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
BASE=("structure_break_after_pullback","ema21_slope5_atr","swing_5",1,1)
OUT=Path("research-results"); OUT.mkdir(exist_ok=True)

async def load_symbol(symbol):
    chunks=[]; archives=[]
    for month in MONTHS:
        df,digest,url=await download_binance_month(symbol,"1m",YEAR,month)
        expected={4:43200,5:44640,6:43200,7:44640,8:44640,9:43200}[month]
        if len(df)!=expected: raise RuntimeError(f"{symbol} {YEAR}-{month:02d}: expected {expected} rows, got {len(df)}")
        validate_ohlcv(df,60000,strict_gaps=True)
        chunks.append(df); archives.append({"month":f"{YEAR}-{month:02d}","sha256":digest,"url":url,"rows":len(df)})
    full=pd.concat(chunks,ignore_index=True).sort_values("open_time").reset_index(drop=True)
    validate_ohlcv(full,60000,strict_gaps=True)
    return symbol,full,{"symbol":symbol,"rows":len(full),"dataset_sha256":frame_hash(full),"archives":archives}

def signals_for(df):
    sigs,atr=sm.signals(df); filters=sm.confirmation_filters(df)
    raw_l,raw_s=sigs[BASE[0]]; fl,fs=filters[BASE[1]]
    return (raw_l & fl.fillna(False)).to_numpy(bool),(raw_s & fs.fillna(False)).to_numpy(bool),atr

def summarize(ledger, start_ms, end_ms, balance, capital, unresolved, conflicts, symbol_counts, cooldown_bars=1):
    span=max((end_ms-start_ms)/86400000,1/1440)
    wins=sum(x["reason"]=="TP" and x["net"]>0 for x in ledger)
    vals=[float(x["net"]) for x in ledger]; gp=sum(max(0,x) for x in vals); gl=abs(sum(min(0,x) for x in vals))
    months={}
    for t in ledger:
        month=pd.to_datetime(t["entry_time"],unit="ms",utc=True).strftime("%Y-%m")
        months[month]=months.get(month,0)+1
    return {"trades":len(ledger),"wins":wins,"losses":len(ledger)-wins,"win_rate_pct":100*wins/len(ledger) if ledger else 0.0,
      "profit_factor":gp/gl if gl else (None if gp else 0.0),"tp_per_day":wins/span,"trades_per_day":len(ledger)/span,
      "max_consecutive_losses":max([int(x["streak"]) for x in ledger]+[0]),"net_pnl":sum(vals),"final_balance":balance,
      "total_costs":sum(float(x["costs"]) for x in ledger),"capital_exhausted":capital,"monthly_trade_counts":months,
      "unresolved_ohlc_bars_count":unresolved,"simultaneous_signal_conflicts_skipped":conflicts,
      "symbol_trade_counts":symbol_counts,"span_days":span,"max_positions":1,"cooldown_bars_after_loss":cooldown_bars}

def portfolio_backtest(frames, split_name, ranking="hash"):
    # Signal calculations are per-symbol; portfolio execution is globally chronological.
    prepared={}
    for sym,df in frames.items():
        l,s,atr=signals_for(df)
        prepared[sym]={"df":df.reset_index(drop=True),"long":l,"short":s,"atr":atr.reset_index(drop=True),
          "index":{int(t):i for i,t in enumerate(df.open_time.to_numpy(np.int64))}}
    common=sorted(set.intersection(*(set(x["index"]) for x in prepared.values())))
    if not common: raise RuntimeError("No common timestamps across portfolio symbols")
    start_ms,end_ms=common[0],common[-1]
    bal=1000.; streak=0; maxstreak=0; capital=False; unresolved=0; conflicts=0
    ledger=[]; symbol_counts={s:0 for s in frames}; pos=None; pending=None; cooldown_until=-1
    for gi,ts in enumerate(common):
        # A signal at t-1 enters at this candle's open. One global position only.
        if pending is not None and pos is None and not capital:
            sym,side,stop,signal_time=pending; row=prepared[sym]; i=row["index"][ts]; df=row["df"]
            entry=float(df.open.iloc[i])
            valid=(stop<entry if side=="LONG" else stop>entry)
            if valid and bal>10+1e-9:
                try:
                    qty,tp,er,xr=_size_and_target(entry,stop,side,10.,sm.CONTRACT)
                    pos={"symbol":sym,"side":side,"entry":entry,"stop":stop,"target":tp,"qty":qty,"entry_rate":er,"exit_rate":xr,
                         "entry_time":ts,"signal_time":signal_time}
                except ValueError: pass
            elif bal<=10+1e-9: capital=True
            pending=None
        # Evaluate existing position against its own symbol's current candle.
        if pos is not None:
            sym=pos["symbol"]; row=prepared[sym]; i=row["index"][ts]; df=row["df"]
            try: resolution=_resolve_bar_exit(pos["side"],float(df.open.iloc[i]),float(df.high.iloc[i]),float(df.low.iloc[i]),pos["stop"],pos["target"])
            except DataIntegrityError:
                resolution=("SL",float(pos["stop"])); unresolved+=1
            if resolution is not None:
                reason,price=resolution
                gross=(price-pos["entry"])*pos["qty"]*(1 if pos["side"]=="LONG" else -1)
                ec=pos["entry"]*pos["qty"]*pos["entry_rate"]; xc=price*pos["qty"]*pos["exit_rate"]
                net=gross-ec-xc; bal+=net
                if reason=="TP": streak=0
                else: streak+=1; maxstreak=max(maxstreak,streak); cooldown_until=gi+1
                ledger.append({"symbol":sym,"signal_time":pos["signal_time"],"entry_time":pos["entry_time"],"exit_time":ts,
                  "side":pos["side"],"entry":pos["entry"],"stop":pos["stop"],"target":pos["target"],"exit":price,
                  "reason":reason,"net":net,"costs":ec+xc,"streak":streak})
                symbol_counts[sym]+=1; pos=None
                if bal<=10+1e-9: capital=True
        # At close t, select at most one candidate for entry at t+1 open.
        if pos is None and pending is None and not capital and gi>cooldown_until:
            candidates=[]
            for sym,row in prepared.items():
                i=row["index"][ts]
                if i<21 or i+1>=len(row["df"]): continue
                is_l=bool(row["long"][i]); is_s=bool(row["short"][i])
                if is_l and is_s: conflicts+=1; continue
                if not (is_l or is_s): continue
                side="LONG" if is_l else "SHORT"; df=row["df"]
                if side=="LONG": stop=float(df.low.iloc[max(0,i-4):i+1].min())
                else: stop=float(df.high.iloc[max(0,i-4):i+1].max())
                # Deterministic hash tie-break avoids always privileging one symbol when signals coincide.
                tie=hashlib.sha256(f"{ts}:{sym}".encode()).hexdigest()
                if ranking == "hash":
                    score=0.0
                else:
                    lookback=int(ranking.replace("mom",""))
                    if i < lookback or not np.isfinite(float(row["atr"].iloc[i])) or float(row["atr"].iloc[i]) <= 0:
                        score=-1e9
                    else:
                        direction=1.0 if side=="LONG" else -1.0
                        score=direction*(float(df.close.iloc[i])-float(df.close.iloc[i-lookback]))/float(row["atr"].iloc[i])
                candidates.append((score,tie,sym,side,stop))
            if candidates:
                if ranking == "hash":
                    _,_,sym,side,stop=min(candidates, key=lambda x:x[1])
                else:
                    _,_,sym,side,stop=max(candidates, key=lambda x:(x[0],x[1]))
                pending=(sym,side,stop,ts)
    metrics=summarize(ledger,start_ms,end_ms,bal,capital,unresolved,conflicts,symbol_counts)
    return metrics,ledger

async def main():
    started=time.monotonic(); sm.CONTRACT=normalize_contract(CFG)
    loaded=await asyncio.gather(*(load_symbol(s) for s in SYMBOLS))
    all_frames={s:df for s,df,_ in loaded}; manifests={s:m for s,_,m in loaded}
    splits={s:sm_split(df) for s,df in all_frames.items()}
    output={"schema":"eleven_loss_10_symbol_shared_single_position_v1","symbols":SYMBOLS,"baseline_spec":BASE,
      "contract":CFG,"data_manifests":manifests,"oos_evaluated":False,
      "selection_policy":"If multiple symbols signal at the same close, deterministic SHA-256(timestamp,symbol) tie-break; no concurrent positions.",
      "ambiguity_note":"Same-bar SL+TP conservatively counted as SL from 1m OHLC. 1-second resolution not implemented; exploratory only.",
      "splits":{}}
    policies=["hash","mom3","mom5","mom10"]
    selected_policy=None
    for split_name in ("train","development"):
        per={s:splits[s][split_name] for s in SYMBOLS}
        eth=per["ETHUSDT"]; l,s,atr=signals_for(eth)
        base_metrics,base_ledger=sm.run_combined(eth,pd.Series(l,index=eth.index),pd.Series(s,index=eth.index),atr,BASE[2],
              loss_cooldown_bars=BASE[3],cooldown_trigger_losses=BASE[4])
        policy_results={}
        for policy in policies:
            pm,pl=portfolio_backtest(per,split_name,ranking=policy)
            policy_results[policy]={"metrics":pm,"ledger_sha256":hashlib.sha256(json.dumps(pl,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),"ledger":pl}
        if split_name=="train":
            def score(item):
                name,v=item; m=v["metrics"]
                meets=(m["tp_per_day"]>=3 and m["win_rate_pct"]>40 and (m["profit_factor"] or 0)>1 and m["max_consecutive_losses"]<13 and m["trades_per_day"]>=10)
                return (int(meets),min(m["tp_per_day"]/3,1.0),m["profit_factor"] or 0,m["win_rate_pct"],-m["max_consecutive_losses"],m["trades_per_day"])
            selected_policy=max(policy_results.items(),key=score)[0]
        best=policy_results[selected_policy]
        portfolio_metrics=best["metrics"]; portfolio_ledger=best["ledger"]
        output["splits"][split_name]={"eth_single_symbol_baseline":base_metrics,"ten_symbol_one_position":portfolio_metrics,
          "selected_policy":selected_policy,"all_policy_metrics":{k:v["metrics"] for k,v in policy_results.items()},
          "delta_trades":portfolio_metrics["trades"]-base_metrics["trades"],
          "delta_win_rate_percentage_points":portfolio_metrics["win_rate_pct"]-base_metrics["win_rate_pct"],
          "baseline_ledger_sha256":hashlib.sha256(json.dumps(base_ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
          "portfolio_ledger_sha256":best["ledger_sha256"],"portfolio_ledger":portfolio_ledger}
        print("MULTISYMBOL_RESULT "+json.dumps({"split":split_name,"selected_policy":selected_policy,"baseline":base_metrics,
          "portfolio":portfolio_metrics,"all_policy_metrics":{k:v["metrics"] for k,v in policy_results.items()},
          "delta_trades":output["splits"][split_name]["delta_trades"],"delta_win_rate_pp":output["splits"][split_name]["delta_win_rate_percentage_points"]},sort_keys=True),flush=True)
    output["elapsed_seconds"]=round(time.monotonic()-started,2)
    (OUT/"11_loss_10_symbol_shared_position_report.json").write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False))
    print("MULTISYMBOL_FINAL "+json.dumps({"report":"research-results/11_loss_10_symbol_shared_position_report.json","elapsed_seconds":output["elapsed_seconds"],
      "summary":{k:{"baseline":v["eth_single_symbol_baseline"],"portfolio":v["ten_symbol_one_position"],
      "delta_trades":v["delta_trades"],"delta_win_rate_pp":v["delta_win_rate_percentage_points"]} for k,v in output["splits"].items()}},sort_keys=True),flush=True)

def sm_split(df):
    from app.data import split_reference_period
    return split_reference_period(df)

if __name__=="__main__": asyncio.run(main())
