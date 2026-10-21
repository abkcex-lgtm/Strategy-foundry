from __future__ import annotations
import asyncio, hashlib, json, gzip
from pathlib import Path
import pandas as pd
from app.data import download_binance_month, validate_ohlcv, split_reference_period
from app.engine import StrategySpec, normalize_contract, prepare_backtest_data, backtest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research-results"; OUT.mkdir(exist_ok=True)
YEAR=2026; MONTHS=[4,5,6,7,8,9]
EXPECTED={4:43200,5:44640,6:43200,7:44640,8:44640,9:43200}
def sha(b): return hashlib.sha256(b).hexdigest()
def frame_hash(df): return sha(df.to_csv(index=False,float_format="%.12g").encode())
async def load_data():
    chunks=[]; archives=[]
    for m in MONTHS:
        print(f"DATA_START 2026-{m:02d}",flush=True)
        part,h,url=await download_binance_month("ETHUSDT","1m",YEAR,m)
        if len(part)!=EXPECTED[m]: raise RuntimeError(f"Expected {EXPECTED[m]} rows for 2026-{m:02d}; got {len(part)}")
        validate_ohlcv(part,60000,strict_gaps=True)
        chunks.append(part); archives.append({"month":f"2026-{m:02d}","rows":len(part),"archive_sha256":h,"url":url})
        print(f"DATA_OK 2026-{m:02d} rows={len(part)}",flush=True)
    df=pd.concat(chunks,ignore_index=True).sort_values("open_time").reset_index(drop=True)
    integrity=validate_ohlcv(df,60000,strict_gaps=True)
    if len(df)!=263520: raise RuntimeError(f"Expected 263520 rows, got {len(df)}")
    expected_start=pd.Timestamp("2026-04-01T00:00:00Z").value//1_000_000
    expected_end=pd.Timestamp("2026-09-30T23:59:00Z").value//1_000_000
    if integrity["first_open_time"]!=expected_start or integrity["last_open_time"]!=expected_end:
        raise RuntimeError("Dataset endpoints do not match the agreed Apr-Sep 2026 window")
    return df,{"symbol":"ETHUSDT","interval":"1m","rows":len(df),"dataset_sha256":frame_hash(df),"monthly_archives":archives,"integrity":integrity}
def summarize(run):
    m=run["metrics"]; ledger=run["ledger"]
    sides={}
    for side in ("LONG","SHORT"):
        ts=[t for t in ledger if t["direction"]==side]
        wins=sum(t["exit_reason"]=="TP" and t["net_pnl"]>0 for t in ts)
        gp=sum(max(0.0,t["net_pnl"]) for t in ts); gl=abs(sum(min(0.0,t["net_pnl"]) for t in ts))
        sides[side]={"trades":len(ts),"wins":wins,"losses":sum(t["net_pnl"]<=0 for t in ts),
                     "win_rate_pct":100*wins/len(ts) if ts else 0.0,
                     "profit_factor":gp/gl if gl else (None if gp else 0.0),
                     "net_pnl":sum(t["net_pnl"] for t in ts)}
    return {"metrics":m,"direction_breakdown":sides,"ledger_sha256":sha(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()),
            "run_hash_sha256":run["run_hash_sha256"]}
async def main():
    df,data=await load_data()
    splits=split_reference_period(df)
    expected={"train":131040,"development":44640,"oos":87840}
    for k,n in expected.items():
        if len(splits[k])!=n: raise RuntimeError(f"Bad {k} split size: {len(splits[k])} != {n}")
    cfg=normalize_contract({"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
       "reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
       "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","min_win_rate_pct":40,
       "min_profit_factor":1,"min_tp_per_day":3,"min_trades_per_day":8,"max_consecutive_losses":13,
       "use_min_win_rate":True,"use_min_profit_factor":True,"use_min_tp_per_day":True,"use_max_consecutive_losses":True,
       "break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False})
    # Exact candidate genome from the prior run; only direction varies.
    base={"name":"Momentum-confirmed breakout OR Volatility squeeze breakout 14/40 ATR 7x4",
          "fast":14,"slow":40,"atr_period":7,"stop_atr":4.0,"seed":1000045,
          "family":"momentum_breakout","context_filter":"none","confirmation_filter":"none",
          "secondary_family":"atr_squeeze_breakout","signal_composition":"or"}
    result={"experiment":"fixed_candidate_direction_comparison","data":data,
            "strategy_genome":base,"contract":cfg,"split_periods":{"train":"2026-04-01 through 2026-06-30","development":"2026-07-01 through 2026-07-31","oos":"2026-08-01 through 2026-09-30"},
            "results":{},"note":"No thresholds or trade rules changed. OOS was not used for selection. Existing engine auto-resolves ambiguous bars with 1s data when available; unresolved cases are treated as losses."}
    prepared={k:prepare_backtest_data(v) for k,v in splits.items()}
    for direction in ("long","short","both"):
        spec=StrategySpec(**base,direction=direction)
        result["results"][direction]={}
        for split_name in ("train","development","oos"):
            run=backtest(splits[split_name],spec,cfg,_prepared_context=prepared[split_name])
            result["results"][direction][split_name]=summarize(run)
            m=run["metrics"]
            print(f"RESULT direction={direction} split={split_name} trades={m['trades']} wr={m['win_rate_pct']:.4f} pf={m['profit_factor']} tp_day={m['avg_net_tp_per_day']:.4f} trades_day={m['trades_per_day']:.4f} max_cl={m['max_consecutive_losses']} pnl={m['net_pnl']:.2f} balance={m['final_balance']:.2f}",flush=True)
    path=OUT/"direction-comparison.json"
    path.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
    print("DIRECTION_REPORT "+str(path),flush=True)
if __name__=="__main__": asyncio.run(main())
