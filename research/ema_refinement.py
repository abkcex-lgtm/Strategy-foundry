"""Focused EMA refinement: preserve the best 12-loss-streak EMA baseline and test EMA50 confirmation."""
import asyncio, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from research.run_factory import get_data, split_reference_period
from research.ema_9_21_123 import CONTRACT, signals, confirmation_filters, run_combined

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)

def ema_filters(df):
    c=df.close.astype(float)
    e9=c.ewm(span=9,adjust=False,min_periods=9).mean()
    e21=c.ewm(span=21,adjust=False,min_periods=21).mean()
    e50=c.ewm(span=50,adjust=False,min_periods=50).mean()
    return {
      "ema50_trend": ((c>e50)&(e21>e50), (c<e50)&(e21<e50)),
      "ema9_21_50_alignment": ((e9>e21)&(e21>e50), (e9<e21)&(e21<e50)),
      "ema50_slope": (e50>e50.shift(5), e50<e50.shift(5)),
      "ema50_plus_strong_candle": (((c>e50)&(e21>e50)), ((c<e50)&(e21<e50))),
    }

def digest(ledger):
    return hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":")).encode()).hexdigest()

async def main():
    df, manifest = await get_data()
    splits=split_reference_period(df)
    results=[]
    entry_methods=("structure_break_pb8_br15","structure_break_pb10_br15","breakout_123","trend_pullback_reversal")
    stop_methods=("signal_candle","swing_3","swing_5","atr_1.5")
    for split in ("train","development"):
        part=splits[split]
        sigs,atr=signals(part)
        filters=confirmation_filters(part)
        filters.update(ema_filters(part))
        for entry in entry_methods:
            raw_l,raw_s=sigs[entry]
            filter_sets=[
              ("baseline", filters["baseline"][0], filters["baseline"][1]),
              ("strong_candle_close", *filters["strong_candle_close"]),
              ("ema21_slope5_atr", *filters["ema21_slope5_atr"]),
              ("ema50_trend", *filters["ema50_trend"]),
              ("ema9_21_50_alignment", *filters["ema9_21_50_alignment"]),
              ("ema50_slope", *filters["ema50_slope"]),
              ("ema50_plus_strong_candle", *filters["ema50_plus_strong_candle"]),
            ]
            for fname,lf,sf in filter_sets:
                for stop in stop_methods:
                    m,ledger=run_combined(part,raw_l & lf,raw_s & sf,atr,stop)
                    row={"split":split,"entry_method":entry,"filter":fname,"stop_method":stop,
                         "side":"LONG+SHORT_SHARED_SINGLE_POSITION","metrics":m,"ledger_sha256":digest(ledger)}
                    results.append(row)
    # Include the historical 12-streak EMA candidate as a same-code reproducibility baseline.
    # This was the asymmetric long-strong / short-baseline pb8/br15 + swing_5 variant.
    for split in ("train","development"):
        part=splits[split]; sigs,atr=signals(part); filters=confirmation_filters(part)
        raw_l,raw_s=sigs["structure_break_pb8_br15"]
        m,ledger=run_combined(part,raw_l & filters["strong_candle_close"][0],raw_s & filters["baseline"][1],atr,"swing_5")
        results.append({"split":split,"entry_method":"HISTORICAL_BASELINE_pb8_br15","filter":"asym_long-strong_candle_close__short-baseline","stop_method":"swing_5","side":"LONG+SHORT_SHARED_SINGLE_POSITION","metrics":m,"ledger_sha256":digest(ledger)})
    def score(x):
        m=x["metrics"]
        gaps=(max(0,40.0001-m["win_rate_pct"])/40 +
              max(0,1.0001-(m["profit_factor"] or 0)) +
              max(0,3-m["tp_per_day"])/3 +
              max(0,10-m["trades_per_day"])/10 +
              max(0,m["max_consecutive_losses"]-12)/12)
        return (not m["passes_all_user_targets"],gaps,-m["net_pnl"],-m["trades"])
    out={"schema":"focused_ema_9_21_plus_ema50_refinement_v1","data":manifest,
         "data_scope":"Binance USD-M ETHUSDT 1m Apr-Sep 2026; Train and Development only; OOS not evaluated",
         "split_rows":{"train":len(splits["train"]),"development":len(splits["development"]),"oos":len(splits["oos"])},
         "fixed_contract":CONTRACT,"entry_methods":list(entry_methods),"stop_methods":list(stop_methods),
         "tested_filters":["baseline","strong_candle_close","ema21_slope5_atr","ema50_trend","ema9_21_50_alignment","ema50_slope","ema50_plus_strong_candle"],
         "result_count":len(results),"validated_count":sum(x["metrics"]["passes_all_user_targets"] for x in results),
         "top_by_split":{},"results":results}
    for split in ("train","development"):
        subset=sorted((x for x in results if x["split"]==split),key=score)
        out["top_by_split"][split]=subset[:12]
    (OUT/"ema_refinement_report.json").write_text(json.dumps(out,indent=2,sort_keys=True))
    print("EMA_REFINEMENT_FINAL "+json.dumps({"report":"research-results/ema_refinement_report.json","result_count":len(results),"validated_count":out["validated_count"],"top_by_split":{k:[{"entry_method":x["entry_method"],"filter":x["filter"],"stop_method":x["stop_method"],"metrics":x["metrics"]} for x in v[:8]] for k,v in out["top_by_split"].items()}},sort_keys=True),flush=True)

if __name__=="__main__": asyncio.run(main())
