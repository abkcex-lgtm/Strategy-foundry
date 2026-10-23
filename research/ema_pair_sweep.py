"""Fast, causal EMA-pair sweep on Train/Development only; fixed user contract unchanged."""
import asyncio, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from research.run_factory import get_data, split_reference_period
from research.ema_9_21_123 import CONTRACT, signals as base_signals, confirmation_filters, run_combined

OUT=Path("research-results")
OUT.mkdir(exist_ok=True)
PAIRS=((5,13),(7,20),(8,21),(9,21),(9,36),(12,26),(13,34),(20,50))
STOPS=("signal_candle","swing_3","swing_5","atr_1.5")

def pair_signals(df, fast, slow):
    c=df.close.astype(float); o=df.open.astype(float); h=df.high.astype(float); lo=df.low.astype(float)
    ef=c.ewm(span=fast,adjust=False,min_periods=fast).mean()
    es=c.ewm(span=slow,adjust=False,min_periods=slow).mean()
    up=(ef>es)&(ef.shift(1)<=es.shift(1))
    dn=(ef<es)&(ef.shift(1)>=es.shift(1))
    up_age=up.astype(int).rolling(31,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age=dn.astype(int).rolling(31,min_periods=1).sum().shift(1).fillna(0)>0
    up_age10=up.astype(int).rolling(11,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age10=dn.astype(int).rolling(11,min_periods=1).sum().shift(1).fillna(0)>0
    touch_l=(lo<=es).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)
    touch_s=(h>=es).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)
    high2=h.shift(1).rolling(2,min_periods=2).max()
    low2=lo.shift(1).rolling(2,min_periods=2).min()
    high3=h.shift(1).rolling(3,min_periods=3).max()
    low3=lo.shift(1).rolling(3,min_periods=3).min()
    specs={
      "cross_pullback_reversal": (up_age&touch_l&(c>es)&(c>o), dn_age&touch_s&(c<es)&(c<o)),
      "cross_pullback_break2": (up_age10&touch_l&(c>es)&(c>o)&(c>high2), dn_age10&touch_s&(c<es)&(c<o)&(c<low2)),
      "cross_pullback_123": (up_age&touch_l&(c>es)&(c>o)&(c>high3), dn_age&touch_s&(c<es)&(c<o)&(c<low3)),
    }
    return specs

def sha(ledger):
    return hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":")).encode()).hexdigest()

async def main():
    df, manifest=await get_data()
    splits=split_reference_period(df)
    rows=[]
    for split in ("train","development"):
        part=splits[split]
        _,atr=base_signals(part)
        for fast,slow in PAIRS:
            specs=pair_signals(part,fast,slow)
            for entry,(ls,ss) in specs.items():
                for stop in STOPS:
                    m,ledger=run_combined(part,ls,ss,atr,stop)
                    rows.append({"split":split,"ema_fast":fast,"ema_slow":slow,"entry":entry,"stop":stop,
                                 "side":"LONG+SHORT_SHARED_SINGLE_POSITION","metrics":m,"ledger_sha256":sha(ledger)})
        # Same-code historical baseline for an apples-to-apples comparison.
        sigs,atr=base_signals(part); filters=confirmation_filters(part)
        ls,ss=sigs["structure_break_pb8_br15"]
        m,ledger=run_combined(part,ls&filters["strong_candle_close"][0],ss&filters["baseline"][1],atr,"swing_5")
        rows.append({"split":split,"ema_fast":9,"ema_slow":21,"entry":"HISTORICAL_12_STREAK_BASELINE",
                     "stop":"swing_5","side":"LONG+SHORT_SHARED_SINGLE_POSITION","metrics":m,"ledger_sha256":sha(ledger)})
    def rank(x):
        m=x["metrics"]
        gap=(max(0,40.0001-m["win_rate_pct"])/40+
             max(0,1.0001-(m["profit_factor"] or 0))+
             max(0,3-m["tp_per_day"])/3+
             max(0,10-m["trades_per_day"])/10+
             max(0,m["max_consecutive_losses"]-12)/12+
             (2 if m["capital_exhausted"] else 0))
        return (not m["passes_all_user_targets"],gap,-m["net_pnl"],-m["trades"])
    report={"schema":"ema_pair_sweep_fixed_contract_v1","data":manifest,
      "data_scope":"Binance USD-M ETHUSDT 1m Apr-Sep 2026; Train and Development only; OOS untouched",
      "split_rows":{"train":len(splits["train"]),"development":len(splits["development"]),"oos":len(splits["oos"])},
      "fixed_contract":CONTRACT,"ema_pairs":[list(p) for p in PAIRS],"entries":["cross_pullback_reversal","cross_pullback_break2","cross_pullback_123"],
      "stops":list(STOPS),"result_count":len(rows),"validated_count":sum(x["metrics"]["passes_all_user_targets"] for x in rows),
      "lower_timeframe_audit":"NOT CLAIMED: this sweep does not load or hash 1-second candles; unresolved_ohlc_bars_count alone is not proof of lower-timeframe resolution",
      "top_by_split":{},"results":rows}
    for split in ("train","development"):
        subset=sorted((x for x in rows if x["split"]==split),key=rank)
        report["top_by_split"][split]=subset[:15]
    path=OUT/"ema_pair_sweep_report.json"
    path.write_text(json.dumps(report,indent=2,sort_keys=True))
    print("EMA_PAIR_SWEEP_FINAL "+json.dumps({"report":str(path),"result_count":len(rows),"validated_count":report["validated_count"],
      "top_by_split":{k:[{"ema_fast":x["ema_fast"],"ema_slow":x["ema_slow"],"entry":x["entry"],"stop":x["stop"],"metrics":x["metrics"]} for x in v[:10]] for k,v in report["top_by_split"].items()}},sort_keys=True),flush=True)

if __name__=="__main__":
    asyncio.run(main())
