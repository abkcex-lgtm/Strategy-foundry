"""Expanded fast sweep: preserve EMA-pair baseline and test structure-break confirmations/stops."""
import asyncio, json, hashlib
from pathlib import Path
from research.run_factory import get_data, split_reference_period
from research.ema_9_21_123 import CONTRACT, signals as base_signals, confirmation_filters, run_combined

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
PAIRS=((5,13),(7,20),(8,21),(9,21),(9,36),(12,26),(13,34),(20,50))
STOPS=("signal_candle","swing_3","swing_5","atr_1.5")
STRUCTURES=("structure_break_after_pullback","structure_break_pb3_br5","structure_break_pb3_br10",
 "structure_break_pb3_br15","structure_break_pb5_br10","structure_break_pb5_br15","structure_break_pb5_br20",
 "structure_break_pb8_br10","structure_break_pb8_br15","structure_break_pb8_br20","structure_break_pb10_br15","structure_break_pb10_br20")
# Directional filter pairs chosen before seeing this run; include the exact historical candidate.
FILTER_PAIRS=(
 ("baseline","baseline"),("strong_candle_close","baseline"),("baseline","strong_candle_close"),
 ("strong_candle_close","strong_candle_close"),("adx22_directional","baseline"),("baseline","adx22_directional"),
 ("ema21_slope5_atr","baseline"),("baseline","ema21_slope5_atr"),
 ("rsi_tight_momentum","baseline"),("baseline","rsi_tight_momentum"),
 ("avoid_extended_entry","baseline"),("baseline","avoid_extended_entry"),
 ("distance_to_ema_atr_1_0","baseline"),("baseline","distance_to_ema_atr_1_0"),
 ("slope_plus_adx","baseline"),("baseline","slope_plus_adx"),
 ("slope_adx_candle_stack","baseline"),("baseline","slope_adx_candle_stack"),
 ("trend_quality_stack","baseline"),("baseline","trend_quality_stack"),
 ("volume_above_sma20","baseline"),("baseline","volume_above_sma20"),
 ("rsi_tight_momentum","strong_candle_close"),("strong_candle_close","rsi_tight_momentum"),
 ("adx22_directional","strong_candle_close"),("strong_candle_close","adx22_directional"),
 ("trend_quality_stack","strong_candle_close"),("strong_candle_close","trend_quality_stack"),
 ("distance_to_ema_atr_0_75","baseline"),("baseline","distance_to_ema_atr_0_75"),
)
def pair_signals(df, fast, slow):
    c=df.close.astype(float); o=df.open.astype(float); h=df.high.astype(float); lo=df.low.astype(float)
    ef=c.ewm(span=fast,adjust=False,min_periods=fast).mean()
    es=c.ewm(span=slow,adjust=False,min_periods=slow).mean()
    up=(ef>es)&(ef.shift(1)<=es.shift(1)); dn=(ef<es)&(ef.shift(1)>=es.shift(1))
    up_age=up.astype(int).rolling(31,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age=dn.astype(int).rolling(31,min_periods=1).sum().shift(1).fillna(0)>0
    up_age10=up.astype(int).rolling(11,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age10=dn.astype(int).rolling(11,min_periods=1).sum().shift(1).fillna(0)>0
    touch_l=(lo<=es).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)
    touch_s=(h>=es).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)
    high2=h.shift(1).rolling(2,min_periods=2).max(); low2=lo.shift(1).rolling(2,min_periods=2).min()
    high3=h.shift(1).rolling(3,min_periods=3).max(); low3=lo.shift(1).rolling(3,min_periods=3).min()
    return {
      "cross_pullback_reversal":(up_age&touch_l&(c>es)&(c>o),dn_age&touch_s&(c<es)&(c<o)),
      "cross_pullback_break2":(up_age10&touch_l&(c>es)&(c>o)&(c>high2),dn_age10&touch_s&(c<es)&(c<o)&(c<low2)),
      "cross_pullback_123":(up_age&touch_l&(c>es)&(c>o)&(c>high3),dn_age&touch_s&(c<es)&(c<o)&(c<low3)),
    }
def sha(ledger):
    return hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def pack(split,entry,fast,slow,stop,lf,sf,m,ledger):
    return {"split":split,"ema_fast":fast,"ema_slow":slow,"entry":entry,"stop":stop,
      "filter_long":lf,"filter_short":sf,"side":"LONG+SHORT_SHARED_SINGLE_POSITION",
      "metrics":m,"ledger_sha256":sha(ledger)}
async def main():
    df,manifest=await get_data(); splits=split_reference_period(df); rows=[]
    for split in ("train","development"):
        part=splits[split]; _,atr=base_signals(part); filters=confirmation_filters(part)
        # Previous pair sweep retained for apples-to-apples comparisons.
        for fast,slow in PAIRS:
            for entry,(ls,ss) in pair_signals(part,fast,slow).items():
                for stop in STOPS:
                    m,ledger=run_combined(part,ls,ss,atr,stop)
                    rows.append(pack(split,entry,fast,slow,stop,"none","none",m,ledger))
        # Focused improvement search around the best max-loss-streak candidate.
        sigs,_=base_signals(part)
        for entry in STRUCTURES:
            ls0,ss0=sigs[entry]
            for lf,sf in FILTER_PAIRS:
                lf_mask=filters[lf][0]; sf_mask=filters[sf][1]
                ls=ls0 & lf_mask; ss=ss0 & sf_mask
                for stop in STOPS:
                    m,ledger=run_combined(part,ls,ss,atr,stop)
                    rows.append(pack(split,entry,9,21,stop,lf,sf,m,ledger))
        # Exact historical baseline, tagged separately and never omitted.
        ls0,ss0=sigs["structure_break_pb8_br15"]
        m,ledger=run_combined(part,ls0&filters["strong_candle_close"][0],ss0&filters["baseline"][1],atr,"swing_5")
        rows.append(pack(split,"HISTORICAL_12_STREAK_BASELINE",9,21,"swing_5","strong_candle_close","baseline",m,ledger))
    def rank(x):
        m=x["metrics"]
        gap=(max(0,40.0001-m["win_rate_pct"])/40+
             max(0,1.0001-(m["profit_factor"] or 0))+
             max(0,3-m["tp_per_day"])/3+
             max(0,10-m["trades_per_day"])/10+
             max(0,m["max_consecutive_losses"]-12)/12+
             (2 if m["capital_exhausted"] else 0))
        return (not m["passes_all_user_targets"],gap, bool(m["capital_exhausted"]),-m["net_pnl"],-m["trades"])
    report={"schema":"ema_structure_confirmation_expanded_v2","data":manifest,
      "data_scope":"Binance USD-M ETHUSDT 1m Apr-Sep 2026; Train and Development only; OOS untouched",
      "split_rows":{"train":len(splits["train"]),"development":len(splits["development"]),"oos":len(splits["oos"])},
      "fixed_contract":CONTRACT,"ema_pairs":[list(p) for p in PAIRS],"structure_entries":list(STRUCTURES),
      "directional_filter_pairs":[list(p) for p in FILTER_PAIRS],"stops":list(STOPS),
      "result_count":len(rows),"validated_count":sum(x["metrics"]["passes_all_user_targets"] for x in rows),
      "lower_timeframe_audit":"NOT CLAIMED: this sweep does not load or hash 1-second candles; unresolved_ohlc_bars_count alone is not proof of lower-timeframe resolution",
      "top_by_split":{},"results":rows}
    for split in ("train","development"):
        subset=sorted((x for x in rows if x["split"]==split),key=rank)
        report["top_by_split"][split]=subset[:20]
    path=OUT/"ema_pair_sweep_report.json"; path.write_text(json.dumps(report,indent=2,sort_keys=True))
    print("EMA_STRUCTURE_EXPANDED_FINAL "+json.dumps({"report":str(path),"result_count":len(rows),
      "validated_count":report["validated_count"],
      "top_by_split":{k:[{"entry":x["entry"],"ema_fast":x["ema_fast"],"ema_slow":x["ema_slow"],"filter_long":x["filter_long"],"filter_short":x["filter_short"],"stop":x["stop"],"metrics":x["metrics"]} for x in v[:10]] for k,v in report["top_by_split"].items()}},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
