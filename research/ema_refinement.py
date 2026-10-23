"""Focused causal refinement of the best EMA 9/21 structure-break candidates.
Research-only: preserves the canonical fixed net-risk/reward and execution contract.
"""
import asyncio, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from research.run_factory import get_data, split_reference_period
from research.ema_9_21_123 import CONTRACT, signals, confirmation_filters, run_combined

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)

def ema_filters(df):
    c=df.close.astype(float)
    e21=c.ewm(span=21,adjust=False,min_periods=21).mean()
    e50=c.ewm(span=50,adjust=False,min_periods=50).mean()
    return {
      "ema50_trend": ((c>e50)&(e21>e50), (c<e50)&(e21<e50)),
      "ema9_21_50_alignment": ((e21>e50), (e21<e50)),
      "ema50_slope": (e50>e50.shift(5), e50<e50.shift(5)),
    }

def digest(ledger):
    return hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def score(x):
    m=x["metrics"]
    # No target is relaxed. This score only orders failed candidates by distance to all gates.
    gaps=(max(0,40.0001-m["win_rate_pct"])/40 +
          max(0,1.0001-(m["profit_factor"] or 0)) +
          max(0,3-m["tp_per_day"])/3 +
          max(0,10-m["trades_per_day"])/10 +
          max(0,m["max_consecutive_losses"]-12)/12 +
          (1 if m["capital_exhausted"] else 0))
    return (not m["passes_all_user_targets"],gaps,-m["net_pnl"],-m["trades"])

async def main():
    df, manifest = await get_data()
    splits=split_reference_period(df)
    results=[]
    entries=("structure_break_after_pullback","structure_break_pb5_br15","structure_break_pb8_br15")
    stops=("swing_3","swing_5")
    cooldowns=((0,2),(1,1),(1,2),(1,3),(2,1),(2,2),(3,1),(3,2),(5,1),(5,2),(8,1),(13,1))
    for split in ("train","development"):
        part=splits[split]
        sigs,atr=signals(part)
        f=confirmation_filters(part); f.update(ema_filters(part))
        # Singles plus causal combinations. Long and short confirmations remain directional.
        specs=[]
        singles=("baseline","strong_candle_close","ema21_slope5_atr","adx22_directional",
                 "rsi_tight_momentum","avoid_extended_entry","distance_to_ema_atr_0_75",
                 "distance_to_ema_atr_1_0","slope_adx_candle_stack","trend_quality_stack",
                 "ema50_trend","ema9_21_50_alignment","ema50_slope")
        for name in singles:
            specs.append((name,f[name][0],f[name][1]))
        combos=(("ema21_slope5_atr","adx22_directional"),
                ("ema21_slope5_atr","strong_candle_close"),
                ("adx22_directional","strong_candle_close"),
                ("ema21_slope5_atr","adx22_directional","strong_candle_close"),
                ("ema21_slope5_atr","adx22_directional","rsi_tight_momentum"),
                ("ema50_trend","strong_candle_close"),
                ("ema50_trend","adx22_directional"),
                ("ema50_slope","adx22_directional"),
                ("ema9_21_50_alignment","strong_candle_close"),
                ("ema9_21_50_alignment","adx22_directional"),
                ("avoid_extended_entry","adx22_directional"),
                ("distance_to_ema_atr_1_0","strong_candle_close"),
                ("distance_to_ema_atr_1_0","adx22_directional"),
                ("rsi_tight_momentum","strong_candle_close"))
        for names in combos:
            specs.append(("+".join(names),
                          np.logical_and.reduce([f[n][0].fillna(False).to_numpy(bool) for n in names]),
                          np.logical_and.reduce([f[n][1].fillna(False).to_numpy(bool) for n in names])))
        # Explicitly test asymmetric confirmations: often the better long and short setups differ.
        asym_pairs=(("strong_candle_close","baseline"),
                    ("baseline","strong_candle_close"),
                    ("ema21_slope5_atr","baseline"),
                    ("baseline","ema21_slope5_atr"),
                    ("adx22_directional","baseline"),
                    ("baseline","adx22_directional"),
                    ("ema21_slope5_atr","adx22_directional"),
                    ("adx22_directional","ema21_slope5_atr"),
                    ("ema50_trend","strong_candle_close"),
                    ("strong_candle_close","ema50_trend"),
                    ("distance_to_ema_atr_1_0","rsi_tight_momentum"),
                    ("rsi_tight_momentum","distance_to_ema_atr_1_0"))
        for ln,sn in asym_pairs:
            specs.append((f"asym_long-{ln}__short-{sn}",f[ln][0],f[sn][1]))
        for entry in entries:
            raw_l,raw_s=sigs[entry]
            for fname,lf,sf in specs:
                lf=pd.Series(lf,index=part.index) if not isinstance(lf,pd.Series) else lf
                sf=pd.Series(sf,index=part.index) if not isinstance(sf,pd.Series) else sf
                long_sig=raw_l & lf.fillna(False)
                short_sig=raw_s & sf.fillna(False)
                for stop in stops:
                    for cool_bars,trigger in cooldowns:
                        m,ledger=run_combined(part,long_sig,short_sig,atr,stop,
                                              loss_cooldown_bars=cool_bars,
                                              cooldown_trigger_losses=trigger)
                        results.append({"split":split,"entry_method":entry,"confirmation_filter":fname,
                            "stop_method":stop,"cooldown_bars":cool_bars,"cooldown_trigger_losses":trigger,
                            "side":"LONG+SHORT_SHARED_SINGLE_POSITION","metrics":m,
                            "ledger_sha256":digest(ledger)})
    out={"schema":"focused_ema_causal_filter_stack_cooldown_v2","data":manifest,
         "data_scope":"Binance USD-M ETHUSDT 1m Apr-Sep 2026; Train and Development only; OOS not evaluated",
         "split_rows":{"train":len(splits["train"]),"development":len(splits["development"]),"oos":len(splits["oos"])},
         "fixed_contract":CONTRACT,"entry_methods":list(entries),"stop_methods":list(stops),
         "cooldown_variants":[{"bars":b,"trigger_losses":t} for b,t in cooldowns],
         "result_count":len(results),"validated_count":sum(x["metrics"]["passes_all_user_targets"] for x in results),
         "top_by_split":{},"results":results}
    for split in ("train","development"):
        subset=sorted((x for x in results if x["split"]==split),key=score)
        out["top_by_split"][split]=subset[:25]
    (OUT/"ema_refinement_report.json").write_text(json.dumps(out,indent=2,sort_keys=True))
    print("EMA_REFINEMENT_FINAL "+json.dumps({"report":"research-results/ema_refinement_report.json",
        "result_count":len(results),"validated_count":out["validated_count"],
        "top_by_split":{k:[{"entry_method":x["entry_method"],"confirmation_filter":x["confirmation_filter"],
        "stop_method":x["stop_method"],"cooldown_bars":x["cooldown_bars"],
        "cooldown_trigger_losses":x["cooldown_trigger_losses"],"metrics":x["metrics"]} for x in v[:10]]
        for k,v in out["top_by_split"].items()}},sort_keys=True),flush=True)

if __name__=="__main__": asyncio.run(main())
