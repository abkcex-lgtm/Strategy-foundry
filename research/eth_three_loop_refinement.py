"""Three-cycle ETH-only research refinement. Research only; OOS remains untouched."""
import asyncio, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
     "reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
     "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only",
     "break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
def clean(m):
 return {k:(float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v) for k,v in m.items()}
def digest(ledger):
 return hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def metrics_row(name, entry, lf, sf, atr, stop, cb, ct, part, filters):
    longs,shorts=filters[lf],filters[sf]
    raw_l,raw_s=entry
    l=pd.Series(raw_l,index=part.index).fillna(False)&longs[0].reindex(part.index).fillna(False)
    s=pd.Series(raw_s,index=part.index).fillna(False)&shorts[1].reindex(part.index).fillna(False)
    m,ledger=sm.run_combined(part,l,s,atr.reindex(part.index),stop,loss_cooldown_bars=cb,cooldown_trigger_losses=ct)
    return {"entry":name,"long_filter":lf,"short_filter":sf,"stop":stop,"cooldown_bars":cb,"cooldown_trigger_losses":ct,
            "metrics":clean(m),"ledger_sha256":digest(ledger)}
def score(m):
    # Gate-preserving ordering: first streak/frequency, then WR/PF/net, never optimize OOS.
    return (m["max_consecutive_losses"]<=12, m["tp_per_day"]>=3,
            m["win_rate_pct"]>40, (m["profit_factor"] or 0)>1, m["net_pnl"]>0,
            m["trades_per_day"]>=7.5, min(m["tp_per_day"]/3,1),
            m["win_rate_pct"]/100, m["profit_factor"] or 0, m["net_pnl"])
async def main():
    started=time.monotonic(); sm.CONTRACT=normalize_contract(CFG)
    df,manifest=await get_data(); splits=split_reference_period(df)
    report={"schema":"eth_three_loop_refinement_v1","data_manifest":manifest,"contract":CFG,
      "oos_evaluated":False,"oos_used_for_selection":False,
      "ambiguity_note":"1m OHLC only; same-candle SL/TP ambiguity is handled conservatively as SL; research is not final validation.",
      "loops":[]}
    # LOOP 1: reproduce the full independent non-MA grid and establish baseline/shortlist.
    import research.eth_independent_non_ma_strategy_search as broad
    await broad.main()
    r=json.loads((OUT/"eth_independent_non_ma_strategy_sweep_report.json").read_text())
    report["loop_1"]={"method":"Full 2,400-configuration non-MA search, reproduced from the committed research module.",
      "tested_per_split":{k:v["tested_candidates"] for k,v in r["splits"].items()},
      "baseline":{k:v["baseline_11_loss_metrics"] for k,v in r["splits"].items()},
      "pass_count":r["cross_split_candidates_meeting_both"]["count"],
      "best_development_streak_safe":r["splits"]["development"]["top_streak_le_12"][:5],
      "best_train_streak_safe":r["splits"]["train"]["top_streak_le_12"][:5]}
    # LOOP 2: refine the strongest signal families with asymmetric long/short filters.
    sm.CONTRACT=normalize_contract(CFG)
    keyrows={}
    # Use the same candidate families and filter pool on both splits so matched configurations can be compared.
    common_safe=r["splits"]["train"]["top_streak_le_12"][:10]+r["splits"]["development"]["top_streak_le_12"][:10]
    entries=sorted(set(x["entry"] for x in common_safe))[:6]
    common_filters=sorted(set(x["filter"] for x in common_safe))
    if len(common_filters)<6:
        common_filters=list(dict.fromkeys(common_filters+["baseline","adx18_directional","rsi_momentum","strong_candle_close","atr_midrange","body_plus_adx22"]))
    common_filters=common_filters[:6]
    for sn in ("train","development"):
        part=splits[sn]; sig,atr=broad.make_signals(part); f=sm.confirmation_filters(part)
        filters=common_filters
        rows=[]
        for en in entries:
            raw=sig[en]
            for lf in filters:
             for sf in filters:
              for stop in ("signal_candle","swing_3","atr_1.0"):
               for cb,ct in ((0,1),(1,1),(2,1)):
                longs=(pd.Series(raw[0],index=part.index), f[lf][0])
                shorts=(pd.Series(raw[1],index=part.index), f[sf][1])
                l=longs[0].fillna(False)&longs[1].fillna(False)
                s=shorts[0].fillna(False)&shorts[1].fillna(False)
                m,ledger=sm.run_combined(part,l,s,atr,stop,loss_cooldown_bars=cb,cooldown_trigger_losses=ct)
                row={"entry":en,"long_filter":lf,"short_filter":sf,"stop":stop,"cooldown_bars":cb,"cooldown_trigger_losses":ct,
                     "metrics":clean(m),"ledger_sha256":digest(ledger)}
                rows.append(row); key=(en,lf,sf,stop,cb,ct); keyrows[(sn,key)]=row
        rows.sort(key=lambda x:score(x["metrics"]),reverse=True)
        report["loops"].append({"loop":2,"split":sn,"tested_candidates":len(rows),"top_candidates":rows[:10]})
    # Cross-split shortlist from the directional refinement; all acceptance gates are unchanged.
    cross=[]
    for _,key in list(keyrows):
        tr=keyrows.get(("train",key)); dv=keyrows.get(("development",key))
        if not tr or not dv: continue
        tm,dm=tr["metrics"],dv["metrics"]
        gates=lambda m:(m["win_rate_pct"]>40 and (m["profit_factor"] or 0)>1 and m["net_pnl"]>0
          and m["tp_per_day"]>=3 and m["trades_per_day"]>=7.5 and m["max_consecutive_losses"]<=12)
        cross.append({"parameters":{k:v for k,v in zip(("entry","long_filter","short_filter","stop","cooldown_bars","cooldown_trigger_losses"),key)},
          "train":tm,"development":dm,"train_pass":gates(tm),"development_pass":gates(dm),
          "train_ledger_sha256":tr["ledger_sha256"],"development_ledger_sha256":dv["ledger_sha256"]})
    cross.sort(key=lambda x:(x["train_pass"] and x["development_pass"],min(score(x["train"]),score(x["development"]))),reverse=True)
    report["loop_2_cross_split"]={"tested_matched_configurations":len(cross),"both_splits_pass_all_gates":sum(x["train_pass"] and x["development_pass"] for x in cross),"top":cross[:20]}
    # LOOP 3: retest the best 20 cross-split candidates on train/development and constituent train months.
    finalists=cross[:20]
    final=[]
    for cand in finalists:
        key=tuple(cand["parameters"][k] for k in ("entry","long_filter","short_filter","stop","cooldown_bars","cooldown_trigger_losses"))
        entry_name,lf,sf,stop,cb,ct=key; item={"parameters":cand["parameters"],"train":{},"development":{},"train_months":{}}
        for sn in ("train","development"):
            part=splits[sn]; sig,atr=broad.make_signals(part); f=sm.confirmation_filters(part); raw=sig[entry_name]
            l=pd.Series(raw[0],index=part.index).fillna(False)&f[lf][0].fillna(False)
            s=pd.Series(raw[1],index=part.index).fillna(False)&f[sf][1].fillna(False)
            m,ledger=sm.run_combined(part,l,s,atr,stop,loss_cooldown_bars=cb,cooldown_trigger_losses=ct)
            item[sn]={"metrics":clean(m),"ledger_sha256":digest(ledger)}
        trpart=splits["train"]; sigtr,atrtr=broad.make_signals(trpart); ftr=sm.confirmation_filters(trpart); raw=sigtr[entry_name]
        train_month_labels=pd.to_datetime(trpart["open_time"],unit="ms",utc=True).dt.strftime("%Y-%m")
        for month in sorted(train_month_labels.unique()):
            mask=(train_month_labels==month).to_numpy(); mp=trpart.loc[mask]
            l=pd.Series(raw[0],index=trpart.index).loc[mask].fillna(False)&ftr[lf][0].loc[mask].fillna(False)
            s=pd.Series(raw[1],index=trpart.index).loc[mask].fillna(False)&ftr[sf][1].loc[mask].fillna(False)
            m,ledger=sm.run_combined(mp,l,s,atrtr.loc[mask],stop,loss_cooldown_bars=cb,cooldown_trigger_losses=ct)
            item["train_months"][month]={"metrics":clean(m),"ledger_sha256":digest(ledger)}
        item["all_gates_both_splits"]=cand["train_pass"] and cand["development_pass"]
        final.append(item)
    report["loop_3_final_retest"]={"tested_finalists":len(final),"all_gates_both_splits_count":sum(x["all_gates_both_splits"] for x in final),"finalists":final}
    report["elapsed_seconds"]=round(time.monotonic()-started,2)
    p=OUT/"eth_three_loop_refinement_report.json"; p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    best=sorted(final,key=lambda x:(x["all_gates_both_splits"],min(score(x["train"]["metrics"]),score(x["development"]["metrics"]))),reverse=True)[:1]
    print("THREE_LOOP_FINAL "+json.dumps({"report":str(p),"elapsed_seconds":report["elapsed_seconds"],"loop2_pass_count":report["loop_2_cross_split"]["both_splits_pass_all_gates"],"loop3_pass_count":report["loop_3_final_retest"]["all_gates_both_splits_count"],"best":best},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
