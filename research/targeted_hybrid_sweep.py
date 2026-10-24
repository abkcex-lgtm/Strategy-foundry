"""Targeted hybrid sweep: structure/sweep-like entries with regime and confirmation filters.
Search uses TRAIN only; candidates are replayed on DEVELOPMENT with an independent ledger.
The final OOS slice is intentionally not evaluated.
"""
import asyncio, hashlib, json
from pathlib import Path
from app.engine import search_candidates, backtest, normalize_contract, prepare_backtest_data, _candidate_distance
from app.audit import independent_replay
from research.run_factory import get_data, split_reference_period, ledger_match, brief

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
FAMILIES=[
 "swing_structure_break","failed_breakout_reversal","wick_rejection",
 "support_resistance_rejection","pivot_reversal","ema_pullback",
 "atr_trend_pullback","engulfing_reversal","micro_double_bottom","micro_double_top"
]
CONTRACT=normalize_contract({
 "symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,
 "compound":False,"reward_risk":3.5,"target_mode":"net_rr",
 "commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
 "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only",
 "min_win_rate_pct":40,"min_profit_factor":1,"min_tp_per_day":3,
 "max_consecutive_losses":12,"use_min_win_rate":True,"use_min_profit_factor":True,
 "use_min_tp_per_day":True,"use_max_consecutive_losses":True,
 "min_trades_per_day":10,"min_trades_per_month":300,
 "break_even_enabled":False,"trailing_stop_enabled":False,
 "partial_exit_enabled":False,"time_exit_enabled":False
})
async def main():
 df,manifest=await get_data()
 splits=split_reference_period(df); train,dev=splits["train"],splits["development"]
 pt=prepare_backtest_data(train); pd=prepare_backtest_data(dev)
 print("HYBRID_START "+json.dumps({"requested_candidates":10000,"seed":271831,"families":FAMILIES,
  "train_rows":len(train),"development_rows":len(dev),"dataset_sha256":manifest.get("dataset_sha256"),
  "oos":"untouched","contract":CONTRACT},sort_keys=True),flush=True)
 search=search_candidates(train,max_candidates=10000,seed=271831,contract=CONTRACT,
  algorithm="surrogate_genetic",
  search_space={"families":FAMILIES,
   "context_filters":["none","ema_trend","adx_strength","volume_confirm","vwap_side","low_volatility"],
   "confirmation_filters":["none","candle_direction","momentum","rsi_midline"],
   "secondary_families":["none","swing_structure_break","failed_breakout_reversal","wick_rejection",
                         "support_resistance_rejection","pivot_reversal","ema_pullback","atr_trend_pullback"]},
  prepared_context=pt)
 pool=[]; seen=set()
 for source, candidates in (("train_ranked",search.get("results",[])),
                            ("train_win_rate_ge_30",search.get("diagnostic_win_rate_30_candidates",[]))):
  for item in candidates:
   key=json.dumps(item.get("strategy",{}),sort_keys=True,separators=(",",":"))
   if key and key not in seen: seen.add(key); pool.append({**item,"_source":source})
 rows=[]; errors=[]
 for i,item in enumerate(pool,1):
  try:
   strategy=item["strategy"]
   a=backtest(dev,strategy,CONTRACT,_prepared_context=pd)
   b=independent_replay(dev,strategy,CONTRACT)
   matched,evidence=ledger_match(a,b); metrics=brief(a["metrics"])
   distance,failed,distances=_candidate_distance(a["metrics"],CONTRACT)
   row={"rank":i,"source":item.get("_source"),"strategy":strategy,
    "train_metrics":brief(item.get("metrics",{})),"train_failed_conditions":item.get("failed_conditions",[]),
    "development_metrics":metrics,"distance":distance,"failed_conditions":failed,
    "criterion_distances":distances,"audit":"MATCH" if matched else "MISMATCH",
    "audit_evidence":evidence,"ledger_sha256":hashlib.sha256(json.dumps(a["ledger"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
    "passes_all_targets":bool(matched and not failed)}
   rows.append(row)
   if not matched: errors.append({"rank":i,"evidence":evidence})
  except Exception as exc: errors.append({"rank":i,"error":f"{type(exc).__name__}: {exc}"})
  if rows: print("HYBRID_CANDIDATE "+json.dumps(rows[-1],sort_keys=True),flush=True)
 rows.sort(key=lambda x:(not x["passes_all_targets"],x["distance"],x["development_metrics"].get("capital_exhausted",False),-float(x["development_metrics"].get("net_pnl") or 0)))
 report={"schema":"targeted_hybrid_structure_regime_confirmation_v1","data":manifest,
  "data_scope":"ETHUSDT 1m Apr-Sep 2026; Train Apr-Jun, Development Jul; OOS not evaluated",
  "split_rows":{"train":len(train),"development":len(dev),"oos_not_evaluated":len(splits["oos"])},
  "fixed_contract":CONTRACT,"families":FAMILIES,
  "search":{k:search.get(k) for k in ["requested_candidates","evaluated_candidates","failed_candidates","duplicate_genomes_skipped","family_evaluated_counts","algorithm","seed","preflight_summary","all_passing","warning"]},
  "candidate_pool_count":len(pool),"audit_errors":errors,
  "development_validated_count":sum(x["passes_all_targets"] for x in rows),
  "development_results":rows,"top_development":rows[:10]}
 (OUT/"targeted_hybrid_report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
 print("HYBRID_FINAL "+json.dumps({"report":"research-results/targeted_hybrid_report.json",
  "requested":report["search"]["requested_candidates"],"evaluated":report["search"]["evaluated_candidates"],
  "pool":len(pool),"validated":report["development_validated_count"],
  "top":[{"strategy":r["strategy"],"train":r["train_metrics"],"development":r["development_metrics"],
          "failed":r["failed_conditions"],"audit":r["audit"]} for r in rows[:5]]},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
