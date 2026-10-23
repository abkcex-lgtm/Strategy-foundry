"""Focused real-data search for high-frequency mean-reversion and failed-breakout entries.

TRAIN-only candidate search; top TRAIN candidates are independently replayed on DEVELOPMENT.
OOS is deliberately not loaded or evaluated. The fixed user contract is not relaxed.
"""
import asyncio, hashlib, json
from pathlib import Path
import numpy as np
from app.engine import search_candidates, backtest, normalize_contract, prepare_backtest_data, _candidate_distance
from app.audit import independent_replay
from research.run_factory import get_data, split_reference_period, ledger_match, brief, frame_hash

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
FAMILIES=[
 "rsi_reversion","bollinger_reversion","stochastic_reversion","williams_r_reversion",
 "money_flow_index_reversion","range_midpoint_reversion","failed_breakout_reversal",
 "wick_rejection","trend_exhaustion_reversal","volume_climax_reversal",
 "micro_double_bottom","micro_double_top","support_resistance_rejection",
 "pivot_reversal","engulfing_reversal","vwap_reclaim"
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
 splits=split_reference_period(df)
 train,dev=splits["train"],splits["development"]
 prepared_train=prepare_backtest_data(train)
 prepared_dev=prepare_backtest_data(dev)
 print("FOCUSED_FAMILY_START "+json.dumps({
  "families":FAMILIES,"requested_candidates":5000,"seed":271830,
  "train_rows":len(train),"development_rows":len(dev),
  "dataset_sha256":manifest.get("dataset_sha256"),
  "policy":"OOS untouched; fixed contract"
 },sort_keys=True),flush=True)
 search=search_candidates(
  train,max_candidates=5000,seed=271830,contract=CONTRACT,
  algorithm="surrogate_genetic",
  search_space={"families":FAMILIES,
   "context_filters":["none","ema_trend","adx_strength","volume_confirm","vwap_side","low_volatility"],
   "confirmation_filters":["none","candle_direction","momentum","rsi_midline"]},
  prepared_context=prepared_train
 )
 pool=[]
 seen_strategies=set()
 for source, candidates in (("train_ranked",search.get("results",[])),
                            ("train_win_rate_ge_30",search.get("diagnostic_win_rate_30_candidates",[]))):
  for candidate in candidates:
   key=json.dumps(candidate.get("strategy",{}),sort_keys=True,separators=(",",":"))
   if key and key not in seen_strategies:
    seen_strategies.add(key); pool.append({**candidate,"_selection_source":source})
 rows=[]
 audit_failures=[]
 for idx,item in enumerate(pool,1):
  try:
   strategy=item["strategy"]
   primary=backtest(dev,strategy,CONTRACT,_prepared_context=prepared_dev)
   secondary=independent_replay(dev,strategy,CONTRACT)
   matched,evidence=ledger_match(primary,secondary)
   metrics=brief(primary["metrics"])
   distance,failed,distances=_candidate_distance(primary["metrics"],CONTRACT)
   row={
    "rank_train":idx,"selection_source":item.get("_selection_source"),"strategy":strategy,
    "train_metrics":brief(item.get("metrics",{})),
    "train_failed_conditions":item.get("failed_conditions",[]),
    "development_metrics":metrics,
    "development_distance":distance,
    "development_failed_conditions":failed,
    "development_criterion_distances":distances,
    "development_audit_status":"MATCH" if matched else "MISMATCH",
    "development_audit_evidence":evidence,
    "development_ledger_sha256":hashlib.sha256(json.dumps(primary["ledger"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
    "development_passes_all_targets":bool(matched and not failed)
   }
   rows.append(row)
   if not matched: audit_failures.append({"rank_train":idx,"strategy":strategy,"evidence":evidence})
  except Exception as exc:
   audit_failures.append({"rank_train":idx,"strategy":item.get("strategy"),"error":f"{type(exc).__name__}: {exc}"})
  print("FOCUSED_FAMILY_CANDIDATE "+json.dumps(rows[-1] if rows else audit_failures[-1],sort_keys=True),flush=True)
 rows.sort(key=lambda x:(not x["development_passes_all_targets"],x["development_distance"],x["development_metrics"].get("capital_exhausted",False),-float(x["development_metrics"].get("net_pnl") or 0)))
 report={
  "schema":"focused_mean_reversion_failed_breakout_search_v2",
  "data":manifest,
  "data_scope":"Binance USD-M ETHUSDT 1m Apr-Sep 2026; Train Apr-Jun and Development Jul only; OOS untouched",
  "split_rows":{"train":len(train),"development":len(dev),"oos_not_evaluated":len(splits["oos"])},
  "fixed_contract":CONTRACT,"families":FAMILIES,
  "search":{"requested_candidates":search.get("requested_candidates"),"evaluated_candidates":search.get("evaluated_candidates"),
   "failed_candidates":search.get("failed_candidates"),"duplicate_genomes_skipped":search.get("duplicate_genomes_skipped"),
   "family_evaluated_counts":search.get("family_evaluated_counts"),"algorithm":search.get("algorithm"),
   "seed":search.get("seed"),"preflight_summary":search.get("preflight_summary"),
   "all_passing_train":search.get("all_passing"),"warning":search.get("warning")},
  "development_audit_failures":audit_failures,"development_candidate_pool_count":len(pool),
  "development_validated_count":sum(x["development_passes_all_targets"] for x in rows),
  "development_results":rows,
  "top_development":rows[:5]
 }
 (OUT/"focused_family_report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
 print("FOCUSED_FAMILY_FINAL "+json.dumps({
  "report":"research-results/focused_family_report.json",
  "requested":report["search"]["requested_candidates"],
  "evaluated":report["search"]["evaluated_candidates"],
  "development_candidate_pool_count":len(pool),"development_validated_count":report["development_validated_count"],
  "top_development":[{"strategy":x["strategy"],"train_metrics":x["train_metrics"],"development_metrics":x["development_metrics"],
   "development_failed_conditions":x["development_failed_conditions"],"development_audit_status":x["development_audit_status"]} for x in rows[:5]]
 },sort_keys=True),flush=True)

if __name__=="__main__": asyncio.run(main())
