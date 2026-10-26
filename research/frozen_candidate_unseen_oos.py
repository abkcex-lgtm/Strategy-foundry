"""Frozen OOS diagnostic for the top-ranked candidate from the completed 15k search.
This is a one-time holdout test, not evidence of a strategy passing Train/Dev gates.
"""
import asyncio, hashlib, json
from pathlib import Path
import numpy as np
from app.engine import backtest, normalize_contract, prepare_backtest_data
from research.run_factory import get_data, split_reference_period, frame_hash

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CFG=normalize_contract({"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
 "reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
 "max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only",
 "min_win_rate_pct":40,"min_profit_factor":1,"min_tp_per_day":3,"max_consecutive_losses":12,
 "use_min_win_rate":True,"use_min_profit_factor":True,"use_min_tp_per_day":True,"use_max_consecutive_losses":True,
 "min_trades_per_day":10,"min_trades_per_month":300,"break_even_enabled":False,"trailing_stop_enabled":False,
 "partial_exit_enabled":False,"time_exit_enabled":False})
SPEC={"atr_period":30,"confirmation_filter":"rsi_midline","context_filter":"none","direction":"both",
 "family":"keltner_breakout","fast":14,"name":"Keltner channel breakout [stop=atr] 14/38 ATR 30x5 both [ctx=none; confirm=rsi_midline]",
 "secondary_family":"none","seed":8511,"signal_composition":"single","slow":38,"stop_atr":5.0,"stop_mode":"atr"}
def clean(x):
 if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items() if not str(k).startswith("_")}
 if isinstance(x,(list,tuple)): return [clean(v) for v in x]
 if isinstance(x,np.integer): return int(x)
 if isinstance(x,np.floating): return float(x)
 if isinstance(x,np.bool_): return bool(x)
 return x
async def main():
 df,manifest=await get_data()
 splits=split_reference_period(df)
 oos=splits["oos"]
 if len(oos)!=87840: raise RuntimeError(f"OOS size mismatch: {len(oos)}")
 bt=backtest(oos,SPEC,CFG,_prepared_context=prepare_backtest_data(oos))
 report={"schema":"frozen_candidate_unseen_oos_diagnostic_v1","candidate_frozen_before_oos":SPEC,
 "contract":CFG,"data_manifest":manifest,"oos_rows":len(oos),"oos_hash_sha256":frame_hash(oos),
 "oos_used_for_selection":False,"note":"This candidate did NOT pass Train+Development gates. OOS is reported only as a diagnostic; do not treat this as validation of a deployable strategy. 1m OHLC only; 1s same-candle ambiguity audit not done.",
 "prior_train_metrics":{"trades":306,"win_rate_pct":19.2810457516,"profit_factor":0.8360323887,"net_pnl":-405.0,"final_balance":595.0,"max_consecutive_losses":15,"avg_net_tp_per_day":0.6483565961,"trades_per_day":3.362663024,"total_costs":601.2532601},
 "prior_development_metrics":{"trades":125,"win_rate_pct":21.6,"profit_factor":0.9642857143,"net_pnl":-35.0,"final_balance":965.0,"max_consecutive_losses":16,"avg_net_tp_per_day":0.8709872533,"trades_per_day":4.032348395,"total_costs":273.7653037},
 "oos_metrics":bt.get("metrics"),"oos_run_hash_sha256":bt.get("run_hash_sha256")}
 path=OUT/"frozen_candidate_unseen_oos_report.json"
 path.write_text(json.dumps(clean(report),indent=2,sort_keys=True,allow_nan=False))
 print("FROZEN_OOS_RESULT "+json.dumps(clean({"report":str(path),"oos_metrics":bt.get("metrics"),"oos_hash":frame_hash(oos)}),default=str),flush=True)
if __name__=="__main__": asyncio.run(main())
