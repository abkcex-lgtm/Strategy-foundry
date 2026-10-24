"""Half-cost sensitivity test for the fixed 11-loss-streak ETHUSDT candidate.
Research-only. Replays the exact candidate with baseline costs and half costs;
all strategy rules and risk/exit settings stay unchanged. OOS remains untouched.
"""
import asyncio, json
from pathlib import Path
import research.ema_9_21_123 as strategy_mod
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT = Path("research-results")
OUT.mkdir(exist_ok=True)
BASE = {
    "symbol":"ETHUSDT", "initial_balance":1000, "risk_mode":"fixed_usd",
    "risk_usd":10, "compound":False, "reward_risk":3.5, "target_mode":"net_rr",
    "max_positions":1, "entry_model":"next_candle_open", "exit_model":"sl_tp_only",
    "min_win_rate_pct":40, "min_profit_factor":1, "min_tp_per_day":3,
    "max_consecutive_losses":12, "use_min_win_rate":True, "use_min_profit_factor":True,
    "use_min_tp_per_day":True, "use_max_consecutive_losses":True,
    "min_trades_per_day":10, "min_trades_per_month":300,
    "break_even_enabled":False, "trailing_stop_enabled":False,
    "partial_exit_enabled":False, "time_exit_enabled":False
}
CANDIDATE = {
    "entry_method":"structure_break_after_pullback",
    "confirmation_filter":"ema21_slope5_atr",
    "stop_method":"swing_5",
    "cooldown_bars":1,
    "cooldown_trigger_losses":1,
    "side":"LONG+SHORT_SHARED_SINGLE_POSITION"
}
def brief(m):
    keys=("trades","wins","win_rate_pct","profit_factor","tp_per_day","trades_per_day",
          "max_consecutive_losses","net_pnl","final_balance","total_costs",
          "capital_exhausted","monthly_trade_counts","unresolved_ohlc_bars_count",
          "long_trades","short_trades")
    return {k:m.get(k) for k in keys}
async def main():
    df, manifest = await get_data()
    splits = split_reference_period(df)
    output = {"schema":"half_cost_sensitivity_11_loss_candidate_v1",
      "data":manifest, "candidate":CANDIDATE,
      "costs":{"baseline":{"commission_pct_per_side":0.04,"slippage_pct_per_side":0.02},
               "half":{"commission_pct_per_side":0.02,"slippage_pct_per_side":0.01}},
      "contract_note":"Only commission and slippage are changed; initial balance $1000, fixed net risk $10, net reward 3.5R, one shared LONG/SHORT position, next-candle-open entries, SL/TP-only exits, same entry/filter/stop/cooldown. OOS not evaluated.",
      "splits":{}}
    for split_name in ("train","development"):
        part = splits[split_name]
        sigs, atr = strategy_mod.signals(part)
        filters = strategy_mod.confirmation_filters(part)
        raw_l, raw_s = sigs[CANDIDATE["entry_method"]]
        lf, sf = filters[CANDIDATE["confirmation_filter"]]
        long_sig = raw_l & lf.fillna(False)
        short_sig = raw_s & sf.fillna(False)
        split_results = {}
        for label, commission, slippage in (("baseline",0.04,0.02),("half_cost",0.02,0.01)):
            contract = normalize_contract({**BASE, "commission_pct_per_side":commission,
                                           "slippage_pct_per_side":slippage})
            strategy_mod.CONTRACT = contract
            metrics, ledger = strategy_mod.run_combined(
                part, long_sig, short_sig, atr, CANDIDATE["stop_method"],
                loss_cooldown_bars=CANDIDATE["cooldown_bars"],
                cooldown_trigger_losses=CANDIDATE["cooldown_trigger_losses"])
            split_results[label] = {"metrics":brief(metrics), "ledger_sha256":strategy_mod.hashlib.sha256(
                json.dumps(ledger,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
                "closed_ledger_rows":len(ledger)}
        b = split_results["baseline"]["metrics"]; h = split_results["half_cost"]["metrics"]
        deltas={}
        for k in ("trades","wins","win_rate_pct","profit_factor","tp_per_day","trades_per_day",
                  "max_consecutive_losses","net_pnl","final_balance","total_costs"):
            bv,hv=b.get(k),h.get(k)
            if isinstance(bv,(int,float)) and isinstance(hv,(int,float)): deltas[k]=round(hv-bv,10)
        split_results["delta_half_minus_baseline"] = deltas
        split_results["baseline_reproduces_reference_development"] = (
            split_name != "development" or (
              b["trades"] == 207 and abs(b["win_rate_pct"]-20.77294685990338)<1e-8 and
              abs(b["profit_factor"]-0.9176829268292664)<1e-8 and
              b["max_consecutive_losses"] == 11 and abs(b["net_pnl"]+135)<1e-8))
        output["splits"][split_name]=split_results
        print("HALF_COST_SPLIT "+json.dumps({"split":split_name,**split_results},sort_keys=True),flush=True)
    output["development_reference_reproduction_pass"] = output["splits"]["development"]["baseline_reproduces_reference_development"]
    output["oos_evaluated"] = False
    path=OUT/"half_cost_11_loss_report.json"
    path.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False))
    print("HALF_COST_FINAL "+json.dumps({"report":str(path),"baseline_reproduction_pass":output["development_reference_reproduction_pass"],
       "development":output["splits"]["development"]},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
