"""Paired zero-cost test of the 11-loss candidate on real Binance ETHUSDT 1m data.
Only commission and slippage change in the zero-cost arm; OOS remains untouched.
"""
import asyncio, json, hashlib
from pathlib import Path
import research.ema_9_21_123 as strategy_mod
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

OUT = Path("research-results")
OUT.mkdir(exist_ok=True)
BASE = {
    "symbol": "ETHUSDT", "initial_balance": 1000, "risk_mode": "fixed_usd",
    "risk_usd": 10, "compound": False, "reward_risk": 3.5, "target_mode": "net_rr",
    "max_positions": 1, "entry_model": "next_candle_open", "exit_model": "sl_tp_only",
    "min_win_rate_pct": 40, "min_profit_factor": 1, "min_tp_per_day": 3,
    "max_consecutive_losses": 12, "use_min_win_rate": True, "use_min_profit_factor": True,
    "use_min_tp_per_day": True, "use_max_consecutive_losses": True,
    "min_trades_per_day": 10, "min_trades_per_month": 300,
    "break_even_enabled": False, "trailing_stop_enabled": False,
    "partial_exit_enabled": False, "time_exit_enabled": False
}
CANDIDATE = {
    "entry_method": "structure_break_after_pullback",
    "confirmation_filter": "ema21_slope5_atr",
    "stop_method": "swing_5",
    "cooldown_bars": 1,
    "cooldown_trigger_losses": 1,
    "side": "LONG+SHORT_SHARED_SINGLE_POSITION"
}
def brief(m):
    keys = ("trades", "wins", "win_rate_pct", "profit_factor", "tp_per_day",
            "trades_per_day", "max_consecutive_losses", "net_pnl", "final_balance",
            "total_costs", "capital_exhausted", "monthly_trade_counts",
            "unresolved_ohlc_bars_count", "long_trades", "short_trades")
    return {k: m.get(k) for k in keys}
async def main():
    df, manifest = await get_data()
    splits = split_reference_period(df)
    out = {
        "schema": "zero_cost_11_loss_candidate_v1",
        "data": manifest,
        "candidate": CANDIDATE,
        "costs": {
            "baseline": {"commission_pct_per_side": 0.04, "slippage_pct_per_side": 0.02},
            "zero_cost": {"commission_pct_per_side": 0.0, "slippage_pct_per_side": 0.0}
        },
        "contract_note": "Paired comparison: same entry/filter/stop/cooldown, initial balance $1000, fixed risk $10, net reward 3.5R, one shared LONG/SHORT position, next-candle-open, SL/TP-only. Zero-cost arm sets commission and slippage to exactly zero. OOS not evaluated.",
        "ambiguity_note": "Same-bar TP/SL ambiguity remains conservatively counted as SL on 1m OHLC; 1s resolution not implemented.",
        "oos_evaluated": False,
        "splits": {}
    }
    for split_name in ("train", "development"):
        part = splits[split_name]
        sigs, atr = strategy_mod.signals(part)
        filters = strategy_mod.confirmation_filters(part)
        raw_l, raw_s = sigs[CANDIDATE["entry_method"]]
        lf, sf = filters[CANDIDATE["confirmation_filter"]]
        long_sig, short_sig = raw_l & lf.fillna(False), raw_s & sf.fillna(False)
        arms = {}
        for label, commission, slippage in (("baseline", 0.04, 0.02), ("zero_cost", 0.0, 0.0)):
            strategy_mod.CONTRACT = normalize_contract({
                **BASE, "commission_pct_per_side": commission,
                "slippage_pct_per_side": slippage
            })
            metrics, ledger = strategy_mod.run_combined(
                part, long_sig, short_sig, atr, CANDIDATE["stop_method"],
                loss_cooldown_bars=CANDIDATE["cooldown_bars"],
                cooldown_trigger_losses=CANDIDATE["cooldown_trigger_losses"])
            arms[label] = {
                "metrics": brief(metrics),
                "closed_ledger_rows": len(ledger),
                "ledger_sha256": hashlib.sha256(
                    json.dumps(ledger, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest()
            }
        b, z = arms["baseline"]["metrics"], arms["zero_cost"]["metrics"]
        deltas = {}
        for k in ("trades", "wins", "win_rate_pct", "profit_factor", "tp_per_day",
                  "trades_per_day", "max_consecutive_losses", "net_pnl",
                  "final_balance", "total_costs"):
            bv, zv = b.get(k), z.get(k)
            if isinstance(bv, (int, float)) and isinstance(zv, (int, float)):
                deltas[k] = round(zv - bv, 10)
        arms["delta_zero_cost_minus_baseline"] = deltas
        arms["baseline_reproduces_reference_development"] = (
            split_name != "development" or (
                b["trades"] == 207 and abs(b["win_rate_pct"] - 20.77294685990338) < 1e-8
                and abs(b["profit_factor"] - 0.9176829268292664) < 1e-8
                and b["max_consecutive_losses"] == 11 and abs(b["net_pnl"] + 135) < 1e-8
            )
        )
        out["splits"][split_name] = arms
        print("ZERO_COST_SPLIT " + json.dumps({"split": split_name, **arms}, sort_keys=True), flush=True)
    out["development_reference_reproduction_pass"] = out["splits"]["development"]["baseline_reproduces_reference_development"]
    path = OUT / "zero_cost_11_loss_report.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
    print("ZERO_COST_FINAL " + json.dumps({
        "report": str(path),
        "baseline_reproduction_pass": out["development_reference_reproduction_pass"],
        "development": out["splits"]["development"]
    }, sort_keys=True), flush=True)
if __name__ == "__main__":
    asyncio.run(main())
