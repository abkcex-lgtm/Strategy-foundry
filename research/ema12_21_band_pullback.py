"""Test the user's EMA 12/21 band-pullback reversal idea against the exact historical 11-loss reference.
Only TRAIN (Apr-Jun 2026) and DEVELOPMENT (Jul 2026) are evaluated; OOS stays untouched.
"""
import asyncio, hashlib, json, time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from app.engine import normalize_contract
import research.ema_9_21_123 as sm
from research.run_factory import get_data, split_reference_period

OUT = Path("research-results")
OUT.mkdir(exist_ok=True)
CFG = {
    "symbol": "ETHUSDT", "initial_balance": 1000, "risk_mode": "fixed_usd",
    "risk_usd": 10, "compound": False, "reward_risk": 3.5, "target_mode": "net_rr",
    "commission_pct_per_side": 0.04, "slippage_pct_per_side": 0.02,
    "max_positions": 1, "entry_model": "next_candle_open", "exit_model": "sl_tp_only",
    "min_win_rate_pct": 40, "min_profit_factor": 1, "min_tp_per_day": 3,
    "max_consecutive_losses": 12, "use_min_win_rate": True, "use_min_profit_factor": True,
    "use_min_tp_per_day": True, "use_max_consecutive_losses": True,
    "min_trades_per_day": 10, "min_trades_per_month": 300,
    "break_even_enabled": False, "trailing_stop_enabled": False,
    "partial_exit_enabled": False, "time_exit_enabled": False,
}

def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def ema1221_band_signals(df):
    c = df.close.astype(float)
    o = df.open.astype(float)
    h = df.high.astype(float)
    lo = df.low.astype(float)
    e12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    e21 = c.ewm(span=21, adjust=False, min_periods=21).mean()
    cross_up = (e12 > e21) & (e12.shift(1) <= e21.shift(1))
    cross_dn = (e12 < e21) & (e12.shift(1) >= e21.shift(1))
    # Cross must precede the setup; allow a pullback for up to 30 completed bars.
    up_recent = cross_up.astype(int).rolling(31, min_periods=1).max().shift(1).fillna(0).astype(bool)
    dn_recent = cross_dn.astype(int).rolling(31, min_periods=1).max().shift(1).fillna(0).astype(bool)
    # A pullback wick reaches the band between EMA12 and EMA21.
    # Reversal confirmation: bullish/bearish candle closes back beyond the fast EMA
    # and beyond the previous candle's high/low. Entry is next candle open.
    long_sig = (
        up_recent & (c > e12) & (c > e21) & (lo <= e12) & (lo >= e21)
        & (c > o) & (c > h.shift(1))
    )
    short_sig = (
        dn_recent & (c < e12) & (c < e21) & (h >= e12) & (h <= e21)
        & (c < o) & (c < lo.shift(1))
    )
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return long_sig.fillna(False), short_sig.fillna(False), atr

def brief(m):
    keys = ("trades", "long_trades", "short_trades", "wins", "win_rate_pct", "profit_factor",
            "tp_per_day", "trades_per_day", "max_consecutive_losses", "net_pnl", "final_balance",
            "total_costs", "monthly_trade_counts", "capital_exhausted", "unresolved_ohlc_bars_count",
            "passes_all_user_targets")
    return {k: m.get(k) for k in keys}

async def main():
    started = time.monotonic()
    sm.CONTRACT = normalize_contract(CFG)
    df, manifest = await get_data()
    splits = split_reference_period(df)
    report = {
        "schema": "ema12_21_band_pullback_vs_11_loss_baseline_v1",
        "data_manifest": manifest,
        "data_scope": "Binance USD-M ETHUSDT 1m; Train Apr-Jun 2026 and Development Jul 2026 only",
        "split_rows": {k: len(splits[k]) for k in ("train", "development", "oos")},
        "oos_evaluated": False, "oos_used_for_selection": False,
        "contract": CFG,
        "strategy_definition": {
            "ema": "EMA12 and EMA21, close-based, adjust=False",
            "long": "After bullish cross, within next 30 bars, close above both EMAs; signal candle low lies between EMA12 and EMA21, bullish close above EMA12 and prior candle high",
            "short": "Mirror after bearish cross; signal candle high lies between EMA12 and EMA21, bearish close below EMA12 and prior candle low",
            "entry": "Next candle open",
            "stop": "Lowest low / highest high of signal candle and prior 2 candles (swing_3 proxy for last small swing)",
            "target": "Net +3.5R; fixed net risk $10; fees 0.04% per side and slippage 0.02% per side",
            "execution": "One shared chronological long/short position; SL/TP only; same-bar OHLC ambiguity conservatively resolves to SL; 1-second audit not claimed"
        },
        "splits": {}
    }
    for split in ("train", "development"):
        part = splits[split]
        # Exact historical candidate associated with the prior 11-loss result:
        # EMA9/21 structure-break after pullback, strong-candle long filter,
        # unfiltered short side, swing_5 stop. This is rerun on the same split.
        sigs, base_atr = sm.signals(part)
        filters = sm.confirmation_filters(part)
        base_l, base_s = sigs["structure_break_pb8_br15"]
        base_l = base_l & filters["strong_candle_close"][0].reindex(part.index).fillna(False)
        base_s = base_s & filters["baseline"][1].reindex(part.index).fillna(False)
        base_m, base_ledger = sm.run_combined(part, base_l, base_s, base_atr, "swing_5")
        # New EMA12/21 band pullback reversal. Stop is the last 3-candle local low/high.
        long_sig, short_sig, atr = ema1221_band_signals(part)
        new_m, new_ledger = sm.run_combined(part, long_sig, short_sig, atr, "swing_3")
        report["splits"][split] = {
            "baseline_11_loss_reference": {"metrics": brief(base_m), "trades": len(base_ledger),
                                           "ledger_sha256": sha(base_ledger)},
            "ema12_21_band_pullback": {"metrics": brief(new_m), "trades": len(new_ledger),
                                       "long_signals": int(long_sig.sum()), "short_signals": int(short_sig.sum()),
                                       "ledger_sha256": sha(new_ledger)},
            "comparison": {
                "trades_delta": int(new_m["trades"] - base_m["trades"]),
                "win_rate_delta_percentage_points": round(float(new_m["win_rate_pct"] - base_m["win_rate_pct"]), 4),
                "profit_factor_delta": (round(float(new_m["profit_factor"] or 0) - float(base_m["profit_factor"] or 0), 6)),
                "tp_per_day_delta": round(float(new_m["tp_per_day"] - base_m["tp_per_day"]), 6),
                "trades_per_day_delta": round(float(new_m["trades_per_day"] - base_m["trades_per_day"]), 6),
                "max_loss_streak_delta": int(new_m["max_consecutive_losses"] - base_m["max_consecutive_losses"]),
                "net_pnl_delta_usd": round(float(new_m["net_pnl"] - base_m["net_pnl"]), 6),
            }
        }
        (OUT / "ema12_21_band_pullback_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        print("EMA12_21_SPLIT " + json.dumps({
            "split": split,
            "baseline": brief(base_m),
            "ema12_21_band_pullback": brief(new_m),
            "comparison": report["splits"][split]["comparison"],
            "long_signals": int(long_sig.sum()), "short_signals": int(short_sig.sum())
        }, sort_keys=True), flush=True)
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    path = OUT / "ema12_21_band_pullback_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print("EMA12_21_FINAL " + json.dumps({
        "report": str(path), "elapsed_seconds": report["elapsed_seconds"],
        "oos_evaluated": False,
        "splits": {k: {"baseline": v["baseline_11_loss_reference"]["metrics"],
                       "ema12_21_band_pullback": v["ema12_21_band_pullback"]["metrics"],
                       "comparison": v["comparison"]} for k, v in report["splits"].items()}
    }, sort_keys=True), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
