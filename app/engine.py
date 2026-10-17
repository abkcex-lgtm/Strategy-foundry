from __future__ import annotations
import hashlib, json, math
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from .data import validate_ohlcv, sha256_bytes, DataIntegrityError

# Preset only: every field is user-configurable through Settings/API.
CONTRACT = {
    "profile_name": "ETHUSDT fixed-risk 3.5R (editable preset)",
    "initial_balance": 1000.0,
    "risk_mode": "fixed_usd",  # fixed_usd | percent_initial | percent_equity
    "risk_usd": 10.0,
    "risk_pct": 1.0,
    "compound": False,
    "reward_risk": 3.5,
    "target_mode": "net_rr",  # net_rr | gross_rr
    "commission_pct_per_side": 0.04,
    "slippage_pct_per_side": 0.02,
    "max_positions": 1,
    "entry_model": "next_candle_open",
    "exit_model": "sl_tp_only",
    "max_consecutive_losses": 13,
    "min_win_rate_pct": 40.0,
    "min_profit_factor": 1.0,
    "min_tp_per_day": 3.0,
    "use_min_win_rate": True,
    "use_min_profit_factor": True,
    "use_min_tp_per_day": True,
    "use_max_consecutive_losses": True,
    "break_even_enabled": False,
    "trailing_stop_enabled": False,
    "partial_exit_enabled": False,
    "time_exit_enabled": False,
}
SUPPORTED_RISK_MODES = {"fixed_usd", "percent_initial", "percent_equity"}
SUPPORTED_TARGET_MODES = {"net_rr", "gross_rr"}

def normalize_contract(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(CONTRACT)
    overrides = dict(overrides or {})
    # Backward compatibility for the previous combined fee+slippage field.
    if "fee_slippage_per_side" in overrides and not ({"commission_pct_per_side", "slippage_pct_per_side"} & overrides.keys()):
        total_pct = float(overrides["fee_slippage_per_side"]) * 100
        overrides["commission_pct_per_side"] = total_pct
        overrides["slippage_pct_per_side"] = 0.0
    cfg.update(overrides)
    numeric = ["initial_balance", "risk_usd", "risk_pct", "reward_risk",
               "commission_pct_per_side", "slippage_pct_per_side",
               "min_win_rate_pct", "min_profit_factor", "min_tp_per_day"]
    for key in numeric:
        try: cfg[key] = float(cfg[key])
        except (TypeError, ValueError): raise ValueError(f"{key} must be numeric")
        if not math.isfinite(cfg[key]): raise ValueError(f"{key} must be finite")
    for key in ("use_min_win_rate", "use_min_profit_factor", "use_min_tp_per_day", "use_max_consecutive_losses", "break_even_enabled", "trailing_stop_enabled", "partial_exit_enabled", "time_exit_enabled", "compound"):
        cfg[key] = bool(cfg[key])
    if cfg["initial_balance"] <= 0: raise ValueError("Initial balance must be positive")
    if cfg["risk_mode"] not in SUPPORTED_RISK_MODES: raise ValueError("Unsupported risk_mode")
    if cfg["risk_usd"] <= 0 or cfg["risk_pct"] <= 0 or cfg["risk_pct"] > 100: raise ValueError("Risk must be positive; risk_pct must be <= 100")
    if cfg["reward_risk"] <= 0 or cfg["reward_risk"] > 100: raise ValueError("reward_risk must be in (0,100]")
    if min(cfg["commission_pct_per_side"], cfg["slippage_pct_per_side"]) < 0: raise ValueError("Costs cannot be negative")
    if cfg["commission_pct_per_side"] + cfg["slippage_pct_per_side"] >= 100: raise ValueError("Per-side costs must be below 100%")
    if cfg["target_mode"] not in SUPPORTED_TARGET_MODES: raise ValueError("Unsupported target_mode")
    if int(cfg["max_positions"]) != 1:
        raise ValueError("This engine currently supports exactly one open position; max_positions must be 1")
    cfg["max_positions"] = 1
    if cfg["entry_model"] != "next_candle_open": raise ValueError("Only next_candle_open is implemented")
    if cfg["exit_model"] != "sl_tp_only": raise ValueError("Only SL/TP exits are implemented in this engine")
    if any(cfg[k] for k in ("break_even_enabled","trailing_stop_enabled","partial_exit_enabled","time_exit_enabled")):
        raise ValueError("Selected trade-management add-on is not implemented in this engine yet; turn it off. No silent fallback is allowed.")
    if int(cfg["max_consecutive_losses"]) < 0: raise ValueError("max_consecutive_losses must be non-negative")
    cfg["max_consecutive_losses"] = int(cfg["max_consecutive_losses"])
    return cfg

@dataclass(frozen=True)
class StrategySpec:
    name: str = "EMA trend crossover"
    fast: int = 9
    slow: int = 36
    atr_period: int = 14
    stop_atr: float = 1.5
    direction: str = "both"
    seed: int = 42
    family: str = "ema_crossover"
    def validate(self):
        if not (2 <= self.fast < self.slow <= 500): raise ValueError("Require 2 <= fast < slow <= 500")
        if not (2 <= self.atr_period <= 200): raise ValueError("atr_period must be 2..200")
        if not (0.2 <= self.stop_atr <= 10): raise ValueError("stop_atr must be 0.2..10")
        if self.direction not in {"both","long","short"}: raise ValueError("Invalid direction")
        if self.family not in {"ema_crossover","channel_breakout","rsi_reversion","volume_breakout"}:
            raise ValueError("Unsupported strategy family")

def indicators(df, spec):
    x=df.copy(); c=x.close.astype(float); prev=c.shift(1)
    x["ema_fast"]=c.ewm(span=spec.fast,adjust=False,min_periods=spec.fast).mean()
    x["ema_slow"]=c.ewm(span=spec.slow,adjust=False,min_periods=spec.slow).mean()
    tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
    long_signal=(x.ema_fast>x.ema_slow)&(x.ema_fast.shift(1)<=x.ema_slow.shift(1))
    short_signal=(x.ema_fast<x.ema_slow)&(x.ema_fast.shift(1)>=x.ema_slow.shift(1))
    if spec.family in {"channel_breakout","volume_breakout"}:
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        long_signal=(c>upper)&(c.shift(1)<=upper.shift(1))
        short_signal=(c<lower)&(c.shift(1)>=lower.shift(1))
        if spec.family=="volume_breakout":
            baseline=x.volume.rolling(spec.fast,min_periods=spec.fast).mean().shift(1)
            active_volume=x.volume>=baseline*1.5
            long_signal=long_signal&active_volume
            short_signal=short_signal&active_volume
    elif spec.family=="rsi_reversion":
        delta=c.diff()
        gain=delta.clip(lower=0).ewm(alpha=1/spec.fast,adjust=False,min_periods=spec.fast).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/spec.fast,adjust=False,min_periods=spec.fast).mean()
        rs=gain/loss.replace(0,np.nan)
        rsi=100-100/(1+rs)
        long_signal=(rsi.shift(1)<=30)&(rsi>30)
        short_signal=(rsi.shift(1)>=70)&(rsi<70)
    x["long_signal"]=long_signal.fillna(False)
    x["short_signal"]=short_signal.fillna(False)
    return x

def _risk_budget(cfg: dict[str, Any], balance: float) -> float:
    if cfg["risk_mode"] == "fixed_usd": return cfg["risk_usd"]
    base = cfg["initial_balance"] if (cfg["risk_mode"] == "percent_initial" and not cfg["compound"]) else balance
    return base * cfg["risk_pct"] / 100.0

def _size_and_target(entry: float, stop: float, side: str, risk_budget: float, cfg: dict[str, Any]):
    entry_cost_rate=(cfg["commission_pct_per_side"]+cfg["slippage_pct_per_side"])/100.0
    exit_cost_rate=entry_cost_rate
    loss_distance=(entry-stop) if side=="LONG" else (stop-entry)
    if loss_distance <= 0: raise ValueError("Stop is on the wrong side of entry")
    # Size to the exact configured net risk budget using the modeled per-side cost rates.
    stop_cost=entry*entry_cost_rate+stop*exit_cost_rate
    qty=risk_budget/(loss_distance+stop_cost)
    if cfg["target_mode"] == "gross_rr":
        reward_distance=loss_distance*cfg["reward_risk"]
        target=entry+reward_distance if side=="LONG" else entry-reward_distance
    else:
        desired_net=risk_budget*cfg["reward_risk"]
        if side=="LONG":
            target=(entry*(1+entry_cost_rate)+desired_net/qty)/(1-exit_cost_rate)
        else:
            target=(entry*(1-entry_cost_rate)-desired_net/qty)/(1+exit_cost_rate)
        if target <= 0: raise ValueError("Configured net target implies non-positive price")
    return qty,target,entry_cost_rate,exit_cost_rate

def backtest(frame, spec=None, contract=None):
    spec=StrategySpec(**spec) if isinstance(spec,dict) else (spec or StrategySpec()); spec.validate()
    cfg=normalize_contract(contract)
    df=frame.copy().reset_index(drop=True)
    if len(df)>1:
        diffs=df.open_time.diff().dropna(); interval=int(diffs.mode().iloc[0]) if len(diffs) else None
        report=validate_ohlcv(df,interval,strict_gaps=True)
    else: report=validate_ohlcv(df,None)
    if len(df)<max(spec.slow,spec.atr_period)+3: raise DataIntegrityError("Not enough candles for warmup")
    data_hash=sha256_bytes(df.to_csv(index=False,float_format="%.12g").encode())
    x=indicators(df,spec); balance=float(cfg["initial_balance"]); ledger=[]; streak=0; max_streak=0
    i=max(spec.slow,spec.atr_period)
    while i < len(x)-1:
        row=x.iloc[i]
        if not np.isfinite(row.atr) or row.atr<=0: i+=1; continue
        side="LONG" if bool(row.long_signal) else ("SHORT" if bool(row.short_signal) else None)
        if side is None or (spec.direction=="long" and side=="SHORT") or (spec.direction=="short" and side=="LONG"): i+=1; continue
        entry_i=i+1; entry=float(x.iloc[entry_i].open); dist=float(row.atr*spec.stop_atr)
        stop=entry-dist if side=="LONG" else entry+dist
        risk_budget=_risk_budget(cfg,balance if cfg["compound"] or cfg["risk_mode"]=="percent_equity" else cfg["initial_balance"])
        if risk_budget<=0 or risk_budget>=balance and cfg["risk_mode"]=="percent_equity": i+=1; continue
        try: qty,target,entry_rate,exit_rate=_size_and_target(entry,stop,side,risk_budget,cfg)
        except ValueError: i+=1; continue
        if qty<=0: i+=1; continue
        exit_i=exit_price=None; reason=None
        for j in range(entry_i,len(x)):
            c=x.iloc[j]
            sl=float(c.low)<=stop if side=="LONG" else float(c.high)>=stop
            tp=float(c.high)>=target if side=="LONG" else float(c.low)<=target
            if sl and tp: raise DataIntegrityError(f"Unresolved same-candle SL/TP ambiguity at row {j}; finer data required")
            if sl: exit_i=j; exit_price=stop; reason="SL"; break
            if tp: exit_i=j; exit_price=target; reason="TP"; break
        if exit_i is None: break
        gross=(exit_price-entry)*qty*(1 if side=="LONG" else -1)
        entry_cost=entry*qty*entry_rate; exit_cost=exit_price*qty*exit_rate
        net=gross-entry_cost-exit_cost; balance+=net
        if reason=="TP": streak=0
        else: streak+=1; max_streak=max(max_streak,streak)
        ledger.append({"trade_id":len(ledger)+1,"signal_time":int(row.open_time),"entry_time":int(x.iloc[entry_i].open_time),"exit_time":int(x.iloc[exit_i].open_time),"direction":side,"signal_price":float(row.close),"entry_price":entry,"stop_loss":stop,"take_profit":target,"exit_price":float(exit_price),"exit_reason":reason,"quantity":qty,"risk_budget":risk_budget,"gross_pnl":gross,"entry_cost":entry_cost,"exit_cost":exit_cost,"fees_slippage":entry_cost+exit_cost,"net_pnl":net,"balance_after":balance,"loss_streak_after":streak})
        i=exit_i+1
    wins=[t for t in ledger if t["exit_reason"]=="TP" and t["net_pnl"]>0]
    gp=sum(max(0,t["net_pnl"]) for t in ledger); gl=abs(sum(min(0,t["net_pnl"]) for t in ledger))
    span_days=max((int(df.open_time.iloc[-1])-int(df.open_time.iloc[0]))/86400000,1/1440)
    wr=100*len(wins)/len(ledger) if ledger else 0.0; pf=gp/gl if gl else (math.inf if gp else 0.0); tp_day=len(wins)/span_days
    passes=(not cfg["use_min_win_rate"] or wr>cfg["min_win_rate_pct"]) and (not cfg["use_min_profit_factor"] or pf>cfg["min_profit_factor"]) and (not cfg["use_min_tp_per_day"] or tp_day>=cfg["min_tp_per_day"]) and (not cfg["use_max_consecutive_losses"] or max_streak<=cfg["max_consecutive_losses"])
    metrics={"trades":len(ledger),"wins":len(wins),"losses":sum(t["net_pnl"]<=0 for t in ledger),"win_rate_pct":wr,"profit_factor":pf if math.isfinite(pf) else None,"avg_net_tp_per_day":tp_day,"max_consecutive_losses":max_streak,"net_pnl":sum(t["net_pnl"] for t in ledger),"final_balance":balance,"total_costs":sum(t["fees_slippage"] for t in ledger),"data_hash_sha256":data_hash,"rows":len(df),"span_days":span_days,"passes_all_targets":bool(passes)}
    run_hash=sha256_bytes(json.dumps({"spec":asdict(spec),"contract":cfg,"data_hash":data_hash},sort_keys=True).encode())
    return {"strategy":asdict(spec),"contract":cfg,"metrics":metrics,"ledger":ledger,"data_report":report,"run_hash_sha256":run_hash}

def search_candidates(frame, max_candidates=500, seed=42, contract=None):
    if not 1<=max_candidates<=10000: raise ValueError("max_candidates must be 1..10000 in this starter API")
    cfg=normalize_contract(contract)
    rng=np.random.default_rng(seed); results=[]; errors=0
    families=("ema_crossover","channel_breakout","rsi_reversion","volume_breakout")
    for _ in range(max_candidates):
        family=str(rng.choice(families))
        if family=="ema_crossover":
            fast=int(rng.integers(4,40)); slow=int(rng.integers(max(fast+2,20),121))
            name_prefix="EMA crossover"
        elif family=="channel_breakout":
            slow=int(rng.integers(20,121)); fast=min(9,slow-2)
            name_prefix="Price channel breakout"
        elif family=="rsi_reversion":
            fast=int(rng.integers(4,40)); slow=fast+2
            name_prefix="RSI reversion"
        else:
            fast=int(rng.integers(4,40)); slow=int(rng.integers(max(fast+2,20),121))
            name_prefix="Volume-confirmed breakout"
        period=int(rng.integers(7,31)); mult=float(rng.choice([0.8,1.0,1.2,1.5,2.0,2.5]))
        spec=StrategySpec(f"{name_prefix} {fast}/{slow} ATR {period}x{mult:g}",fast,slow,period,mult,"both",seed,family)
        try:
            r=backtest(frame,spec,cfg); m=r["metrics"]
            failed=[]; criterion_distances={}
            if cfg["use_min_win_rate"]:
                gap=max(0.0,cfg["min_win_rate_pct"]+0.0001-m["win_rate_pct"])
                criterion_distances["win_rate_pct"]=gap/max(cfg["min_win_rate_pct"],1.0)
                if m["win_rate_pct"]<=cfg["min_win_rate_pct"]: failed.append("win_rate_pct")
            if cfg["use_min_profit_factor"]:
                pf=math.inf if m["profit_factor"] is None else float(m["profit_factor"])
                gap=max(0.0,cfg["min_profit_factor"]+0.0001-pf)
                criterion_distances["profit_factor"]=gap
                if pf<=cfg["min_profit_factor"]: failed.append("profit_factor")
            if cfg["use_min_tp_per_day"]:
                gap=max(0.0,cfg["min_tp_per_day"]-m["avg_net_tp_per_day"])
                criterion_distances["avg_net_tp_per_day"]=gap/max(cfg["min_tp_per_day"],1.0)
                if m["avg_net_tp_per_day"]<cfg["min_tp_per_day"]: failed.append("avg_net_tp_per_day")
            if cfg["use_max_consecutive_losses"]:
                gap=max(0.0,m["max_consecutive_losses"]-cfg["max_consecutive_losses"])
                criterion_distances["max_consecutive_losses"]=gap/max(cfg["max_consecutive_losses"],1)
                if m["max_consecutive_losses"]>cfg["max_consecutive_losses"]: failed.append("max_consecutive_losses")
            dist=sum(criterion_distances.values())
            results.append({"strategy":r["strategy"],"contract":cfg,"metrics":m,"distance":dist,
                            "failed_conditions":failed,"criterion_distances":criterion_distances,
                            "run_hash_sha256":r["run_hash_sha256"]})
        except (ValueError,DataIntegrityError,IndexError): errors+=1
    results.sort(key=lambda r:(not r["metrics"]["passes_all_targets"],r["distance"],r["metrics"]["max_consecutive_losses"],-r["metrics"]["win_rate_pct"],-r["metrics"]["avg_net_tp_per_day"],-(math.inf if r["metrics"]["profit_factor"] is None else float(r["metrics"]["profit_factor"]))))
    passing=[r for r in results if r["metrics"]["passes_all_targets"]]
    selected=passing if passing else results[:5]
    near_miss_shortfall=max(0,5-len(selected)) if not passing else 0
    return {"requested_candidates":max_candidates,"evaluated_candidates":len(results),"failed_candidates":errors,
            "seed":seed,"contract":cfg,"all_passing":bool(passing),"results":selected,
            "requested_near_misses":0 if passing else 5,"returned_near_misses":0 if passing else len(selected),
            "near_miss_shortfall":near_miss_shortfall,
            "warning":"Bounded random search only; ranked on TRAIN to preserve untouched OOS. No 500k+ or OOS certification."}
