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

STRATEGY_FAMILIES = (
    "ema_cross", "momentum", "rsi_reversal", "bollinger_breakout",
    "donchian_breakout", "volume_breakout", "candle_engulfing",
    "support_resistance", "channel_breakout", "macd_cross",
    "trend_pullback", "range_reversion", "candle_patterns", "classic_patterns", "stochastic_reversal", "vwap_reversion",
)
# Compose base entry families with confirmation/regime layers and parameter grids.
FILTER_FAMILIES = (
    "none", "trend_alignment", "volume_expansion", "atr_expansion",
    "candle_direction", "rsi_bias", "momentum_alignment", "ema_slope",
    "range_guard", "breakout_confirmation", "volume_and_trend",
    "trend_and_momentum", "candle_and_volume", "low_volatility",
    "trend_regime", "range_regime", "high_volatility_regime",
    "low_volatility_regime", "trend_or_high_vol",
)

@dataclass(frozen=True)
class StrategySpec:
    name: str = "EMA trend crossover"
    fast: int = 9
    slow: int = 36
    atr_period: int = 14
    stop_atr: float = 1.5
    direction: str = "both"
    seed: int = 42
    family: str = "ema_cross"
    filter_family: str = "none"
    lookback: int = 20
    rsi_period: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    volume_mult: float = 1.5
    band_std: float = 2.0
    momentum_threshold: float = 0.0
    def validate(self):
        if not (2 <= self.fast < self.slow <= 500): raise ValueError("Require 2 <= fast < slow <= 500")
        if not (2 <= self.atr_period <= 200): raise ValueError("atr_period must be 2..200")
        if not (0.2 <= self.stop_atr <= 10): raise ValueError("stop_atr must be 0.2..10")
        if self.direction not in {"both","long","short"}: raise ValueError("Invalid direction")
        if self.family not in STRATEGY_FAMILIES: raise ValueError(f"Unsupported strategy family: {self.family}")
        if self.filter_family not in FILTER_FAMILIES: raise ValueError(f"Unsupported filter family: {self.filter_family}")
        if not 3 <= self.lookback <= 500: raise ValueError("lookback must be 3..500")
        if not 2 <= self.rsi_period <= 100: raise ValueError("rsi_period must be 2..100")
        if not 0 < self.rsi_low < self.rsi_high < 100: raise ValueError("Require 0 < rsi_low < rsi_high < 100")
        if not 0.1 <= self.volume_mult <= 10: raise ValueError("volume_mult must be 0.1..10")
        if not 0.5 <= self.band_std <= 4: raise ValueError("band_std must be 0.5..4")
        if not -0.2 <= self.momentum_threshold <= 0.2: raise ValueError("momentum_threshold must be -0.2..0.2")

def indicators(df, spec):
    x=df.copy(); c=x.close.astype(float); o=x.open.astype(float); h=x.high.astype(float); l=x.low.astype(float); v=x.volume.astype(float); prev=c.shift(1)
    x["ema_fast"]=c.ewm(span=spec.fast,adjust=False,min_periods=spec.fast).mean()
    x["ema_slow"]=c.ewm(span=spec.slow,adjust=False,min_periods=spec.slow).mean()
    tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
    long_signal=(x.ema_fast>x.ema_slow)&(x.ema_fast.shift(1)<=x.ema_slow.shift(1))
    short_signal=(x.ema_fast<x.ema_slow)&(x.ema_fast.shift(1)>=x.ema_slow.shift(1))
    family=spec.family
    if family == "momentum":
        mom=c.pct_change(spec.lookback)
        long_signal=mom > spec.momentum_threshold; short_signal=mom < -spec.momentum_threshold
    elif family in {"rsi_reversal","range_reversion"}:
        delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/spec.rsi_period,adjust=False,min_periods=spec.rsi_period).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/spec.rsi_period,adjust=False,min_periods=spec.rsi_period).mean()
        rs=gain/loss.replace(0,np.nan); rsi=100-(100/(1+rs))
        if family == "rsi_reversal":
            long_signal=(rsi.shift(1)<spec.rsi_low)&(rsi>=spec.rsi_low)
            short_signal=(rsi.shift(1)>spec.rsi_high)&(rsi<=spec.rsi_high)
        else:
            long_signal=rsi<=spec.rsi_low; short_signal=rsi>=spec.rsi_high
    elif family == "bollinger_breakout":
        mid=c.rolling(spec.lookback,min_periods=spec.lookback).mean()
        sd=c.rolling(spec.lookback,min_periods=spec.lookback).std(ddof=0)
        upper=mid+spec.band_std*sd; lower=mid-spec.band_std*sd
        long_signal=(c>upper)&(prev<=upper.shift(1)); short_signal=(c<lower)&(prev>=lower.shift(1))
    elif family in {"donchian_breakout","channel_breakout","support_resistance"}:
        n=spec.lookback
        ceiling=h.shift(1).rolling(n,min_periods=n).max()
        floor=l.shift(1).rolling(n,min_periods=n).min()
        if family == "support_resistance":
            long_signal=(l<=floor*1.002)&(c>o)&(c.shift(1)<=o.shift(1))
            short_signal=(h>=ceiling*0.998)&(c<o)&(c.shift(1)>=o.shift(1))
        else:
            long_signal=(c>ceiling)&(prev<=ceiling)
            short_signal=(c<floor)&(prev>=floor)
    elif family == "volume_breakout":
        n=spec.lookback
        ceiling=h.shift(1).rolling(n,min_periods=n).max()
        floor=l.shift(1).rolling(n,min_periods=n).min()
        volbase=v.shift(1).rolling(n,min_periods=n).mean()
        active=v>volbase*spec.volume_mult
        long_signal=(c>ceiling)&(prev<=ceiling)&active
        short_signal=(c<floor)&(prev>=floor)&active
    elif family in {"candle_engulfing","candle_patterns"}:
        prev_open=o.shift(1); prev_close=c.shift(1)
        bullish=(c>o)&(prev_close<prev_open)&(o<=prev_close)&(c>=prev_open)
        bearish=(c<o)&(prev_close>prev_open)&(o>=prev_close)&(c<=prev_open)
        body=(c-o).abs().replace(0,np.nan); candle_range=(h-l).replace(0,np.nan)
        lower_wick=np.minimum(o,c)-l; upper_wick=h-np.maximum(o,c)
        hammer=(lower_wick>=2*body)&(upper_wick<=body)&(c>o)
        shooting_star=(upper_wick>=2*body)&(lower_wick<=body)&(c<o)
        inside=(h<h.shift(1))&(l>l.shift(1))
        inside_break_up=inside.shift(1).fillna(False)&(c>h.shift(2))
        inside_break_down=inside.shift(1).fillna(False)&(c<l.shift(2))
        if family=="candle_engulfing":
            long_signal=bullish; short_signal=bearish
        else:
            long_signal=bullish|hammer|inside_break_up
            short_signal=bearish|shooting_star|inside_break_down
    elif family == "classic_patterns":
        # Confirm pivots only after the right-hand candle has closed; no centered/future rolling window.
        pivot_high=(h.shift(2)>h.shift(3))&(h.shift(2)>h.shift(1))
        pivot_low=(l.shift(2)<l.shift(3))&(l.shift(2)<l.shift(1))
        ph=h.shift(2).where(pivot_high); pl=l.shift(2).where(pivot_low)
        prior_high=ph.shift(1).rolling(spec.lookback,min_periods=2).max()
        prior_low=pl.shift(1).rolling(spec.lookback,min_periods=2).min()
        tol=spec.band_std/100.0
        double_top=pivot_high&(ph>=prior_high*(1-tol))&(ph<=prior_high*(1+tol))
        double_bottom=pivot_low&(pl>=prior_low*(1-tol))&(pl<=prior_low*(1+tol))
        long_signal=double_bottom.shift(1).fillna(False)&(c>o)
        short_signal=double_top.shift(1).fillna(False)&(c<o)
    elif family == "stochastic_reversal":
        low_n=l.rolling(spec.lookback,min_periods=spec.lookback).min()
        high_n=h.rolling(spec.lookback,min_periods=spec.lookback).max()
        k=100*(c-low_n)/(high_n-low_n).replace(0,np.nan)
        dline=k.rolling(max(2,spec.rsi_period//3),min_periods=max(2,spec.rsi_period//3)).mean()
        long_signal=(k.shift(1)<spec.rsi_low)&(k>=spec.rsi_low)&(k>dline)
        short_signal=(k.shift(1)>spec.rsi_high)&(k<=spec.rsi_high)&(k<dline)
    elif family == "vwap_reversion":
        typical=(h+l+c)/3
        vol_sum=v.rolling(spec.lookback,min_periods=spec.lookback).sum()
        vwap=(typical*v).rolling(spec.lookback,min_periods=spec.lookback).sum()/vol_sum.replace(0,np.nan)
        sd=c.rolling(spec.lookback,min_periods=spec.lookback).std(ddof=0)
        long_signal=(c<vwap-spec.band_std*sd)&(c>o)
        short_signal=(c>vwap+spec.band_std*sd)&(c<o)
    elif family == "macd_cross":
        fast=c.ewm(span=spec.fast,adjust=False,min_periods=spec.fast).mean()
        slow=c.ewm(span=spec.slow,adjust=False,min_periods=spec.slow).mean()
        macd=fast-slow; signal=macd.ewm(span=max(2,spec.rsi_period//2),adjust=False,min_periods=max(2,spec.rsi_period//2)).mean()
        long_signal=(macd>signal)&(macd.shift(1)<=signal.shift(1))
        short_signal=(macd<signal)&(macd.shift(1)>=signal.shift(1))
    elif family == "trend_pullback":
        trend=x.ema_fast>x.ema_slow
        long_signal=trend&(l<=x.ema_fast)&(c>x.ema_fast)&(c.shift(1)<=x.ema_fast.shift(1))
        trend_down=x.ema_fast<x.ema_slow
        short_signal=trend_down&(h>=x.ema_fast)&(c<x.ema_fast)&(c.shift(1)>=x.ema_fast.shift(1))
    # Causal confirmation/regime layer: current and past candles only.
    filt=spec.filter_family
    if filt != "none":
        n=spec.lookback
        volbase=v.shift(1).rolling(n,min_periods=n).mean()
        atrbase=x["atr"].shift(1).rolling(n,min_periods=n).median()
        delta=c.diff()
        gain=delta.clip(lower=0).ewm(alpha=1/spec.rsi_period,adjust=False,min_periods=spec.rsi_period).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/spec.rsi_period,adjust=False,min_periods=spec.rsi_period).mean()
        rsi=100-(100/(1+gain/loss.replace(0,np.nan)))
        mom=c.pct_change(spec.lookback)
        trend_up=x.ema_fast>x.ema_slow; trend_down=x.ema_fast<x.ema_slow
        slope_n=max(2,spec.rsi_period//3)
        slope_up=x.ema_fast>x.ema_fast.shift(slope_n); slope_down=x.ema_fast<x.ema_fast.shift(slope_n)
        vol_active=v>volbase*spec.volume_mult
        atr_active=x.atr>atrbase
        trend_strength=(x.ema_fast-x.ema_slow).abs()/x.atr.replace(0,np.nan)
        high_vol_regime=(x.atr/c.abs().replace(0,np.nan))>(atrbase/c.abs().replace(0,np.nan))
        low_vol_regime=~high_vol_regime
        trend_regime=trend_strength>0.5
        range_regime=trend_strength<0.25
        bullish=c>o; bearish=c<o
        high_prev=h.shift(1).rolling(n,min_periods=n).max()
        low_prev=l.shift(1).rolling(n,min_periods=n).min()
        if filt=="trend_alignment": allow_l,allow_s=trend_up,trend_down
        elif filt=="volume_expansion": allow_l=allow_s=vol_active
        elif filt=="atr_expansion": allow_l=allow_s=atr_active
        elif filt=="candle_direction": allow_l,allow_s=bullish,bearish
        elif filt=="rsi_bias": allow_l,allow_s=rsi>=50,rsi<=50
        elif filt=="momentum_alignment": allow_l,allow_s=mom>0,mom<0
        elif filt=="ema_slope": allow_l,allow_s=slope_up,slope_down
        elif filt=="range_guard": allow_l=allow_s=(x.atr/c.abs().replace(0,np.nan)<0.02)
        elif filt=="breakout_confirmation": allow_l,allow_s=c>high_prev,c<low_prev
        elif filt=="volume_and_trend": allow_l,allow_s=vol_active&trend_up,vol_active&trend_down
        elif filt=="trend_and_momentum": allow_l,allow_s=trend_up&(mom>0),trend_down&(mom<0)
        elif filt=="candle_and_volume": allow_l,allow_s=bullish&vol_active,bearish&vol_active
        elif filt=="low_volatility": allow_l=allow_s=(x.atr/c.abs().replace(0,np.nan)<0.01)
        elif filt=="trend_regime": allow_l=allow_s=trend_regime
        elif filt=="range_regime": allow_l=allow_s=range_regime
        elif filt=="high_volatility_regime": allow_l=allow_s=high_vol_regime
        elif filt=="low_volatility_regime": allow_l=allow_s=low_vol_regime
        elif filt=="trend_or_high_vol": allow_l,allow_s=(trend_regime|high_vol_regime)&trend_up,(trend_regime|high_vol_regime)&trend_down
        else: raise ValueError(f"Unsupported filter family: {filt}")
        long_signal=long_signal & allow_l.fillna(False)
        short_signal=short_signal & allow_s.fillna(False)
    x["long_signal"]=long_signal.fillna(False).astype(bool)
    x["short_signal"]=short_signal.fillna(False).astype(bool)
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
    if not 1<=max_candidates<=500000: raise ValueError("max_candidates must be 1..500000")
    cfg=normalize_contract(contract)
    rng=np.random.default_rng(seed); results=[]; errors=0
    families=np.asarray(STRATEGY_FAMILIES,dtype=object)
    filters=np.asarray(FILTER_FAMILIES,dtype=object)
    for _ in range(max_candidates):
        fast=int(rng.integers(4,40)); slow=int(rng.integers(max(fast+2,20),121))
        period=int(rng.integers(7,31)); mult=float(rng.choice([0.8,1.0,1.2,1.5,2.0,2.5]))
        family=str(rng.choice(families)); filter_family=str(rng.choice(filters)); lookback=int(rng.integers(8,81))
        rsi_period=int(rng.integers(7,31)); rsi_low=float(rng.choice([20,25,30,35,40])); rsi_high=float(rng.choice([60,65,70,75,80]))
        if rsi_low>=rsi_high: rsi_low,rsi_high=30.0,70.0
        volume_mult=float(rng.choice([1.1,1.25,1.5,1.75,2.0,2.5]))
        band_std=float(rng.choice([1.0,1.5,2.0,2.25,2.5,3.0]))
        momentum_threshold=float(rng.choice([0.0,0.0005,0.001,0.002,0.003]))
        spec=StrategySpec(
            name=f"{family}+{filter_family} f{fast}/s{slow} lb{lookback} ATR {period}x{mult:g}",
            fast=fast,slow=slow,atr_period=period,stop_atr=mult,direction="both",seed=seed,
            family=family,filter_family=filter_family,lookback=lookback,rsi_period=rsi_period,rsi_low=rsi_low,
            rsi_high=rsi_high,volume_mult=volume_mult,band_std=band_std,
            momentum_threshold=momentum_threshold)
        try:
            r=backtest(frame,spec,cfg); m=r["metrics"]
            dist=(max(0,cfg["min_win_rate_pct"]+0.0001-m["win_rate_pct"])/max(cfg["min_win_rate_pct"],1) if cfg["use_min_win_rate"] else 0)+(max(0,cfg["min_profit_factor"]+0.0001-(m["profit_factor"] or 0)) if cfg["use_min_profit_factor"] else 0)+(max(0,cfg["min_tp_per_day"]-m["avg_net_tp_per_day"])/max(cfg["min_tp_per_day"],1) if cfg["use_min_tp_per_day"] else 0)+(max(0,m["max_consecutive_losses"]-cfg["max_consecutive_losses"])/max(cfg["max_consecutive_losses"],1) if cfg["use_max_consecutive_losses"] else 0)
            results.append({"strategy":r["strategy"],"contract":cfg,"metrics":m,"distance":dist,"run_hash_sha256":r["run_hash_sha256"]})
        except (ValueError,DataIntegrityError,IndexError,ZeroDivisionError,FloatingPointError): errors+=1
    results.sort(key=lambda r:(not r["metrics"]["passes_all_targets"],r["distance"],r["metrics"]["max_consecutive_losses"],-r["metrics"]["win_rate_pct"],-r["metrics"]["avg_net_tp_per_day"],-(r["metrics"]["profit_factor"] or 0)))
    passing=[r for r in results if r["metrics"]["passes_all_targets"]]
    selected=passing if passing else results[:5]
    return {"requested_candidates":max_candidates,"evaluated_candidates":len(results),"failed_candidates":errors,"seed":seed,"contract":cfg,"families_searched":list(STRATEGY_FAMILIES),"filter_families_searched":list(FILTER_FAMILIES),
        "combinatorial_genome_space":"base family × confirmation/regime layer × parameter grid (thousands of variants)",
        "all_passing":bool(passing),"results":selected,"warning":"Multi-family randomized search. Large candidate counts may be computationally expensive; run on TRAIN data only. OOS certification is separate."}
