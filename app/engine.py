from __future__ import annotations
import hashlib, json, math, time
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
    context_filter: str = "none"
    confirmation_filter: str = "none"
    def validate(self):
        if not (2 <= self.fast < self.slow <= 500): raise ValueError("Require 2 <= fast < slow <= 500")
        if not (2 <= self.atr_period <= 200): raise ValueError("atr_period must be 2..200")
        if not (0.2 <= self.stop_atr <= 10): raise ValueError("stop_atr must be 0.2..10")
        if self.direction not in {"both","long","short"}: raise ValueError("Invalid direction")
        if self.family not in {"ema_crossover","channel_breakout","rsi_reversion","volume_breakout","ema_pullback","macd_crossover","bollinger_reversion","inside_bar_breakout","engulfing_reversal","momentum_breakout","stochastic_reversion","roc_momentum","adx_trend","keltner_breakout","atr_squeeze_breakout","obv_trend","vwap_reclaim","pivot_reversal"}:
            raise ValueError("Unsupported strategy family")
        if self.context_filter not in {"none","ema_trend","adx_strength","volume_confirm","vwap_side","low_volatility"}:
            raise ValueError("Unsupported context filter")
        if self.confirmation_filter not in {"none","candle_direction","momentum","rsi_midline"}:
            raise ValueError("Unsupported confirmation filter")

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
    elif spec.family=="ema_pullback":
        trend_up=x.ema_fast>x.ema_slow; trend_down=x.ema_fast<x.ema_slow
        long_signal=trend_up&(c> x.ema_fast)&(c.shift(1)<=x.ema_fast.shift(1))
        short_signal=trend_down&(c< x.ema_fast)&(c.shift(1)>=x.ema_fast.shift(1))
    elif spec.family=="macd_crossover":
        macd=c.ewm(span=spec.fast,adjust=False,min_periods=spec.fast).mean()-c.ewm(span=spec.slow,adjust=False,min_periods=spec.slow).mean()
        signal=macd.ewm(span=max(2,spec.fast//2),adjust=False,min_periods=max(2,spec.fast//2)).mean()
        long_signal=(macd>signal)&(macd.shift(1)<=signal.shift(1))
        short_signal=(macd<signal)&(macd.shift(1)>=signal.shift(1))
    elif spec.family=="bollinger_reversion":
        mid=c.rolling(spec.slow,min_periods=spec.slow).mean(); dev=c.rolling(spec.slow,min_periods=spec.slow).std(ddof=0)
        upper=mid+2*dev; lower=mid-2*dev
        long_signal=(c>lower)&(c.shift(1)<=lower.shift(1))
        short_signal=(c<upper)&(c.shift(1)>=upper.shift(1))
    elif spec.family=="inside_bar_breakout":
        inside=(x.high.shift(1)<x.high.shift(2))&(x.low.shift(1)>x.low.shift(2))
        long_signal=inside&(c>x.high.shift(1))&(c.shift(1)<=x.high.shift(1))
        short_signal=inside&(c<x.low.shift(1))&(c.shift(1)>=x.low.shift(1))
    elif spec.family=="engulfing_reversal":
        prev_open=x.open.shift(1); prev_close=c.shift(1)
        bullish=(c>x.open)&(prev_close<prev_open)&(x.open<=prev_close)&(c>=prev_open)
        bearish=(c<x.open)&(prev_close>prev_open)&(x.open>=prev_close)&(c<=prev_open)
        long_signal=bullish; short_signal=bearish
    elif spec.family=="momentum_breakout":
        momentum=c>c.shift(spec.fast); negative=c<c.shift(spec.fast)
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        long_signal=momentum&(c>upper)&(c.shift(1)<=upper.shift(1))
        short_signal=negative&(c<lower)&(c.shift(1)>=lower.shift(1))
    elif spec.family=="stochastic_reversion":
        lo=x.low.rolling(spec.fast,min_periods=spec.fast).min()
        hi=x.high.rolling(spec.fast,min_periods=spec.fast).max()
        k=100*(c-lo)/(hi-lo).replace(0,np.nan)
        long_signal=(k.shift(1)<=20)&(k>20)
        short_signal=(k.shift(1)>=80)&(k<80)
    elif spec.family=="roc_momentum":
        roc=c.pct_change(periods=spec.fast)*100
        long_signal=(roc>0)&(roc.shift(1)<=0)
        short_signal=(roc<0)&(roc.shift(1)>=0)
    elif spec.family=="adx_trend":
        up=x.high.diff(); down=-x.low.diff()
        plus=up.where((up>down)&(up>0),0.0).ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
        minus=down.where((down>up)&(down>0),0.0).ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
        plus_di=100*plus/x.atr.replace(0,np.nan); minus_di=100*minus/x.atr.replace(0,np.nan)
        dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
        adx=dx.ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
        long_signal=(plus_di>minus_di)&(plus_di.shift(1)<=minus_di.shift(1))&(adx>=20)
        short_signal=(minus_di>plus_di)&(minus_di.shift(1)<=plus_di.shift(1))&(adx>=20)
    elif spec.family=="keltner_breakout":
        mid=c.ewm(span=spec.slow,adjust=False,min_periods=spec.slow).mean()
        upper=mid+1.5*x.atr; lower=mid-1.5*x.atr
        long_signal=(c>upper)&(c.shift(1)<=upper.shift(1))
        short_signal=(c<lower)&(c.shift(1)>=lower.shift(1))
    elif spec.family=="atr_squeeze_breakout":
        mid=c.rolling(spec.slow,min_periods=spec.slow).mean()
        dev=c.rolling(spec.slow,min_periods=spec.slow).std(ddof=0)
        width=(4*dev/mid.replace(0,np.nan))
        squeezed=width<width.rolling(spec.slow,min_periods=spec.slow).quantile(0.25).shift(1)
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        long_signal=squeezed.shift(1)&(c>upper)&(c.shift(1)<=upper.shift(1))
        short_signal=squeezed.shift(1)&(c<lower)&(c.shift(1)>=lower.shift(1))
    elif spec.family=="obv_trend":
        signed=np.sign(c.diff()).fillna(0.0)*x.volume.astype(float)
        obv=signed.cumsum()
        obv_base=obv.ewm(span=spec.slow,adjust=False,min_periods=spec.slow).mean()
        long_signal=(obv>obv_base)&(obv.shift(1)<=obv_base.shift(1))&(c>x.ema_fast)
        short_signal=(obv<obv_base)&(obv.shift(1)>=obv_base.shift(1))&(c<x.ema_fast)
    elif spec.family=="vwap_reclaim":
        typical=(x.high+x.low+c)/3
        vol=x.volume.astype(float)
        vw=(typical*vol).rolling(spec.slow,min_periods=spec.slow).sum()/vol.rolling(spec.slow,min_periods=spec.slow).sum().replace(0,np.nan)
        long_signal=(c>vw)&(c.shift(1)<=vw.shift(1))
        short_signal=(c<vw)&(c.shift(1)>=vw.shift(1))
    elif spec.family=="pivot_reversal":
        support=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        resistance=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        tolerance=x.atr*0.25
        long_signal=(x.low<=support+tolerance)&(c>support)
        short_signal=(x.high>=resistance-tolerance)&(c<resistance)
    elif spec.family=="ema_pullback":
        trend_up=x.ema_fast>x.ema_slow; trend_down=x.ema_fast<x.ema_slow
        long_signal=trend_up&(c>x.ema_fast)&(c.shift(1)<=x.ema_fast.shift(1))
        short_signal=trend_down&(c<x.ema_fast)&(c.shift(1)>=x.ema_fast.shift(1))
    elif spec.family=="macd_crossover":
        macd=c.ewm(span=spec.fast,adjust=False,min_periods=spec.fast).mean()-c.ewm(span=spec.slow,adjust=False,min_periods=spec.slow).mean()
        signal=macd.ewm(span=max(2,spec.fast//2),adjust=False,min_periods=max(2,spec.fast//2)).mean()
        long_signal=(macd>signal)&(macd.shift(1)<=signal.shift(1))
        short_signal=(macd<signal)&(macd.shift(1)>=signal.shift(1))
    elif spec.family=="bollinger_reversion":
        mid=c.rolling(spec.slow,min_periods=spec.slow).mean(); dev=c.rolling(spec.slow,min_periods=spec.slow).std(ddof=0)
        upper=mid+2*dev; lower=mid-2*dev
        long_signal=(c>lower)&(c.shift(1)<=lower.shift(1))
        short_signal=(c<upper)&(c.shift(1)>=upper.shift(1))
    elif spec.family=="inside_bar_breakout":
        inside=(x.high.shift(1)<x.high.shift(2))&(x.low.shift(1)>x.low.shift(2))
        long_signal=inside&(c>x.high.shift(1))&(c.shift(1)<=x.high.shift(1))
        short_signal=inside&(c<x.low.shift(1))&(c.shift(1)>=x.low.shift(1))
    elif spec.family=="engulfing_reversal":
        prev_open=x.open.shift(1); prev_close=c.shift(1)
        long_signal=(c>x.open)&(prev_close<prev_open)&(x.open<=prev_close)&(c>=prev_open)
        short_signal=(c<x.open)&(prev_close>prev_open)&(x.open>=prev_close)&(c<=prev_open)
    elif spec.family=="momentum_breakout":
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        long_signal=(c>c.shift(spec.fast))&(c>upper)&(c.shift(1)<=upper.shift(1))
        short_signal=(c<c.shift(spec.fast))&(c<lower)&(c.shift(1)>=lower.shift(1))
    if spec.context_filter=="ema_trend":
        long_signal=long_signal&(x.ema_fast>x.ema_slow)
        short_signal=short_signal&(x.ema_fast<x.ema_slow)
    elif spec.context_filter=="adx_strength":
        up=x.high.diff(); down=-x.low.diff()
        plus=up.where((up>down)&(up>0),0.0).ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
        minus=down.where((down>up)&(down>0),0.0).ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
        pdi=100*plus/x.atr.replace(0,np.nan); mdi=100*minus/x.atr.replace(0,np.nan)
        dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
        adx=dx.ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
        long_signal=long_signal&(adx>=20)&(pdi>mdi)
        short_signal=short_signal&(adx>=20)&(mdi>pdi)
    elif spec.context_filter=="volume_confirm":
        baseline=x.volume.rolling(spec.fast,min_periods=spec.fast).mean().shift(1)
        active=x.volume>=baseline*1.5
        long_signal=long_signal&active; short_signal=short_signal&active
    elif spec.context_filter=="vwap_side":
        typical=(x.high+x.low+c)/3; vol=x.volume.astype(float)
        vw=(typical*vol).rolling(spec.slow,min_periods=spec.slow).sum()/vol.rolling(spec.slow,min_periods=spec.slow).sum().replace(0,np.nan)
        long_signal=long_signal&(c>vw); short_signal=short_signal&(c<vw)
    elif spec.context_filter=="low_volatility":
        ratio=x.atr/c.replace(0,np.nan)
        threshold=ratio.rolling(spec.slow,min_periods=spec.slow).quantile(0.35).shift(1)
        calm=ratio<=threshold
        long_signal=long_signal&calm; short_signal=short_signal&calm
    if spec.confirmation_filter=="candle_direction":
        long_signal=long_signal&(c>x.open); short_signal=short_signal&(c<x.open)
    elif spec.confirmation_filter=="momentum":
        long_signal=long_signal&(c>c.shift(spec.fast)); short_signal=short_signal&(c<c.shift(spec.fast))
    elif spec.confirmation_filter=="rsi_midline":
        delta=c.diff()
        gain=delta.clip(lower=0).ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/spec.atr_period,adjust=False,min_periods=spec.atr_period).mean()
        rsi=100-100/(1+gain/loss.replace(0,np.nan))
        long_signal=long_signal&(rsi>50); short_signal=short_signal&(rsi<50)
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

def _candidate_distance(metrics: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, list[str], dict[str, float]]:
    failed: list[str] = []
    distances: dict[str, float] = {}
    if cfg["use_min_win_rate"]:
        gap=max(0.0,cfg["min_win_rate_pct"]+0.0001-float(metrics["win_rate_pct"]))
        distances["win_rate_pct"]=gap/max(cfg["min_win_rate_pct"],1.0)
        if float(metrics["win_rate_pct"])<=cfg["min_win_rate_pct"]: failed.append("win_rate_pct")
    if cfg["use_min_profit_factor"]:
        pf=math.inf if metrics["profit_factor"] is None else float(metrics["profit_factor"])
        gap=max(0.0,cfg["min_profit_factor"]+0.0001-pf)
        distances["profit_factor"]=gap
        if pf<=cfg["min_profit_factor"]: failed.append("profit_factor")
    if cfg["use_min_tp_per_day"]:
        tp_day=float(metrics["avg_net_tp_per_day"])
        gap=max(0.0,cfg["min_tp_per_day"]-tp_day)
        distances["avg_net_tp_per_day"]=gap/max(cfg["min_tp_per_day"],1.0)
        if tp_day<cfg["min_tp_per_day"]: failed.append("avg_net_tp_per_day")
    if cfg["use_max_consecutive_losses"]:
        streak=int(metrics["max_consecutive_losses"])
        gap=max(0.0,streak-int(cfg["max_consecutive_losses"]))
        distances["max_consecutive_losses"]=gap/max(int(cfg["max_consecutive_losses"]),1)
        if streak>int(cfg["max_consecutive_losses"]): failed.append("max_consecutive_losses")
    return sum(distances.values()), failed, distances


def search_candidates(frame, max_candidates=500, seed=42, contract=None, algorithm="surrogate_genetic", search_space=None, progress_callback=None):
    """Deterministic bounded evolutionary search on TRAIN only.

    max_candidates counts distinct parameter genomes actually evaluated. This is
    deliberately capped by the API until resumable, resource-bounded batch jobs
    are available; this function does not claim to support 500k candidates yet.
    """
    if not 1<=max_candidates<=10000: raise ValueError("max_candidates must be 1..10000 until resumable batch execution is enabled")
    if algorithm not in {"hybrid_genetic","surrogate_genetic","stratified_random"}:
        raise ValueError("Unsupported search algorithm")
    cfg=normalize_contract(contract)
    rng=np.random.default_rng(seed)
    all_families=("ema_crossover","channel_breakout","rsi_reversion","volume_breakout","ema_pullback","macd_crossover","bollinger_reversion","inside_bar_breakout","engulfing_reversal","momentum_breakout","stochastic_reversion","roc_momentum","adx_trend","keltner_breakout","atr_squeeze_breakout","obv_trend","vwap_reclaim","pivot_reversal")
    all_context_filters=("none","ema_trend","adx_strength","volume_confirm","vwap_side","low_volatility")
    all_confirmation_filters=("none","candle_direction","momentum","rsi_midline")
    space=dict(search_space or {})
    families=tuple(dict.fromkeys(space["families"])) if "families" in space else all_families
    context_filters=tuple(dict.fromkeys(space["context_filters"])) if "context_filters" in space else all_context_filters
    confirmation_filters=tuple(dict.fromkeys(space["confirmation_filters"])) if "confirmation_filters" in space else all_confirmation_filters
    if not families or any(f not in all_families for f in families): raise ValueError("Search family selection is empty or contains unsupported families")
    if not context_filters or any(f not in all_context_filters for f in context_filters): raise ValueError("Context filter selection is empty or unsupported")
    if not confirmation_filters or any(f not in all_confirmation_filters for f in confirmation_filters): raise ValueError("Confirmation filter selection is empty or unsupported")
    names={"ema_crossover":"EMA crossover","channel_breakout":"Price channel breakout","rsi_reversion":"RSI reversion","volume_breakout":"Volume-confirmed breakout","ema_pullback":"EMA trend pullback","macd_crossover":"MACD crossover","bollinger_reversion":"Bollinger mean reversion","inside_bar_breakout":"Inside-bar breakout","engulfing_reversal":"Engulfing candle reversal","momentum_breakout":"Momentum-confirmed breakout","stochastic_reversion":"Stochastic reversion","roc_momentum":"Rate-of-change momentum","adx_trend":"ADX directional trend","keltner_breakout":"Keltner channel breakout","atr_squeeze_breakout":"Volatility squeeze breakout","obv_trend":"OBV volume trend","vwap_reclaim":"VWAP reclaim","pivot_reversal":"Support/resistance pivot reversal"}
    fast_values=np.array([3,4,5,6,8,9,10,12,14,16,18,21,24,28,32,36,40,48,55,64,72,89],dtype=int)
    slow_values=np.array([12,16,18,20,24,28,32,36,40,48,55,64,72,89,100,120,144,180,200,240,300,400,500],dtype=int)
    periods=np.array([5,7,9,10,12,14,16,18,21,24,28,30,40,50],dtype=int)
    stops=np.array([0.5,0.7,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0,3.5,4.0,5.0,6.0],dtype=float)
    directions=("both","long","short")
    evaluated:dict[tuple,dict[str,Any]]={}
    errors=0
    family_counts={f:0 for f in families}
    duplicate_genomes=0
    started_at=time.monotonic()
    last_progress_count=-1

    def emit_progress(force=False):
        nonlocal last_progress_count
        if progress_callback is None: return
        count=len(evaluated)
        if not force and count%25!=0: return
        if count==last_progress_count and not force: return
        last_progress_count=count
        progress_callback({"status":"running","requested_candidates":max_candidates,
                           "evaluated_candidates":count,"failed_candidates":errors,
                           "duplicate_genomes_skipped":duplicate_genomes,
                           "elapsed_seconds":round(time.monotonic()-started_at,2),
                           "family_evaluated_counts":dict(family_counts)})

    def make_spec(family=None, genes=None):
        if genes is not None:
            family,fast,slow,period,stop,direction,context_filter,confirmation_filter=genes
        else:
            family=str(family or rng.choice(families))
            fast=int(rng.choice(fast_values))
            slow=int(rng.choice(slow_values))
            if slow<=fast: slow=int(min(500,fast+max(2,int(rng.choice([4,8,12,16,24,32,48])))))
            period=int(rng.choice(periods)); stop=float(rng.choice(stops)); direction=str(rng.choice(directions))
            context_filter=str(rng.choice(context_filters)); confirmation_filter=str(rng.choice(confirmation_filters))
        fast=int(fast); slow=int(slow); period=int(period); stop=float(stop); direction=str(direction)
        if family=="channel_breakout": fast=min(fast, max(2,slow-2))
        elif family=="rsi_reversion": slow=max(fast+2, min(500,fast*2))
        elif family in {"inside_bar_breakout","engulfing_reversal"}: fast=9; slow=max(20,slow)
        if slow<=fast: fast=max(2,slow-2)
        return StrategySpec(f"{names[family]} {fast}/{slow} ATR {period}x{stop:g} {direction} [ctx={context_filter}; confirm={confirmation_filter}]",fast,slow,period,stop,direction,seed,family,context_filter,confirmation_filter)

    def genome(spec):
        return (spec.family,spec.fast,spec.slow,spec.atr_period,round(spec.stop_atr,6),spec.direction,spec.context_filter,spec.confirmation_filter)

    def fitness(row):
        m=row["metrics"]; pf=math.inf if m["profit_factor"] is None else float(m["profit_factor"])
        return (not m["passes_all_targets"],row["distance"],int(m["max_consecutive_losses"]),
                -min(float(m["win_rate_pct"]),100.0),-min(float(m["avg_net_tp_per_day"]),1000.0),
                -min(pf,1000.0),-int(m["trades"]))

    def feature_vector(spec):
        vals=[1.0 if spec.family==f else 0.0 for f in all_families]
        vals.extend([spec.fast/500.0,spec.slow/500.0,spec.atr_period/200.0,spec.stop_atr/10.0])
        vals.extend([1.0 if spec.direction==d else 0.0 for d in directions])
        vals.extend([1.0 if spec.context_filter==f else 0.0 for f in all_context_filters])
        vals.extend([1.0 if spec.confirmation_filter==f else 0.0 for f in all_confirmation_filters])
        return vals

    def surrogate_target(row):
        m=row["metrics"]
        pf=10.0 if m["profit_factor"] is None else min(10.0,float(m["profit_factor"]))
        return (float(row["distance"])+(0.0 if m["passes_all_targets"] else 1.0)
                +min(float(m["max_consecutive_losses"]),100.0)*0.01
                -min(float(m["win_rate_pct"]),100.0)*0.001
                -min(float(m["avg_net_tp_per_day"]),100.0)*0.002-pf*0.01)

    last_model_fit_at=-1
    surrogate_fit_count=0

    def evaluate(spec):
        nonlocal errors,duplicate_genomes
        key=genome(spec)
        if key in evaluated:
            duplicate_genomes+=1
            return evaluated[key]
        try:
            r=backtest(frame,spec,cfg); m=r["metrics"]
            dist,failed,distances=_candidate_distance(m,cfg)
            row={"strategy":r["strategy"],"contract":cfg,"metrics":m,"distance":dist,
                 "failed_conditions":failed,"criterion_distances":distances,
                 "run_hash_sha256":r["run_hash_sha256"],"_genome":key}
            evaluated[key]=row; family_counts[spec.family]+=1
            return row
        except (ValueError,DataIntegrityError,IndexError,KeyError,FloatingPointError):
            errors+=1
            return None

    initial=[]
    quota=max(1,min(max_candidates//len(families),max(1,64//len(families))))
    for family in families:
        for _ in range(quota):
            if len(initial)>=max_candidates: break
            initial.append(make_spec(family=family))
    while len(initial)<min(max_candidates,64):
        initial.append(make_spec())
    population=[]
    for spec in initial:
        if len(evaluated)>=max_candidates: break
        row=evaluate(spec)
        if row is not None: population.append((spec,row))
        emit_progress()

    generation=0
    while len(evaluated)<max_candidates:
        generation+=1
        if algorithm=="stratified_random" or len(population)<8:
            spec=make_spec()
        else:
            ranked=sorted(population,key=lambda pair:fitness(pair[1]))
            elite_count=max(4,min(48,int(math.sqrt(max_candidates))+4,len(ranked)))
            elites=ranked[:elite_count]
            if rng.random()<0.20:
                spec=make_spec()
            else:
                def parent():
                    k=min(4,len(elites))
                    contestants=[elites[int(rng.integers(0,len(elites)))] for _ in range(k)]
                    return min(contestants,key=lambda pair:fitness(pair[1]))[0]
                a=parent(); b=parent()
                family=str(rng.choice([a.family,b.family])) if rng.random()<0.55 else str(rng.choice(families))
                fast=int(a.fast if rng.random()<0.5 else b.fast)
                slow=int(a.slow if rng.random()<0.5 else b.slow)
                period=int(a.atr_period if rng.random()<0.5 else b.atr_period)
                stop=float(a.stop_atr if rng.random()<0.5 else b.stop_atr)
                direction=str(a.direction if rng.random()<0.5 else b.direction)
                context_filter=str(a.context_filter if rng.random()<0.5 else b.context_filter)
                confirmation_filter=str(a.confirmation_filter if rng.random()<0.5 else b.confirmation_filter)
                if rng.random()<0.35: fast=int(rng.choice(fast_values))
                if rng.random()<0.35: slow=int(rng.choice(slow_values))
                if rng.random()<0.25: period=int(rng.choice(periods))
                if rng.random()<0.25: stop=float(rng.choice(stops))
                if rng.random()<0.18: direction=str(rng.choice(directions))
                if rng.random()<0.12: family=str(rng.choice(families))
                if rng.random()<0.25: context_filter=str(rng.choice(context_filters))
                if rng.random()<0.20: confirmation_filter=str(rng.choice(confirmation_filters))
                spec=make_spec(genes=(family,fast,slow,period,stop,direction,context_filter,confirmation_filter))
        if algorithm=="surrogate_genetic" and len(evaluated)>=32 and (last_model_fit_at<0 or len(evaluated)-last_model_fit_at>=64):
            try:
                from sklearn.ensemble import ExtraTreesRegressor
                training_rows=list(evaluated.values())
                if len(training_rows)>2000: training_rows=training_rows[-2000:]
                X=[]; y=[]
                for training_row in training_rows:
                    training_spec=StrategySpec(**training_row["strategy"])
                    X.append(feature_vector(training_spec)); y.append(surrogate_target(training_row))
                model=ExtraTreesRegressor(n_estimators=48,min_samples_leaf=2,max_features=0.9,
                                          random_state=int(seed+len(evaluated)),n_jobs=1)
                model.fit(np.asarray(X,dtype=float),np.asarray(y,dtype=float))
                proposals=[spec]
                for _ in range(11):
                    proposals.append(make_spec())
                unseen=[candidate for candidate in proposals if genome(candidate) not in evaluated]
                if unseen:
                    predictions=model.predict(np.asarray([feature_vector(candidate) for candidate in unseen],dtype=float))
                    spec=unseen[int(np.argmin(predictions))]
                last_model_fit_at=len(evaluated); surrogate_fit_count+=1
            except ImportError as exc:
                raise ValueError("Surrogate ML search requires scikit-learn; install the declared requirements and restart. Search stopped without fallback.") from exc
        if genome(spec) in evaluated:
            duplicate_genomes+=1
            for _ in range(12):
                spec=make_spec()
                if genome(spec) not in evaluated: break
            if genome(spec) in evaluated:
                if duplicate_genomes>max_candidates*30: break
                continue
        row=evaluate(spec)
        if row is not None:
            population.append((spec,row))
            if len(population)>256:
                population=sorted(population,key=lambda pair:fitness(pair[1]))[:192]
        emit_progress()
        if errors+duplicate_genomes>max_candidates*30: break

    emit_progress(force=True)
    results=list(evaluated.values())
    results.sort(key=fitness)
    passing=[r for r in results if r["metrics"]["passes_all_targets"]]
    # Independently replay every candidate that will be presented as passing.
    # If no candidate passes, independently replay candidates in rank order until
    # five verifiable near-misses are collected. This is separate code in audit.py.
    from .audit import independent_replay, FIELDS
    audit_failures=[]
    audited=[]
    audit_pool=passing if passing else results
    for row in audit_pool:
        try:
            primary=backtest(frame,StrategySpec(**row["strategy"]),cfg)
            second=independent_replay(frame,row["strategy"],cfg)
            a=primary["ledger"]; b=second["ledger"]
            matched=len(a)==len(b)
            if matched:
                for ta,tb in zip(a,b):
                    for field in FIELDS:
                        va,vb=ta[field],tb[field]
                        if isinstance(va,(int,float)) and not isinstance(va,bool):
                            if not math.isclose(float(va),float(vb),rel_tol=1e-10,abs_tol=1e-8):
                                matched=False; break
                        elif va!=vb:
                            matched=False; break
                    if not matched: break
            row["independent_audit"]={"status":"MATCH" if matched else "MISMATCH",
                                      "primary_trades":len(a),"independent_trades":len(b),
                                      "ledger_fields_checked":len(FIELDS),
                                      "engine":"app.audit.independent_replay"}
            if matched:
                audited.append(row)
            else:
                audit_failures.append({"strategy":row["strategy"],"reason":"Trade ledger mismatch"})
        except Exception as exc:
            row["independent_audit"]={"status":"FAILED","reason":str(exc),"engine":"app.audit.independent_replay"}
            audit_failures.append({"strategy":row["strategy"],"reason":str(exc)})
        if not passing and len(audited)>=5:
            break
    if passing:
        selected=[r for r in audited if r["metrics"]["passes_all_targets"]]
        all_passing=bool(selected)
        if not selected:
            # Fail closed: do not call a strategy passing if independent replay disagrees.
            selected=[]
    else:
        selected=audited[:5]
        all_passing=False
    near_miss_shortfall=max(0,5-len(selected)) if not all_passing else 0
    for row in results: row.pop("_genome",None)
    return {"requested_candidates":max_candidates,"evaluated_candidates":len(results),
            "failed_candidates":errors,"duplicate_genomes_skipped":duplicate_genomes,
            "seed":seed,"algorithm":algorithm,"generations":generation,"surrogate_model_fits":surrogate_fit_count,
            "family_evaluated_counts":family_counts,"contract":cfg,"all_passing":all_passing,
            "independent_audit_failures":audit_failures,
            "results":selected,"requested_near_misses":0 if all_passing else 5,
            "returned_near_misses":0 if all_passing else len(selected),"near_miss_shortfall":near_miss_shortfall,
            "warning":("Deterministic stratified random baseline; selected results independently replayed; OOS not certified." if algorithm=="stratified_random" else
                       ("Surrogate-assisted genetic search: ExtraTrees model ranks proposal pools; model fits and independent ledger replay are recorded. Bounded to 10,000 candidates pending resumable resource-managed batch jobs; ranked on TRAIN only. Not 500k-certified." if algorithm=="surrogate_genetic" else
                        "Hybrid genetic search with stratified seeds, crossover, mutation, immigrant exploration and independent ledger replay of reported results. Bounded to 10,000 candidates pending resumable resource-managed batch jobs; ranked on TRAIN only. Not 500k-certified.")) }
