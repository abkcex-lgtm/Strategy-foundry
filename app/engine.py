from __future__ import annotations
import hashlib, json, math, time
from collections import OrderedDict
from dataclasses import dataclass, asdict, replace
from typing import Any
import numpy as np
import pandas as pd
from .data import validate_ohlcv, sha256_bytes, DataIntegrityError, download_binance_execution_window

# Preset only: every field is user-configurable through Settings/API.
CONTRACT = {
    "profile_name": "ETHUSDT fixed-risk 3.5R (editable preset)",
    "symbol": "ETHUSDT",
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
    "min_profit_factor": 1.6,
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
    if not str(cfg.get("symbol", "ETHUSDT")).isalnum(): raise ValueError("symbol must be alphanumeric")
    cfg["symbol"] = str(cfg.get("symbol", "ETHUSDT")).upper()
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
    secondary_family: str = "none"
    signal_composition: str = "single"
    def validate(self):
        if not (2 <= self.fast < self.slow <= 500): raise ValueError("Require 2 <= fast < slow <= 500")
        if not (2 <= self.atr_period <= 200): raise ValueError("atr_period must be 2..200")
        if not (0.2 <= self.stop_atr <= 10): raise ValueError("stop_atr must be 0.2..10")
        if self.direction not in {"both","long","short"}: raise ValueError("Invalid direction")
        if self.family not in {"ema_crossover","channel_breakout","rsi_reversion","volume_breakout","ema_pullback","macd_crossover","bollinger_reversion","inside_bar_breakout","engulfing_reversal","momentum_breakout","stochastic_reversion","roc_momentum","adx_trend","keltner_breakout","atr_squeeze_breakout","obv_trend","vwap_reclaim","pivot_reversal","swing_structure_break","failed_breakout_reversal","wick_rejection","range_expansion_breakout","three_bar_momentum","volatility_regime_breakout","trend_exhaustion_reversal","volume_climax_reversal","range_compression_breakout","support_resistance_rejection","micro_double_bottom","micro_double_top","williams_r_reversion","donchian_midline_trend","dual_thrust_breakout","chaikin_money_flow","money_flow_index_reversion","atr_trend_pullback","volume_weighted_breakout","range_midpoint_reversion","roc_acceleration","atr_normalized_momentum"}:
            raise ValueError("Unsupported strategy family")
        valid_families={"ema_crossover","channel_breakout","rsi_reversion","volume_breakout","ema_pullback","macd_crossover","bollinger_reversion","inside_bar_breakout","engulfing_reversal","momentum_breakout","stochastic_reversion","roc_momentum","adx_trend","keltner_breakout","atr_squeeze_breakout","obv_trend","vwap_reclaim","pivot_reversal","swing_structure_break","failed_breakout_reversal","wick_rejection","range_expansion_breakout","three_bar_momentum","volatility_regime_breakout","trend_exhaustion_reversal","volume_climax_reversal","range_compression_breakout","support_resistance_rejection","micro_double_bottom","micro_double_top","williams_r_reversion","donchian_midline_trend","dual_thrust_breakout","chaikin_money_flow","money_flow_index_reversion","atr_trend_pullback","volume_weighted_breakout","range_midpoint_reversion","roc_acceleration","atr_normalized_momentum"}
        if self.secondary_family not in valid_families | {"none"}: raise ValueError("Unsupported secondary strategy family")
        if self.signal_composition not in {"single","and","or","xor"}: raise ValueError("Unsupported signal composition")
        if self.signal_composition=="single" and self.secondary_family!="none": raise ValueError("Single-family strategy cannot set secondary_family")
        if self.signal_composition!="single" and (self.secondary_family=="none" or self.secondary_family==self.family): raise ValueError("Composite strategy requires a different secondary family")
        if self.context_filter not in {"none","ema_trend","adx_strength","volume_confirm","vwap_side","low_volatility"}:
            raise ValueError("Unsupported context filter")
        if self.confirmation_filter not in {"none","candle_direction","momentum","rsi_midline"}:
            raise ValueError("Unsupported confirmation filter")

def _cancel_conflicting_signals(long_signal, short_signal):
    """Cancel bars where a strategy emits both directions without an explicit priority."""
    long_signal=long_signal.fillna(False).astype(bool)
    short_signal=short_signal.fillna(False).astype(bool)
    conflict=long_signal & short_signal
    return long_signal & ~conflict, short_signal & ~conflict


def indicators(df, spec):
    # Signal construction only adds/replaces derived columns; OHLCV inputs are
    # read-only. A shallow frame copy avoids duplicating the entire 1m dataset
    # for every genome while keeping derived columns isolated from the caller.
    x=df.copy(deep=False); c=x.close.astype(float); prev=c.shift(1)
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

    elif spec.family=="swing_structure_break":
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1); lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        long_signal=(c>upper)&(c.shift(1)<=upper.shift(1)); short_signal=(c<lower)&(c.shift(1)>=lower.shift(1))
    elif spec.family=="failed_breakout_reversal":
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(2); lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(2)
        long_signal=(x.low.shift(1)<lower)&(c.shift(1)<lower)&(c>lower)&(c>x.open)
        short_signal=(x.high.shift(1)>upper)&(c.shift(1)>upper)&(c<upper)&(c<x.open)
    elif spec.family=="wick_rejection":
        body=(c-x.open).abs().clip(lower=1e-12); lower_wick=np.minimum(x.open,c)-x.low; upper_wick=x.high-np.maximum(x.open,c)
        location=(c-x.low)/(x.high-x.low).replace(0,np.nan)
        long_signal=(lower_wick>=2.0*body)&(location>=0.65)&(c>x.open); short_signal=(upper_wick>=2.0*body)&(location<=0.35)&(c<x.open)
    elif spec.family=="range_expansion_breakout":
        bar_range=x.high-x.low; baseline=bar_range.rolling(spec.fast,min_periods=spec.fast).mean().shift(1)
        long_signal=(bar_range>=1.5*baseline)&(c>x.high.shift(1))&(c.shift(1)<=x.high.shift(1))
        short_signal=(bar_range>=1.5*baseline)&(c<x.low.shift(1))&(c.shift(1)>=x.low.shift(1))
    elif spec.family=="three_bar_momentum":
        upper=x.high.rolling(spec.fast,min_periods=spec.fast).max().shift(1); lower=x.low.rolling(spec.fast,min_periods=spec.fast).min().shift(1)
        rising=(c>c.shift(1))&(c.shift(1)>c.shift(2))&(c.shift(2)>c.shift(3)); falling=(c<c.shift(1))&(c.shift(1)<c.shift(2))&(c.shift(2)<c.shift(3))
        long_signal=rising&(c>upper); short_signal=falling&(c<lower)
    elif spec.family=="volatility_regime_breakout":
        atr_base=x.atr.rolling(spec.slow,min_periods=spec.slow).median().shift(1)
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1); lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        active=x.atr>atr_base
        long_signal=active&(c>upper)&(c.shift(1)<=upper.shift(1)); short_signal=active&(c<lower)&(c.shift(1)>=lower.shift(1))
    elif spec.family=="trend_exhaustion_reversal":
        body=(c-x.open).abs().clip(lower=1e-12); lower_wick=np.minimum(x.open,c)-x.low; upper_wick=x.high-np.maximum(x.open,c)
        long_signal=(c.shift(1)<c.shift(spec.fast+1))&(c.shift(1)<x.open.shift(1))&(c>x.open)&(lower_wick>=body)
        short_signal=(c.shift(1)>c.shift(spec.fast+1))&(c.shift(1)>x.open.shift(1))&(c<x.open)&(upper_wick>=body)
    elif spec.family=="volume_climax_reversal":
        vol_threshold=x.volume.rolling(spec.slow,min_periods=spec.slow).quantile(0.90).shift(1)
        body=(c-x.open).abs().clip(lower=1e-12); lower_wick=np.minimum(x.open,c)-x.low; upper_wick=x.high-np.maximum(x.open,c)
        long_signal=(x.volume>vol_threshold)&(lower_wick>=1.5*body)&(c>x.open); short_signal=(x.volume>vol_threshold)&(upper_wick>=1.5*body)&(c<x.open)
    elif spec.family=="range_compression_breakout":
        bar_range=x.high-x.low; avg_range=bar_range.rolling(spec.fast,min_periods=spec.fast).mean()
        compression_level=avg_range.rolling(spec.slow,min_periods=spec.slow).quantile(0.25).shift(1); compressed=avg_range.shift(1)<compression_level
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1); lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        long_signal=compressed&(c>upper)&(c.shift(1)<=upper.shift(1)); short_signal=compressed&(c<lower)&(c.shift(1)>=lower.shift(1))
    elif spec.family=="support_resistance_rejection":
        support=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1); resistance=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        body=(c-x.open).abs().clip(lower=1e-12); lower_wick=np.minimum(x.open,c)-x.low; upper_wick=x.high-np.maximum(x.open,c)
        long_signal=(x.low<=support+x.atr*0.3)&(c>support)&(lower_wick>=body); short_signal=(x.high>=resistance-x.atr*0.3)&(c<resistance)&(upper_wick>=body)
    elif spec.family=="micro_double_bottom":
        current_low=x.low.rolling(spec.fast,min_periods=spec.fast).min(); prior_low=x.low.shift(spec.fast).rolling(spec.fast,min_periods=spec.fast).min()
        trigger=x.high.shift(1).rolling(spec.fast,min_periods=spec.fast).max()
        long_signal=((current_low-prior_low).abs()<=x.atr*0.35)&(x.low<=current_low+x.atr*0.1)&(c>trigger)&(c>x.open); short_signal=pd.Series(False,index=x.index)
    elif spec.family=="micro_double_top":
        current_high=x.high.rolling(spec.fast,min_periods=spec.fast).max(); prior_high=x.high.shift(spec.fast).rolling(spec.fast,min_periods=spec.fast).max()
        trigger=x.low.shift(1).rolling(spec.fast,min_periods=spec.fast).min()
        short_signal=((current_high-prior_high).abs()<=x.atr*0.35)&(x.high>=current_high-x.atr*0.1)&(c<trigger)&(c<x.open); long_signal=pd.Series(False,index=x.index)
    elif spec.family=="williams_r_reversion":
        highest=x.high.rolling(spec.fast,min_periods=spec.fast).max()
        lowest=x.low.rolling(spec.fast,min_periods=spec.fast).min()
        williams=-100*(highest-c)/(highest-lowest).replace(0,np.nan)
        long_signal=(williams.shift(1)<=-80)&(williams>-80)
        short_signal=(williams.shift(1)>=-20)&(williams<-20)
    elif spec.family=="donchian_midline_trend":
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        midpoint=(upper+lower)/2
        long_signal=(c>midpoint)&(c.shift(1)<=midpoint.shift(1))
        short_signal=(c<midpoint)&(c.shift(1)>=midpoint.shift(1))
    elif spec.family=="dual_thrust_breakout":
        range_up=(x.high-c).rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        range_down=(c-x.low).rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        thrust=np.maximum(range_up,range_down)*0.5
        upper=x.open.shift(1)+thrust
        lower=x.open.shift(1)-thrust
        long_signal=(c>upper)&(c.shift(1)<=upper.shift(1))
        short_signal=(c<lower)&(c.shift(1)>=lower.shift(1))
    elif spec.family=="chaikin_money_flow":
        spread=(x.high-x.low).replace(0,np.nan)
        multiplier=((c-x.low)-(x.high-c))/spread
        flow=multiplier*x.volume.astype(float)
        cmf=flow.rolling(spec.slow,min_periods=spec.slow).sum()/x.volume.astype(float).rolling(spec.slow,min_periods=spec.slow).sum().replace(0,np.nan)
        long_signal=(cmf>0)&(cmf.shift(1)<=0)&(c>x.ema_fast)
        short_signal=(cmf<0)&(cmf.shift(1)>=0)&(c<x.ema_fast)
    elif spec.family=="money_flow_index_reversion":
        typical=(x.high+x.low+c)/3
        raw_flow=typical*x.volume.astype(float)
        positive=raw_flow.where(typical>typical.shift(1),0.0)
        negative=raw_flow.where(typical<typical.shift(1),0.0)
        ratio=positive.rolling(spec.fast,min_periods=spec.fast).sum()/negative.rolling(spec.fast,min_periods=spec.fast).sum().replace(0,np.nan)
        mfi=100-100/(1+ratio)
        long_signal=(mfi.shift(1)<=20)&(mfi>20)
        short_signal=(mfi.shift(1)>=80)&(mfi<80)
    elif spec.family=="atr_trend_pullback":
        trend_up=x.ema_fast>x.ema_slow; trend_down=x.ema_fast<x.ema_slow
        long_signal=trend_up&(x.low<=x.ema_fast+0.1*x.atr)&(c>x.ema_fast)&(c>x.open)
        short_signal=trend_down&(x.high>=x.ema_fast-0.1*x.atr)&(c<x.ema_fast)&(c<x.open)
    elif spec.family=="volume_weighted_breakout":
        upper=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        lower=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        baseline=x.volume.rolling(spec.fast,min_periods=spec.fast).median().shift(1)
        active=x.volume>baseline*1.5
        long_signal=active&(c>upper)&(c.shift(1)<=upper.shift(1))
        short_signal=active&(c<lower)&(c.shift(1)>=lower.shift(1))
    elif spec.family=="range_midpoint_reversion":
        support=x.low.rolling(spec.slow,min_periods=spec.slow).min().shift(1)
        resistance=x.high.rolling(spec.slow,min_periods=spec.slow).max().shift(1)
        width=(resistance-support).replace(0,np.nan)
        long_signal=(x.low<=support+0.15*width)&(c>=support+0.35*width)&(c>x.open)
        short_signal=(x.high>=resistance-0.15*width)&(c<=resistance-0.35*width)&(c<x.open)
    elif spec.family=="roc_acceleration":
        roc=c.pct_change(periods=spec.fast)*100
        long_signal=(roc>0)&(roc>roc.shift(1))&(c>x.ema_slow)
        short_signal=(roc<0)&(roc<roc.shift(1))&(c<x.ema_slow)
    elif spec.family=="atr_normalized_momentum":
        normalized=(c-c.shift(spec.fast))/x.atr.replace(0,np.nan)
        long_signal=(normalized>=1.0)&(c>x.ema_slow)
        short_signal=(normalized<=-1.0)&(c<x.ema_slow)
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
    if spec.signal_composition!="single":
        secondary=indicators(df,replace(spec,family=spec.secondary_family,secondary_family="none",signal_composition="single"))
        sec_long=secondary["long_signal"]; sec_short=secondary["short_signal"]
        if spec.signal_composition=="and":
            long_signal=long_signal&sec_long; short_signal=short_signal&sec_short
        elif spec.signal_composition=="or":
            long_signal=long_signal|sec_long; short_signal=short_signal|sec_short
        elif spec.signal_composition=="xor":
            long_signal=long_signal^sec_long; short_signal=short_signal^sec_short
    # Composite OR/XOR rules can otherwise emit LONG and SHORT on the same
    # candle. The backtester previously resolved that silently as LONG because
    # it checks long_signal first. With no explicit priority in the strategy
    # contract, a contradictory bar must be a no-trade bar.
    long_signal,short_signal=_cancel_conflicting_signals(long_signal,short_signal)
    x["long_signal"]=long_signal
    x["short_signal"]=short_signal
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

def _resolve_bar_exit(side: str, bar_open: float, bar_high: float, bar_low: float, stop: float, target: float):
    """Resolve OHLC exits conservatively, honoring gaps at the bar open first."""
    if side == "LONG":
        if bar_open <= stop:
            return "SL", bar_open
        if bar_open >= target:
            return "TP", target  # conservative limit fill; do not assume price improvement
        sl = bar_low <= stop
        tp = bar_high >= target
    else:
        if bar_open >= stop:
            return "SL", bar_open
        if bar_open <= target:
            return "TP", target  # conservative limit fill; do not assume price improvement
        sl = bar_high >= stop
        tp = bar_low <= target
    if sl and tp:
        raise DataIntegrityError("Unresolved same-candle SL/TP ambiguity; finer data required")
    if sl:
        return "SL", stop
    if tp:
        return "TP", target
    return None


def _resolve_execution_interval(side: str, execution_frame: pd.DataFrame, start_ms: int, interval_ms: int, stop: float, target: float):
    """Resolve a base candle using every 1-second USD-M execution candle in time order.

    Missing seconds and unresolved same-second SL/TP hits fail closed. This deliberately
    does not infer the path from the enclosing minute OHLC.
    """
    if interval_ms < 1000 or interval_ms % 1000:
        raise DataIntegrityError("Execution resolution requires a base interval divisible by one second")
    times = execution_frame["open_time"].to_numpy(dtype=np.int64)
    first = int(np.searchsorted(times, start_ms, side="left"))
    end_ms = start_ms + interval_ms
    last = int(np.searchsorted(times, end_ms, side="left"))
    expected = interval_ms // 1000
    bars = execution_frame.iloc[first:last]
    if len(bars) != expected or int(bars.open_time.iloc[0]) != start_ms or int(bars.open_time.iloc[-1]) != end_ms - 1000:
        raise DataIntegrityError(f"Missing/incomplete 1-second execution data for interval starting {start_ms}; expected {expected} consecutive candles")
    for bar in bars.itertuples(index=False):
        resolution = _resolve_bar_exit(side, float(bar.open), float(bar.high), float(bar.low), stop, target)
        if resolution is not None:
            reason, price = resolution
            return reason, price, int(bar.open_time)
    return None


# Cache only the compact immutable indicator arrays, and bound their estimated
# memory footprint independently of dataset length. Each entry stores one float64
# ATR array plus two boolean signal arrays (approximately 10 bytes per candle).
# This matters on NetRun's 256 MiB instance: a fixed 128-entry cache on a full
# 263k-candle dataset could otherwise retain roughly 336 MiB before other state.
INDICATOR_CACHE_BUDGET_BYTES = 16 * 1024 * 1024
INDICATOR_CACHE_MAX_ENTRIES = 128
INDICATOR_BYTES_PER_CANDLE = 8 + 1 + 1


def _indicator_cache_limit_for_rows(row_count: int) -> int:
    rows = max(1, int(row_count))
    by_memory = INDICATOR_CACHE_BUDGET_BYTES // (rows * INDICATOR_BYTES_PER_CANDLE)
    return max(1, min(INDICATOR_CACHE_MAX_ENTRIES, int(by_memory)))


class _PreparedBacktestData:
    """One-time validated immutable market-data view shared by candidate backtests."""
    __slots__ = ("source_frame", "frame", "base_interval", "report", "data_hash",
                 "indicator_cache", "indicator_cache_hits", "indicator_cache_misses",
                 "indicator_cache_limit")
    def __init__(self, frame):
        df=frame.copy().reset_index(drop=True)
        required=["open_time","open","high","low","close","volume"]
        try:
            for column in required:
                df[column]=pd.to_numeric(df[column],errors="raise")
        except (KeyError, TypeError, ValueError) as exc:
            raise DataIntegrityError(f"OHLCV columns must be numeric before backtest: {exc}") from exc
        if df["open_time"].isna().any() or not np.isfinite(df["open_time"].to_numpy(dtype=float)).all():
            raise DataIntegrityError("open_time contains invalid timestamps")
        if not np.equal(df["open_time"].to_numpy(dtype=float),np.floor(df["open_time"].to_numpy(dtype=float))).all():
            raise DataIntegrityError("open_time timestamps must be integer milliseconds")
        df["open_time"]=df["open_time"].astype("int64")
        base_interval=None
        if len(df)>1:
            diffs=df.open_time.diff().dropna()
            base_interval=int(diffs.mode().iloc[0]) if len(diffs) else None
            report=validate_ohlcv(df,base_interval,strict_gaps=True)
        else:
            report=validate_ohlcv(df,None)
        self.source_frame=frame
        self.frame=df
        self.base_interval=base_interval
        self.report=report
        self.data_hash=sha256_bytes(df.to_csv(index=False,float_format="%.12g").encode())
        # Stop distance and trade direction do not alter signal/ATR arrays. A
        # bounded LRU reuses exact indicator results across those genome variants.
        self.indicator_cache=OrderedDict()
        self.indicator_cache_hits=0
        self.indicator_cache_misses=0
        self.indicator_cache_limit=_indicator_cache_limit_for_rows(len(df))

    def indicator_arrays(self, spec):
        key=(spec.fast,spec.slow,spec.atr_period,spec.family,spec.context_filter,
             spec.confirmation_filter,spec.secondary_family,spec.signal_composition)
        cached=self.indicator_cache.get(key)
        if cached is not None:
            self.indicator_cache_hits+=1
            self.indicator_cache.move_to_end(key)
            return cached
        self.indicator_cache_misses+=1
        x=indicators(self.frame,spec)
        arrays=tuple(np.asarray(x[column].to_numpy(copy=True),dtype=dtype)
                     for column,dtype in (("atr",np.float64),("long_signal",np.bool_),("short_signal",np.bool_)))
        for array in arrays:
            array.setflags(write=False)
        self.indicator_cache[key]=arrays
        if len(self.indicator_cache)>self.indicator_cache_limit:
            self.indicator_cache.popitem(last=False)
        return arrays


def prepare_backtest_data(frame):
    """Fully validate and fingerprint a market frame for repeated exact-data backtests."""
    return _PreparedBacktestData(frame)


def _first_exit_candidate(side, opens, highs, lows, start, stop, target):
    """Return the first touched bar using bounded-memory vectorized chunks.

    A full-tail boolean mask for every trade creates large temporary arrays on
    multi-month 1m data. Scan in bounded chunks instead; this preserves the
    exact first-hit order and leaves gap/SL-TP ambiguity resolution to the
    scalar reference resolver.
    """
    n=len(opens)
    chunk_size=4096
    for chunk_start in range(int(start),n,chunk_size):
        chunk_end=min(chunk_start+chunk_size,n)
        op=opens[chunk_start:chunk_end]
        hi=highs[chunk_start:chunk_end]
        lo=lows[chunk_start:chunk_end]
        if side=="LONG":
            touched=(op<=stop)|(op>=target)|(lo<=stop)|(hi>=target)
        else:
            touched=(op>=stop)|(op<=target)|(hi>=stop)|(lo<=target)
        hits=np.flatnonzero(touched)
        if hits.size:
            return chunk_start+int(hits[0])
    return None


def backtest(frame, spec=None, contract=None, execution_frame=None, _prepared_context=None):
    spec=StrategySpec(**spec) if isinstance(spec,dict) else (spec or StrategySpec()); spec.validate()
    cfg=normalize_contract(contract)
    prepared=_prepared_context if _prepared_context is not None else prepare_backtest_data(frame)
    if prepared.source_frame is not frame:
        raise ValueError("Prepared market data can only be reused with its exact source DataFrame")
    df=prepared.frame
    base_interval=prepared.base_interval
    report=prepared.report
    if len(df)<max(spec.slow,spec.atr_period)+3: raise DataIntegrityError("Not enough candles for warmup")
    data_hash=prepared.data_hash
    execution_hash=None
    exec_df=None
    execution_frames_used=[]
    if execution_frame is not None:
        exec_df=execution_frame.copy().reset_index(drop=True)
        validate_ohlcv(exec_df,1000,strict_gaps=False)
        execution_hash=sha256_bytes(exec_df.to_csv(index=False,float_format="%.12g").encode())
        if len(df)>1:
            base_interval=int(df.open_time.diff().dropna().mode().iloc[0])
            if base_interval % 1000:
                raise DataIntegrityError("Base candle interval cannot be resolved with 1-second candles")
    atr_values,long_signal_values,short_signal_values=prepared.indicator_arrays(spec)
    balance=float(cfg["initial_balance"]); ledger=[]; streak=0; max_streak=0; open_position=None
    # OHLCV arrays come from the fully validated shared frame. Signal/ATR arrays
    # are cached by every signal-affecting parameter, but not stop/direction.
    open_times=df["open_time"].to_numpy(dtype=np.int64)
    open_values=df["open"].to_numpy(dtype=float)
    high_values=df["high"].to_numpy(dtype=float)
    low_values=df["low"].to_numpy(dtype=float)
    close_values=df["close"].to_numpy(dtype=float)
    signal_mask=long_signal_values | short_signal_values
    signal_positions=np.flatnonzero(signal_mask)
    warmup=max(spec.slow,spec.atr_period)
    signal_positions=signal_positions[(signal_positions>=warmup)&(signal_positions<len(df)-1)]
    signal_cursor=0
    while signal_cursor < len(signal_positions):
        i=int(signal_positions[signal_cursor]); signal_cursor+=1
        atr=float(atr_values[i])
        if not np.isfinite(atr) or atr<=0: continue
        side="LONG" if bool(long_signal_values[i]) else ("SHORT" if bool(short_signal_values[i]) else None)
        if side is None or (spec.direction=="long" and side=="SHORT") or (spec.direction=="short" and side=="LONG"): continue
        entry_i=i+1; entry=float(open_values[entry_i]); dist=float(atr*spec.stop_atr)
        stop=entry-dist if side=="LONG" else entry+dist
        risk_budget=_risk_budget(cfg,balance if cfg["compound"] or cfg["risk_mode"]=="percent_equity" else cfg["initial_balance"])
        if risk_budget<=0: i+=1; continue
        # A trade risking the entire remaining equity or more is not executable.
        # Stop conservatively before fixed-risk losses can drive the account negative.
        if balance<=risk_budget+1e-9: break
        try: qty,target,entry_rate,exit_rate=_size_and_target(entry,stop,side,risk_budget,cfg)
        except ValueError: i+=1; continue
        if qty<=0: i+=1; continue
        exit_i=exit_price=exit_time=None; reason=None; resolution_status="resolved_base_ohlc"
        if exec_df is not None:
            for j in range(entry_i,len(df)):
                try:
                    resolution=_resolve_execution_interval(side,exec_df,int(open_times[j]),base_interval,stop,target)
                except DataIntegrityError as exc:
                    raise DataIntegrityError(f"Execution preflight failed at {int(open_times[j])}: {exc}") from exc
                if resolution is not None:
                    reason,exit_price,exit_time=resolution; exit_i=j
                    resolution_status="resolved_1s_ohlc"
                    break
        else:
            # Vectorized event detection; scalar resolver still enforces the
            # exact open-gap precedence and same-candle ambiguity policy.
            j=_first_exit_candidate(side,open_values,high_values,low_values,entry_i,stop,target)
            if j is not None:
                auto_resolved=False
                try:
                    resolution=_resolve_bar_exit(side,float(open_values[j]),float(high_values[j]),float(low_values[j]),stop,target)
                except DataIntegrityError as exc:
                    if "Unresolved same-candle SL/TP ambiguity" not in str(exc):
                        raise DataIntegrityError(f"Execution preflight failed at {int(open_times[j])}: {exc}") from exc
                    try:
                        seconds=download_binance_execution_window(cfg.get("symbol","ETHUSDT"),int(open_times[j]),base_interval)
                    except Exception as fetch_exc:
                        raise DataIntegrityError(f"Unresolved same-candle SL/TP ambiguity at {int(open_times[j])}; automatic 1-second resolution failed: {fetch_exc}") from fetch_exc
                    execution_frames_used.append(seconds)
                    resolution=_resolve_execution_interval(side,seconds,int(open_times[j]),base_interval,stop,target)
                    if resolution is None:
                        raise DataIntegrityError(f"1-second data contradicts ambiguous base candle at {int(open_times[j])}")
                    reason,exit_price,exit_time=resolution; exit_i=j
                    resolution_status="resolved_1s_ohlc_auto"
                    auto_resolved=True
                if resolution is not None and not auto_resolved:
                    reason,exit_price=resolution; exit_i=j; exit_time=int(open_times[j])
                    resolution_status="resolved_base_ohlc"
        if exit_i is None:
            last_price=float(close_values[-1])
            unrealized_gross=(last_price-entry)*qty*(1 if side=="LONG" else -1)
            open_position={"signal_time":int(open_times[i]),"entry_time":int(open_times[entry_i]),"direction":side,
                           "entry_price":entry,"stop_loss":stop,"take_profit":target,"quantity":qty,"risk_budget":risk_budget,
                           "last_price":last_price,"unrealized_gross_pnl":unrealized_gross,
                           "unrealized_net_pnl":unrealized_gross-entry*qty*entry_rate-last_price*qty*exit_rate}
            break
        gross=(exit_price-entry)*qty*(1 if side=="LONG" else -1)
        entry_cost=entry*qty*entry_rate; exit_cost=exit_price*qty*exit_rate
        net=gross-entry_cost-exit_cost; balance+=net
        if reason=="TP": streak=0
        else: streak+=1; max_streak=max(max_streak,streak)
        ledger.append({"trade_id":len(ledger)+1,"signal_time":int(open_times[i]),"entry_time":int(open_times[entry_i]),"exit_time":int(exit_time),"direction":side,"signal_price":float(close_values[i]),"entry_price":entry,"stop_loss":stop,"take_profit":target,"exit_price":float(exit_price),"exit_reason":reason,"quantity":qty,"risk_budget":risk_budget,"gross_pnl":gross,"entry_cost":entry_cost,"exit_cost":exit_cost,"fees_slippage":entry_cost+exit_cost,"net_pnl":net,"balance_after":balance,"loss_streak_after":streak,"resolution_status":resolution_status,"execution_ambiguity":resolution_status=="unresolved_as_loss"})
        signal_cursor=int(np.searchsorted(signal_positions,exit_i+1,side="left"))
    wins=[t for t in ledger if t["exit_reason"]=="TP" and t["net_pnl"]>0]
    gp=sum(max(0,t["net_pnl"]) for t in ledger); gl=abs(sum(min(0,t["net_pnl"]) for t in ledger))
    span_days=max((int(df.open_time.iloc[-1])-int(df.open_time.iloc[0]))/86400000,1/1440)
    wr=100*len(wins)/len(ledger) if ledger else 0.0; pf=gp/gl if gl else (math.inf if gp else 0.0); tp_day=len(wins)/span_days
    evidence_metrics={"span_days":span_days,"trades":len(ledger),"win_rate_pct":wr,"avg_net_tp_per_day":tp_day}
    passes=(not cfg["use_min_win_rate"] or wr>cfg["min_win_rate_pct"]) and (not cfg["use_min_profit_factor"] or pf>cfg["min_profit_factor"]) and (not cfg["use_min_tp_per_day"] or tp_day>=cfg["min_tp_per_day"]) and (not cfg["use_max_consecutive_losses"] or max_streak<=cfg["max_consecutive_losses"]) and open_position is None and _goal_activity_evidence(evidence_metrics,cfg)
    if execution_frames_used:
        combined_execution=pd.concat(execution_frames_used,ignore_index=True).drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        execution_hash=sha256_bytes(combined_execution.to_csv(index=False,float_format="%.12g").encode())
    execution_resolution="1s_ohlc" if exec_df is not None else ("1s_ohlc_auto" if execution_frames_used else "base_ohlc_only")
    metrics={"trades":len(ledger),"wins":len(wins),"losses":sum(t["net_pnl"]<=0 for t in ledger),"win_rate_pct":wr,"profit_factor":pf if math.isfinite(pf) else None,"avg_net_tp_per_day":tp_day,"max_consecutive_losses":max_streak,"net_pnl":sum(t["net_pnl"] for t in ledger),"final_balance":balance,"total_costs":sum(t["fees_slippage"] for t in ledger),"data_hash_sha256":data_hash,"execution_data_hash_sha256":execution_hash,"execution_resolution":execution_resolution,"unresolved_as_loss_count":sum(t.get("execution_ambiguity",False) for t in ledger),"open_positions_at_end":1 if open_position else 0,"open_position_unrealized_net_pnl":open_position["unrealized_net_pnl"] if open_position else 0.0,"rows":len(df),"span_days":span_days,"passes_all_targets":bool(passes)}
    run_hash=sha256_bytes(json.dumps({"spec":asdict(spec),"contract":cfg,"data_hash":data_hash,"execution_data_hash":execution_hash,"execution_resolution":execution_resolution},sort_keys=True).encode())
    return {"strategy":asdict(spec),"contract":cfg,"metrics":metrics,"ledger":ledger,"open_position":open_position,"data_report":report,"run_hash_sha256":run_hash}

def _capability_evidence(metrics: dict[str, Any]) -> bool:
    """Search-steering gate: avoid optimizing statistically tiny, low-activity samples."""
    pf=math.inf if metrics.get("profit_factor") is None else float(metrics.get("profit_factor",0.0))
    return (float(metrics.get("avg_net_tp_per_day",0.0))>=0.5
            and int(metrics.get("trades",0))>=50
            and (float(metrics.get("win_rate_pct",0.0))>=25.0 or pf>=0.5))


def _goal_activity_evidence(metrics: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Reject samples too small to substantiate the requested full-TP/day rate.

    This is a ranking/evidence gate, not an extra acceptance threshold.
    """
    span_days=max(float(metrics.get("span_days", 0.0) or 0.0), 0.0)
    trades=int(metrics.get("trades", 0) or 0)
    wr=max(0.0, min(1.0, float(metrics.get("win_rate_pct", 0.0) or 0.0)/100.0))
    target_tp=float(cfg.get("min_tp_per_day", 3.0)) if cfg.get("use_min_tp_per_day", True) else 3.0
    if span_days <= 0.0 or trades < 100:
        return False
    # Disabled TP/day must not silently steer or block selection.
    if not cfg.get("use_min_tp_per_day", True):
        return True
    required=span_days*target_tp/max(wr, 0.01)
    return trades >= math.ceil(required)


def _candidate_distance(metrics: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, list[str], dict[str, float]]:
    """Equal-weight normalized distance to each active target."""
    failed: list[str] = []
    distances: dict[str, float] = {}
    if cfg["use_min_win_rate"]:
        wr=float(metrics["win_rate_pct"])
        gap=max(0.0,cfg["min_win_rate_pct"]+0.0001-wr)
        distances["win_rate_pct"]=gap/max(cfg["min_win_rate_pct"],1.0)
        if wr<=cfg["min_win_rate_pct"]: failed.append("win_rate_pct")
    if cfg["use_min_profit_factor"]:
        pf=math.inf if metrics["profit_factor"] is None else float(metrics["profit_factor"])
        gap=max(0.0,cfg["min_profit_factor"]+0.0001-pf)
        distances["profit_factor"]=gap/max(abs(cfg["min_profit_factor"]),1.0)
        if pf<=cfg["min_profit_factor"]: failed.append("profit_factor")
    if cfg["use_min_tp_per_day"]:
        tp_day=float(metrics["avg_net_tp_per_day"])
        gap=max(0.0,cfg["min_tp_per_day"]-tp_day)
        distances["avg_net_tp_per_day"]=gap/max(cfg["min_tp_per_day"],1.0)
        if tp_day<cfg["min_tp_per_day"]: failed.append("avg_net_tp_per_day")
    if cfg["use_max_consecutive_losses"]:
        streak=int(metrics["max_consecutive_losses"])
        limit=max(int(cfg["max_consecutive_losses"]),1)
        # Keep the hard pass/fail boundary, but also prefer lower streak risk
        # among candidates already under the limit.
        distances["max_consecutive_losses"]=max(0.0,streak)/limit
        if streak>int(cfg["max_consecutive_losses"]): failed.append("max_consecutive_losses")
    # Evidence sufficiency is an eligibility gate, not a fifth weighted target.
    if not _goal_activity_evidence(metrics,cfg):
        failed.append("insufficient_activity_evidence")
    if int(metrics.get("open_positions_at_end",0))>0:
        failed.append("open_positions_at_end")
    # Equal arithmetic mean: every enabled target contributes exactly 1/N.
    return (sum(distances.values()) / len(distances) if distances else 0.0), failed, distances


def search_candidates(frame, max_candidates=500, seed=42, contract=None, algorithm="surrogate_genetic", search_space=None, progress_callback=None, excluded_genome_hashes=None, prior_candidates=None, prepared_context=None):
    """Deterministic bounded evolutionary search on TRAIN only.

    Each invocation is bounded to 10,000 genomes. The background API orchestrates
    multiple invocations with globally persisted genome fingerprints and atomic
    batch checkpoints for requests up to 500,000 candidates.
    """
    if not 1<=max_candidates<=10000: raise ValueError("max_candidates must be 1..10000 until resumable batch execution is enabled")
    if algorithm not in {"hybrid_genetic","surrogate_genetic","stratified_random"}:
        raise ValueError("Unsupported search algorithm")
    cfg=normalize_contract(contract)
    rng=np.random.default_rng(seed)
    excluded_genome_hashes=set(excluded_genome_hashes or ())
    def genome_hash(key):
        return hashlib.sha256(json.dumps(list(key),separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    all_families=("ema_crossover","channel_breakout","rsi_reversion","volume_breakout","ema_pullback","macd_crossover","bollinger_reversion","inside_bar_breakout","engulfing_reversal","momentum_breakout","stochastic_reversion","roc_momentum","adx_trend","keltner_breakout","atr_squeeze_breakout","obv_trend","vwap_reclaim","pivot_reversal","swing_structure_break","failed_breakout_reversal","wick_rejection","range_expansion_breakout","three_bar_momentum","volatility_regime_breakout","trend_exhaustion_reversal","volume_climax_reversal","range_compression_breakout","support_resistance_rejection","micro_double_bottom","micro_double_top","williams_r_reversion","donchian_midline_trend","dual_thrust_breakout","chaikin_money_flow","money_flow_index_reversion","atr_trend_pullback","volume_weighted_breakout","range_midpoint_reversion","roc_acceleration","atr_normalized_momentum")
    all_context_filters=("none","ema_trend","adx_strength","volume_confirm","vwap_side","low_volatility")
    all_confirmation_filters=("none","candle_direction","momentum","rsi_midline")
    space=dict(search_space or {})
    families=tuple(dict.fromkeys(space["families"])) if "families" in space else all_families
    context_filters=tuple(dict.fromkeys(space["context_filters"])) if "context_filters" in space else all_context_filters
    confirmation_filters=tuple(dict.fromkeys(space["confirmation_filters"])) if "confirmation_filters" in space else all_confirmation_filters
    if not families or any(f not in all_families for f in families): raise ValueError("Search family selection is empty or contains unsupported families")
    if not context_filters or any(f not in all_context_filters for f in context_filters): raise ValueError("Context filter selection is empty or unsupported")
    if not confirmation_filters or any(f not in all_confirmation_filters for f in confirmation_filters): raise ValueError("Confirmation filter selection is empty or unsupported")
    if prepared_context is None:
        required={"open_time","open","high","low","close","volume"}
        if not required.issubset(frame.columns) and getattr(backtest,"__module__",None)!=__name__:
            # Test doubles may intentionally supply synthetic metrics without market data.
            # Production backtests still fail closed on missing/invalid OHLCV columns.
            prepared_context=None
        else:
            prepared_context=prepare_backtest_data(frame)
    if prepared_context is not None and prepared_context.source_frame is not frame:
        raise ValueError("Prepared market data can only be reused with its exact source DataFrame")
    names={"ema_crossover":"EMA crossover","channel_breakout":"Price channel breakout","rsi_reversion":"RSI reversion","volume_breakout":"Volume-confirmed breakout","ema_pullback":"EMA trend pullback","macd_crossover":"MACD crossover","bollinger_reversion":"Bollinger mean reversion","inside_bar_breakout":"Inside-bar breakout","engulfing_reversal":"Engulfing candle reversal","momentum_breakout":"Momentum-confirmed breakout","stochastic_reversion":"Stochastic reversion","roc_momentum":"Rate-of-change momentum","adx_trend":"ADX directional trend","keltner_breakout":"Keltner channel breakout","atr_squeeze_breakout":"Volatility squeeze breakout","obv_trend":"OBV volume trend","vwap_reclaim":"VWAP reclaim","pivot_reversal":"Support/resistance pivot reversal","swing_structure_break":"شکست ساختار سوئینگ","failed_breakout_reversal":"برگشت شکست ناموفق","wick_rejection":"رد قیمت با سایه کندل","range_expansion_breakout":"شکست با انبساط دامنه","three_bar_momentum":"مومنتوم سه‌کندلی","volatility_regime_breakout":"شکست در رژیم نوسان","trend_exhaustion_reversal":"برگشت فرسودگی روند","volume_climax_reversal":"برگشت اوج حجم","range_compression_breakout":"شکست پس از فشردگی دامنه","support_resistance_rejection":"رد قیمت حمایت/مقاومت","micro_double_bottom":"کف دوقلوی کوتاه‌مدت","micro_double_top":"سقف دوقلوی کوتاه‌مدت","williams_r_reversion":"Williams %R reversion","donchian_midline_trend":"Donchian midline trend","dual_thrust_breakout":"Dual Thrust breakout","chaikin_money_flow":"Chaikin money-flow trend","money_flow_index_reversion":"Money Flow Index reversion","atr_trend_pullback":"ATR-aware trend pullback","volume_weighted_breakout":"Volume-weighted range breakout","range_midpoint_reversion":"Range midpoint reversion","roc_acceleration":"Rate-of-change acceleration","atr_normalized_momentum":"ATR-normalized momentum"}
    fast_values=np.array([2,3,4,5,6,8,9,10,12,14,16,18,21,24,28,32,36,40,48,55,64,72,89],dtype=int)
    slow_values=np.array([5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,48,55,64,72,89,100,120,144,180,200,240,300,400,500],dtype=int)
    periods=np.array([3,4,5,6,7,8,9,10,12,14,16,18,21,24,28,30,40,50],dtype=int)
    stops=np.array([0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0,3.5,4.0,5.0,6.0],dtype=float)
    directions=("both","long","short")
    evaluated:dict[tuple,dict[str,Any]]={}
    errors=0
    family_counts={f:0 for f in families}
    duplicate_genomes=0

    def choose_family(exclude=None):
        """Sample under-explored families more often without banning exploitation."""
        options=[family for family in families if family!=exclude]
        if not options:
            options=list(families)
        weights=np.asarray([1.0/math.sqrt(1.0+family_counts.get(family,0)) for family in options],dtype=float)
        weights=weights/weights.sum()
        return str(rng.choice(options,p=weights))
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

    def _filter_probabilities(options, kind):
        # High-frequency hypotheses should not be suppressed by stacking filters.
        # Preserve every filter option while sampling the unfiltered baseline more often.
        weights=[]
        for option in options:
            if option=="none":
                weights.append(0.68 if kind=="context" else 0.62)
            else:
                weights.append((0.32 if kind=="context" else 0.38)/max(1,len(options)-1))
        weights=np.asarray(weights,dtype=float)
        return weights/weights.sum()

    def make_spec(family=None, genes=None):
        if genes is not None:
            family,fast,slow,period,stop,direction,context_filter,confirmation_filter,secondary_family,signal_composition=genes
        else:
            family=str(family or choose_family())
            fast=int(rng.choice(fast_values))
            slow=int(rng.choice(slow_values))
            if slow<=fast: slow=int(min(500,fast+max(2,int(rng.choice([4,8,12,16,24,32,48])))))
            period=int(rng.choice(periods)); stop=float(rng.choice(stops)); direction=str(rng.choice(directions))
            context_filter=str(rng.choice(context_filters,p=_filter_probabilities(context_filters, "context"))); confirmation_filter=str(rng.choice(confirmation_filters,p=_filter_probabilities(confirmation_filters, "confirmation")))
            if len(families)>1 and rng.random()<0.65:
                signal_composition=str(rng.choice(["and","or","xor"],p=[0.30,0.60,0.10]))
                secondary_family=str(rng.choice([f for f in families if f!=family]))
            else:
                signal_composition="single"; secondary_family="none"
        fast=int(fast); slow=int(slow); period=int(period); stop=float(stop); direction=str(direction)
        if family=="channel_breakout": fast=min(fast, max(2,slow-2))
        elif family=="rsi_reversion": slow=max(fast+2, min(500,fast*2))
        elif family in {"inside_bar_breakout","engulfing_reversal"}: fast=9; slow=max(20,slow)
        if slow<=fast: fast=max(2,slow-2)
        if signal_composition=="single": secondary_family="none"
        elif secondary_family in {"none",family}:
            options=[f for f in families if f!=family]
            if not options: signal_composition="single"; secondary_family="none"
            else: secondary_family=str(rng.choice(options))
        label=f"{names[family]}"+(f" {signal_composition.upper()} {names[secondary_family]}" if signal_composition!="single" else "")
        return StrategySpec(f"{label} {fast}/{slow} ATR {period}x{stop:g} {direction} [ctx={context_filter}; confirm={confirmation_filter}]",fast,slow,period,stop,direction,seed,family,context_filter,confirmation_filter,secondary_family,signal_composition)

    def genome(spec):
        return (spec.family,spec.fast,spec.slow,spec.atr_period,round(spec.stop_atr,6),spec.direction,spec.context_filter,spec.confirmation_filter,spec.secondary_family,spec.signal_composition)

    def fitness(row):
        # Evidence/pass status are eligibility gates, separate from the score.
        # The four selectable performance criteria are ranked only by their
        # equal-weight normalized distance, with no metric-specific tie-break.
        m=row["metrics"]
        return (not m["passes_all_targets"], not _goal_activity_evidence(m,cfg),
                row["distance"], -int(m["trades"]))

    def feature_vector(spec):
        vals=[1.0 if spec.family==f else 0.0 for f in all_families]
        vals.extend([spec.fast/500.0,spec.slow/500.0,spec.atr_period/200.0,spec.stop_atr/10.0])
        vals.extend([1.0 if spec.direction==d else 0.0 for d in directions])
        vals.extend([1.0 if spec.context_filter==f else 0.0 for f in all_context_filters])
        vals.extend([1.0 if spec.confirmation_filter==f else 0.0 for f in all_confirmation_filters])
        vals.extend([1.0 if spec.secondary_family==f else 0.0 for f in all_families]+[1.0 if spec.secondary_family=="none" else 0.0])
        vals.extend([1.0 if spec.signal_composition==f else 0.0 for f in ("single","and","or","xor")])
        return vals

    def surrogate_target(row):
        m=row["metrics"]
        # Learn the same equal-weight score as final ranking. Evidence and
        # pass status are separate penalties, not hidden metric-specific weights.
        evidence_penalty=0.0 if _goal_activity_evidence(m,cfg) else 2.0
        pass_penalty=0.0 if m["passes_all_targets"] else 1.0
        return float(row["distance"])+evidence_penalty+pass_penalty

    last_model_fit_at=-1
    surrogate_fit_count=0

    def evaluate(spec):
        nonlocal errors,duplicate_genomes
        key=genome(spec)
        if key in evaluated:
            duplicate_genomes+=1
            return evaluated[key]
        if genome_hash(key) in excluded_genome_hashes:
            duplicate_genomes+=1
            return None
        try:
            r=(backtest(frame,spec,cfg,_prepared_context=prepared_context)
               if prepared_context is not None else backtest(frame,spec,cfg)); m=r["metrics"]
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
    # Seed later batches with audited candidates from earlier batches. Reuse
    # their metrics only when dataset and contract fingerprints match exactly.
    prior_rows=[]
    expected_data_hash=(prepared_context.data_hash if prepared_context is not None
                        else sha256_bytes(frame.reset_index(drop=True).to_csv(index=False,float_format="%.12g").encode()))
    seen_prior=set()
    for candidate in list(prior_candidates or [])[:192]:
        try:
            if not isinstance(candidate,dict) or not isinstance(candidate.get("strategy"),dict): continue
            if candidate.get("contract",cfg)!=cfg: continue
            metrics=candidate.get("metrics") or {}
            if metrics.get("data_hash_sha256")!=expected_data_hash: continue
            prior_spec=StrategySpec(**candidate["strategy"]); prior_spec.validate()
            key=genome(prior_spec); key_hash=genome_hash(key)
            if key_hash in seen_prior or key_hash not in excluded_genome_hashes: continue
            dist,miss,distances=_candidate_distance(metrics,cfg)
            row={"strategy":asdict(prior_spec),"contract":cfg,"metrics":metrics,"distance":dist,
                 "failed_conditions":miss,"criterion_distances":distances,
                 "run_hash_sha256":candidate.get("run_hash_sha256"),"_genome":key}
            population.append((prior_spec,row)); prior_rows.append(row); seen_prior.add(key_hash)
        except (TypeError,ValueError,KeyError):
            continue
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
                family=str(rng.choice([a.family,b.family])) if rng.random()<0.45 else choose_family()
                fast=int(a.fast if rng.random()<0.5 else b.fast)
                slow=int(a.slow if rng.random()<0.5 else b.slow)
                period=int(a.atr_period if rng.random()<0.5 else b.atr_period)
                stop=float(a.stop_atr if rng.random()<0.5 else b.stop_atr)
                direction=str(a.direction if rng.random()<0.5 else b.direction)
                context_filter=str(a.context_filter if rng.random()<0.5 else b.context_filter)
                confirmation_filter=str(a.confirmation_filter if rng.random()<0.5 else b.confirmation_filter)
                secondary_family=str(a.secondary_family if rng.random()<0.5 else b.secondary_family)
                signal_composition=str(a.signal_composition if rng.random()<0.5 else b.signal_composition)
                if rng.random()<0.20: signal_composition=str(rng.choice(["single","and","or","xor"]))
                if rng.random()<0.25: secondary_family=str(rng.choice(["none"]+[f for f in families if f!=family]))
                if rng.random()<0.35: fast=int(rng.choice(fast_values))
                if rng.random()<0.35: slow=int(rng.choice(slow_values))
                if rng.random()<0.25: period=int(rng.choice(periods))
                if rng.random()<0.25: stop=float(rng.choice(stops))
                if rng.random()<0.18: direction=str(rng.choice(directions))
                # Stronger family mutation counters early genetic convergence.
                if rng.random()<0.28: family=choose_family()
                if rng.random()<0.25: context_filter=str(rng.choice(context_filters,p=_filter_probabilities(context_filters, "context")))
                if rng.random()<0.20: confirmation_filter=str(rng.choice(confirmation_filters,p=_filter_probabilities(confirmation_filters, "confirmation")))
                spec=make_spec(genes=(family,fast,slow,period,stop,direction,context_filter,confirmation_filter,secondary_family,signal_composition))
        if algorithm=="surrogate_genetic" and len(evaluated)>=32 and (last_model_fit_at<0 or len(evaluated)-last_model_fit_at>=256):
            try:
                from sklearn.ensemble import ExtraTreesRegressor
                training_rows=(prior_rows+list(evaluated.values()))[-2000:]
                if len(training_rows)>2000: training_rows=training_rows[-2000:]
                X=[]; y=[]
                for training_row in training_rows:
                    training_spec=StrategySpec(**training_row["strategy"])
                    X.append(feature_vector(training_spec)); y.append(surrogate_target(training_row))
                model=ExtraTreesRegressor(n_estimators=64,min_samples_leaf=2,max_features=0.9,
                                          random_state=int(seed+len(evaluated)),n_jobs=1)
                model.fit(np.asarray(X,dtype=float),np.asarray(y,dtype=float))
                # Active-learning proposal pool: exploit low predicted loss, explore
                # model disagreement, and prefer structurally novel genomes.
                proposals=[spec]
                for _ in range(63):
                    proposals.append(make_spec())
                unseen=[candidate for candidate in proposals if genome(candidate) not in evaluated and genome_hash(genome(candidate)) not in excluded_genome_hashes]
                if unseen:
                    candidate_X=np.asarray([feature_vector(candidate) for candidate in unseen],dtype=float)
                    tree_predictions=np.vstack([tree.predict(candidate_X) for tree in model.estimators_])
                    predicted_mean=tree_predictions.mean(axis=0)
                    predicted_uncertainty=tree_predictions.std(axis=0)
                    known_X=np.asarray(X,dtype=float)
                    novelty=np.asarray([
                        np.sqrt(np.mean(np.square(known_X-vector),axis=1)).min()
                        for vector in candidate_X
                    ],dtype=float)
                    uncertainty_scale=max(float(predicted_uncertainty.max()),1e-12)
                    novelty_scale=max(float(novelty.max()),1e-12)
                    acquisition=(predicted_mean
                                -0.50*(predicted_uncertainty/uncertainty_scale)
                                -0.15*(novelty/novelty_scale))
                    spec=unseen[int(np.argmin(acquisition))]
                last_model_fit_at=len(evaluated); surrogate_fit_count+=1
            except ImportError as exc:
                raise ValueError("Surrogate ML search requires scikit-learn; install the declared requirements and restart. Search stopped without fallback.") from exc
        if genome(spec) in evaluated or genome_hash(genome(spec)) in excluded_genome_hashes:
            duplicate_genomes+=1
            for _ in range(12):
                spec=make_spec()
                if genome(spec) not in evaluated and genome_hash(genome(spec)) not in excluded_genome_hashes: break
            if genome(spec) in evaluated or genome_hash(genome(spec)) in excluded_genome_hashes:
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
    from .audit import independent_replay, FIELDS, open_position_matches
    audit_failures=[]
    audited=[]
    passing_ids={id(row) for row in passing}
    audit_pool=passing+[row for row in results if id(row) not in passing_ids]
    for audit_index,row in enumerate(audit_pool):
        try:
            primary=(backtest(frame,StrategySpec(**row["strategy"]),cfg,_prepared_context=prepared_context)
                     if prepared_context is not None else backtest(frame,StrategySpec(**row["strategy"]),cfg))
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
            matched = matched and open_position_matches(primary, second)
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
        # Audit every raw passing candidate. If none survives independent replay,
        # continue through ranked non-passing candidates to provide verified near-misses.
        has_verified_pass=any(item["metrics"]["passes_all_targets"] for item in audited)
        if audit_index>=len(passing) and len(audited)>=5 and (not passing or has_verified_pass):
            break
    verified_passing=[r for r in audited if r["metrics"]["passes_all_targets"]]
    if verified_passing:
        selected=verified_passing
        all_passing=True
    else:
        # Fail closed: a raw pass that fails independent replay is not a pass.
        # Return up to five independently matched near-misses instead.
        selected=[r for r in audited if not r["metrics"]["passes_all_targets"]][:5]
        all_passing=False
    near_miss_shortfall=max(0,5-len(selected)) if not all_passing else 0
    def _pilot_pf(metrics):
        return math.inf if metrics.get("profit_factor") is None else float(metrics["profit_factor"])
    preflight_summary={
        "evaluated_candidates":len(results),
        "best_tp_per_day":max((float(r["metrics"].get("avg_net_tp_per_day",0.0)) for r in results),default=0.0),
        "best_trade_count":max((int(r["metrics"].get("trades",0)) for r in results),default=0),
        "candidates_tp_per_day_ge_0_5":sum(float(r["metrics"].get("avg_net_tp_per_day",0.0))>=0.5 for r in results),
        "candidates_trades_ge_50":sum(int(r["metrics"].get("trades",0))>=50 for r in results),
        "candidates_win_rate_ge_30":sum(float(r["metrics"].get("win_rate_pct",0.0))>=30.0 for r in results),
        "candidates_pf_ge_0_7":sum(_pilot_pf(r["metrics"])>=0.7 for r in results),
        "candidates_frequency_and_trade_count":sum(
            float(r["metrics"].get("avg_net_tp_per_day",0.0))>=0.5 and int(r["metrics"].get("trades",0))>=50
            for r in results
        ),
        "capability_evidence_candidates":sum(
            float(r["metrics"].get("avg_net_tp_per_day",0.0))>=0.5
            and int(r["metrics"].get("trades",0))>=50
            and (float(r["metrics"].get("win_rate_pct",0.0))>=25.0 or _pilot_pf(r["metrics"])>=0.5)
            for r in results
        ),
        "promising_candidates":sum(
            float(r["metrics"].get("avg_net_tp_per_day",0.0))>=0.5
            and float(r["metrics"].get("win_rate_pct",0.0))>=30.0
            and _pilot_pf(r["metrics"])>=0.7
            and int(r["metrics"].get("trades",0))>=50
            for r in results
        ),
    }
    for row in results: row.pop("_genome",None)
    return {"requested_candidates":max_candidates,"evaluated_candidates":len(results),
            "failed_candidates":errors,"duplicate_genomes_skipped":duplicate_genomes,
            "seed":seed,"algorithm":algorithm,"generations":generation,"surrogate_model_fits":surrogate_fit_count,
            "family_evaluated_counts":family_counts,"contract":cfg,"all_passing":all_passing,
            "prior_candidates_used":len(prior_rows),"preflight_summary":preflight_summary,            "indicator_cache_stats":(
                {"hits":prepared_context.indicator_cache_hits,
                 "misses":prepared_context.indicator_cache_misses,
                 "entries":len(prepared_context.indicator_cache),
                 "limit":prepared_context.indicator_cache_limit}
                if prepared_context is not None else {"hits":0,"misses":0,"entries":0,"limit":128}),
            "evaluated_genome_hashes":[genome_hash(key) for key in evaluated.keys()],
            "independent_audit_failures":audit_failures,
            "results":selected,"requested_near_misses":0 if all_passing else 5,
            "returned_near_misses":0 if all_passing else len(selected),"near_miss_shortfall":near_miss_shortfall,
            "warning":("Deterministic stratified random baseline; selected results independently replayed; OOS not certified." if algorithm=="stratified_random" else
                       ("Surrogate-assisted genetic search: ExtraTrees model ranks proposal pools; model fits and independent ledger replay are recorded. This is one 10,000-candidate batch inside the resource-managed background search; ranked on TRAIN only. Full 500k runtime has not been production-certified." if algorithm=="surrogate_genetic" else
                        "Hybrid genetic search with stratified seeds, crossover, mutation, immigrant exploration and independent ledger replay of reported results. This invocation is one 10,000-candidate batch inside the background orchestrator; ranked on TRAIN only. Full 500k runtime is not production-certified.")) }
