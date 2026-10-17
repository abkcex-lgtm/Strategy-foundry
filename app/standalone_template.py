'''Standalone replay: generated per run; does not import Strategy Foundry.'''
import sys, json, hashlib, math
import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from typing import Any
SPEC = __SPEC__
CFG = __CONTRACT__
EXPECTED_DATA_SHA256 = '__DATA_HASH__'
EXPECTED_RUN_SHA256 = '__RUN_HASH__'
def digest(b): return hashlib.sha256(b).hexdigest()
def validate(df):
    required=['open_time','open','high','low','close','volume']
    if any(c not in df for c in required): raise ValueError('Missing OHLCV columns')
    if df.empty or df[required].isna().any().any(): raise ValueError('Empty/missing OHLCV data')
    if df.open_time.duplicated().any() or not df.open_time.is_monotonic_increasing: raise ValueError('Duplicate or out-of-order timestamps')
    if (df[['open','high','low','close']]<=0).any().any() or (df.volume<0).any(): raise ValueError('Invalid price/volume')
    if (df.high<df[['open','close','low']].max(axis=1)).any() or (df.low>df[['open','close','high']].min(axis=1)).any(): raise ValueError('Invalid OHLC')
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
        if self.family not in {"ema_crossover","channel_breakout","rsi_reversion","volume_breakout","ema_pullback","macd_crossover","bollinger_reversion","inside_bar_breakout","engulfing_reversal","momentum_breakout","stochastic_reversion","roc_momentum","adx_trend","keltner_breakout","atr_squeeze_breakout","obv_trend","vwap_reclaim","pivot_reversal","swing_structure_break","failed_breakout_reversal","wick_rejection","range_expansion_breakout","three_bar_momentum","volatility_regime_breakout","trend_exhaustion_reversal","volume_climax_reversal","range_compression_breakout","support_resistance_rejection","micro_double_bottom","micro_double_top"}:
            raise ValueError("Unsupported strategy family")
        valid_families={"ema_crossover","channel_breakout","rsi_reversion","volume_breakout","ema_pullback","macd_crossover","bollinger_reversion","inside_bar_breakout","engulfing_reversal","momentum_breakout","stochastic_reversion","roc_momentum","adx_trend","keltner_breakout","atr_squeeze_breakout","obv_trend","vwap_reclaim","pivot_reversal","swing_structure_break","failed_breakout_reversal","wick_rejection","range_expansion_breakout","three_bar_momentum","volatility_regime_breakout","trend_exhaustion_reversal","volume_climax_reversal","range_compression_breakout","support_resistance_rejection","micro_double_bottom","micro_double_top"}
        if self.secondary_family not in valid_families | {"none"}: raise ValueError("Unsupported secondary strategy family")
        if self.signal_composition not in {"single","and","or","xor"}: raise ValueError("Unsupported signal composition")
        if self.signal_composition=="single" and self.secondary_family!="none": raise ValueError("Single-family strategy cannot set secondary_family")
        if self.signal_composition!="single" and (self.secondary_family=="none" or self.secondary_family==self.family): raise ValueError("Composite strategy requires a different secondary family")
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
    x["long_signal"]=long_signal.fillna(False)
    x["short_signal"]=short_signal.fillna(False)
    return x

def risk_budget(balance):
    if CFG['risk_mode']=='fixed_usd': return float(CFG['risk_usd'])
    base=balance if (CFG['risk_mode']=='percent_equity' or CFG['compound']) else float(CFG['initial_balance'])
    return base*float(CFG['risk_pct'])/100
def size_target(entry,stop,side,risk):
    rate=(float(CFG['commission_pct_per_side'])+float(CFG['slippage_pct_per_side']))/100
    dist=entry-stop if side=='LONG' else stop-entry
    if dist<=0: raise ValueError('Invalid stop placement')
    qty=risk/(dist+entry*rate+stop*rate)
    if CFG['target_mode']=='gross_rr':
        target=entry+dist*float(CFG['reward_risk']) if side=='LONG' else entry-dist*float(CFG['reward_risk'])
    else:
        reward=risk*float(CFG['reward_risk'])
        target=(entry*(1+rate)+reward/qty)/(1-rate) if side=='LONG' else (entry*(1-rate)-reward/qty)/(1+rate)
    if target<=0: raise ValueError('Non-positive target')
    return qty,target,rate
def resolve_execution_interval(side, execution_df, start_ms, interval_ms, stop, target):
    if interval_ms < 1000 or interval_ms % 1000: raise ValueError('Base interval cannot be resolved with 1-second data')
    times=execution_df.open_time.to_numpy(dtype=np.int64)
    first=int(np.searchsorted(times,start_ms,side='left')); end_ms=start_ms+interval_ms
    last=int(np.searchsorted(times,end_ms,side='left')); expected=interval_ms//1000
    bars=execution_df.iloc[first:last]
    if len(bars)!=expected or int(bars.open_time.iloc[0])!=start_ms or int(bars.open_time.iloc[-1])!=end_ms-1000:
        raise ValueError('Missing/incomplete 1-second execution data')
    for bar in bars.itertuples(index=False):
        op=float(bar.open); hi=float(bar.high); lo=float(bar.low)
        if side=='LONG':
            if op<=stop: return 'SL',op,int(bar.open_time)
            if op>=target: return 'TP',target,int(bar.open_time)
            sl=lo<=stop; tp=hi>=target
        else:
            if op>=stop: return 'SL',op,int(bar.open_time)
            if op<=target: return 'TP',target,int(bar.open_time)
            sl=hi>=stop; tp=lo<=target
        if sl and tp: raise ValueError('Unresolved same-second SL/TP ambiguity; tick-level data required')
        if sl: return 'SL',stop,int(bar.open_time)
        if tp: return 'TP',target,int(bar.open_time)
    return None


def replay(df, execution_df=None):
    df=df.copy().reset_index(drop=True); validate(df)
    if len(df)>1:
        diffs=df.open_time.diff().dropna()
        interval=int(diffs.mode().iloc[0]) if len(diffs) else None
        if interval is not None and (diffs != interval).any():
            raise ValueError('Timestamp gaps detected; replay requires contiguous candles')
    data_hash=digest(df.to_csv(index=False,float_format='%.12g').encode())
    execution_hash=None
    base_interval=int(df.open_time.diff().dropna().mode().iloc[0]) if len(df)>1 else None
    if execution_df is not None:
        execution_df=execution_df.copy().reset_index(drop=True); validate(execution_df)
        # Missing seconds are resolved per held-trade interval as conservative losses.
        if base_interval is None or base_interval%1000: raise ValueError('Base interval cannot be resolved with 1-second data')
        execution_hash=digest(execution_df.to_csv(index=False,float_format='%.12g').encode())
    s=SPEC
    spec_obj=StrategySpec(**s)
    spec_obj.validate()
    x=indicators(df,spec_obj)
    balance=float(CFG['initial_balance']); ledger=[]; streak=0; max_streak=0; i=max(s['slow'],s['atr_period'])
    while i<len(x)-1:
        row=x.iloc[i]
        if not np.isfinite(row.atr) or row.atr<=0: i+=1; continue
        side='LONG' if bool(row.long_signal) else ('SHORT' if bool(row.short_signal) else None)
        if side is None or (s['direction']=='long' and side=='SHORT') or (s['direction']=='short' and side=='LONG'): i+=1; continue
        entry_i=i+1; entry=float(x.iloc[entry_i].open); dist=float(row.atr*s['stop_atr'])
        stop=entry-dist if side=='LONG' else entry+dist
        risk=risk_budget(balance if (CFG['compound'] or CFG['risk_mode']=='percent_equity') else CFG['initial_balance'])
        if risk<=0 or (CFG['risk_mode']=='percent_equity' and risk>=balance): i+=1; continue
        try: qty,target,rate=size_target(entry,stop,side,risk)
        except ValueError: i+=1; continue
        exit_i=exit_price=reason=exit_time=None; resolution_status='resolved_base_ohlc'
        for j in range(entry_i,len(x)):
            bar=x.iloc[j]; bar_open=float(bar.open)
            if execution_df is not None:
                try:
                    resolution=resolve_execution_interval(side,execution_df,int(bar.open_time),base_interval,stop,target)
                except ValueError:
                    reason,exit_price,exit_time='SL_UNRESOLVED',stop,int(bar.open_time)
                    exit_i=j; resolution_status='unresolved_as_loss'; break
                if resolution is not None:
                    reason,exit_price,exit_time=resolution; exit_i=j; resolution_status='resolved_1s_ohlc'; break
                continue
            if side=='LONG':
                if bar_open<=stop: exit_i=j; exit_price=bar_open; exit_time=int(bar.open_time); reason='SL'; break
                if bar_open>=target: exit_i=j; exit_price=target; exit_time=int(bar.open_time); reason='TP'; break
                sl=float(bar.low)<=stop; tp=float(bar.high)>=target
            else:
                if bar_open>=stop: exit_i=j; exit_price=bar_open; exit_time=int(bar.open_time); reason='SL'; break
                if bar_open<=target: exit_i=j; exit_price=target; exit_time=int(bar.open_time); reason='TP'; break
                sl=float(bar.high)>=stop; tp=float(bar.low)<=target
            if sl and tp:
                reason='SL_UNRESOLVED'; exit_i=j; exit_price=stop; exit_time=int(bar.open_time); resolution_status='unresolved_as_loss'; break
            if sl: exit_i=j; exit_price=stop; exit_time=int(bar.open_time); reason='SL'; resolution_status='resolved_base_ohlc'; break
            if tp: exit_i=j; exit_price=target; exit_time=int(bar.open_time); reason='TP'; resolution_status='resolved_base_ohlc'; break
        if exit_i is None: break
        gross=(exit_price-entry)*qty*(1 if side=='LONG' else -1); entry_cost=entry*qty*rate; exit_cost=exit_price*qty*rate; net=gross-entry_cost-exit_cost; balance+=net
        if reason=='TP': streak=0
        else: streak+=1; max_streak=max(max_streak,streak)
        ledger.append({'trade_id':len(ledger)+1,'signal_time':int(row.open_time),'entry_time':int(x.iloc[entry_i].open_time),'exit_time':int(exit_time),'direction':side,'signal_price':float(row.close),'entry_price':entry,'stop_loss':stop,'take_profit':target,'exit_price':float(exit_price),'exit_reason':reason,'quantity':qty,'risk_budget':risk,'gross_pnl':gross,'entry_cost':entry_cost,'exit_cost':exit_cost,'fees_slippage':entry_cost+exit_cost,'net_pnl':net,'balance_after':balance,'loss_streak_after':streak,'resolution_status':resolution_status,'execution_ambiguity':resolution_status=='unresolved_as_loss'})
        i=exit_i+1
    wins=[t for t in ledger if t['exit_reason']=='TP' and t['net_pnl']>0]; gp=sum(max(0,t['net_pnl']) for t in ledger); gl=abs(sum(min(0,t['net_pnl']) for t in ledger))
    span=max((int(df.open_time.iloc[-1])-int(df.open_time.iloc[0]))/86400000,1/1440)
    wr=100*len(wins)/len(ledger) if ledger else 0.0; pf=gp/gl if gl else (math.inf if gp else 0.0); tpday=len(wins)/span
    metrics={'trades':len(ledger),'wins':len(wins),'losses':sum(t['net_pnl']<=0 for t in ledger),'win_rate_pct':wr,'profit_factor':pf if math.isfinite(pf) else None,'avg_net_tp_per_day':tpday,'max_consecutive_losses':max_streak,'net_pnl':sum(t['net_pnl'] for t in ledger),'final_balance':balance,'total_costs':sum(t['fees_slippage'] for t in ledger),'data_hash_sha256':data_hash,'execution_data_hash_sha256':execution_hash,'execution_resolution':'1s_ohlc' if execution_df is not None else 'base_ohlc_only','unresolved_as_loss_count':sum(t.get('execution_ambiguity',False) for t in ledger),'rows':len(df),'span_days':span,'passes_all_targets':bool((not CFG.get('use_min_win_rate',True) or wr>CFG['min_win_rate_pct']) and (not CFG.get('use_min_profit_factor',True) or pf>CFG['min_profit_factor']) and (not CFG.get('use_min_tp_per_day',True) or tpday>=CFG['min_tp_per_day']) and (not CFG.get('use_max_consecutive_losses',True) or max_streak<=CFG['max_consecutive_losses']))}
    run_hash=digest(json.dumps({'spec':s,'contract':CFG,'data_hash':data_hash,'execution_data_hash':execution_hash,'execution_resolution':'1s_ohlc' if execution_df is not None else 'base_ohlc_only'},sort_keys=True).encode())
    return {'metrics':metrics,'ledger':ledger,'run_hash_sha256':run_hash}
if __name__=='__main__':
    if len(sys.argv) not in (2,3): raise SystemExit('Usage: python replay.py EXACT_SOURCE.csv [EXACT_1S_EXECUTION.csv]')
    result=replay(pd.read_csv(sys.argv[1]),pd.read_csv(sys.argv[2]) if len(sys.argv)==3 else None)
    if result['metrics']['data_hash_sha256']!=EXPECTED_DATA_SHA256: raise SystemExit('DATA HASH MISMATCH')
    if result['run_hash_sha256']!=EXPECTED_RUN_SHA256: raise SystemExit('RUN HASH MISMATCH')
    print(json.dumps(result,indent=2,ensure_ascii=False))
