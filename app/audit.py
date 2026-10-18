"""Independent audit engine. Intentionally does not import app.engine."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from .data import download_binance_execution_window

FIELDS = ("trade_id","signal_time","entry_time","exit_time","direction","signal_price",
          "entry_price","stop_loss","take_profit","exit_price","exit_reason","quantity",
          "risk_budget","gross_pnl","entry_cost","exit_cost","fees_slippage","net_pnl",
          "balance_after","loss_streak_after","resolution_status","execution_ambiguity")

def _independent_signals(d: pd.DataFrame, spec: dict[str, Any]):
    close=d["close"].astype(float); prev=close.shift(1)
    fast=int(spec["fast"]); slow=int(spec["slow"]); period=int(spec["atr_period"])
    ef=close.ewm(span=fast,adjust=False,min_periods=fast).mean()
    es=close.ewm(span=slow,adjust=False,min_periods=slow).mean()
    tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    long=(ef>es)&(ef.shift(1)<=es.shift(1))
    short=(ef<es)&(ef.shift(1)>=es.shift(1))
    family=spec.get("family","ema_crossover")
    if family in {"channel_breakout","volume_breakout"}:
        upper=d.high.rolling(slow,min_periods=slow).max().shift(1)
        lower=d.low.rolling(slow,min_periods=slow).min().shift(1)
        long=(close>upper)&(close.shift(1)<=upper.shift(1))
        short=(close<lower)&(close.shift(1)>=lower.shift(1))
        if family=="volume_breakout":
            baseline=d.volume.rolling(fast,min_periods=fast).mean().shift(1)
            active=d.volume>=baseline*1.5
            long=long&active
            short=short&active
    elif family=="rsi_reversion":
        delta=close.diff()
        gain=delta.clip(lower=0).ewm(alpha=1/fast,adjust=False,min_periods=fast).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/fast,adjust=False,min_periods=fast).mean()
        rs=gain/loss.replace(0,np.nan)
        rsi=100-100/(1+rs)
        long=(rsi.shift(1)<=30)&(rsi>30)
        short=(rsi.shift(1)>=70)&(rsi<70)
    elif family=="ema_pullback":
        trend_up=ef>es; trend_down=ef<es
        long=(trend_up)&(close>ef)&(close.shift(1)<=ef.shift(1))
        short=(trend_down)&(close<ef)&(close.shift(1)>=ef.shift(1))
    elif family=="macd_crossover":
        macd=close.ewm(span=fast,adjust=False,min_periods=fast).mean()-close.ewm(span=slow,adjust=False,min_periods=slow).mean()
        sig=macd.ewm(span=max(2,fast//2),adjust=False,min_periods=max(2,fast//2)).mean()
        long=(macd>sig)&(macd.shift(1)<=sig.shift(1)); short=(macd<sig)&(macd.shift(1)>=sig.shift(1))
    elif family=="bollinger_reversion":
        mid=close.rolling(slow,min_periods=slow).mean(); dev=close.rolling(slow,min_periods=slow).std(ddof=0)
        upper=mid+2*dev; lower=mid-2*dev
        long=(close>lower)&(close.shift(1)<=lower.shift(1)); short=(close<upper)&(close.shift(1)>=upper.shift(1))
    elif family=="inside_bar_breakout":
        inside=(d.high.shift(1)<d.high.shift(2))&(d.low.shift(1)>d.low.shift(2))
        long=inside&(close>d.high.shift(1))&(close.shift(1)<=d.high.shift(1))
        short=inside&(close<d.low.shift(1))&(close.shift(1)>=d.low.shift(1))
    elif family=="engulfing_reversal":
        po=d.open.shift(1); pc=close.shift(1)
        long=(close>d.open)&(pc<po)&(d.open<=pc)&(close>=po)
        short=(close<d.open)&(pc>po)&(d.open>=pc)&(close<=po)
    elif family=="momentum_breakout":
        upper=d.high.rolling(slow,min_periods=slow).max().shift(1); lower=d.low.rolling(slow,min_periods=slow).min().shift(1)
        long=(close>close.shift(fast))&(close>upper)&(close.shift(1)<=upper.shift(1))
        short=(close<close.shift(fast))&(close<lower)&(close.shift(1)>=lower.shift(1))
    elif family=="stochastic_reversion":
        lo=d.low.rolling(fast,min_periods=fast).min(); hi=d.high.rolling(fast,min_periods=fast).max()
        k=100*(close-lo)/(hi-lo).replace(0,np.nan)
        long=(k.shift(1)<=20)&(k>20); short=(k.shift(1)>=80)&(k<80)
    elif family=="roc_momentum":
        roc=close.pct_change(periods=fast)*100
        long=(roc>0)&(roc.shift(1)<=0); short=(roc<0)&(roc.shift(1)>=0)
    elif family=="adx_trend":
        up=d.high.diff(); down=-d.low.diff()
        plus=up.where((up>down)&(up>0),0.0).ewm(alpha=1/period,adjust=False,min_periods=period).mean()
        minus=down.where((down>up)&(down>0),0.0).ewm(alpha=1/period,adjust=False,min_periods=period).mean()
        plus_di=100*plus/atr.replace(0,np.nan); minus_di=100*minus/atr.replace(0,np.nan)
        dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
        adx=dx.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
        long=(plus_di>minus_di)&(plus_di.shift(1)<=minus_di.shift(1))&(adx>=20)
        short=(minus_di>plus_di)&(minus_di.shift(1)<=plus_di.shift(1))&(adx>=20)
    elif family=="keltner_breakout":
        mid=close.ewm(span=slow,adjust=False,min_periods=slow).mean()
        upper=mid+1.5*atr; lower=mid-1.5*atr
        long=(close>upper)&(close.shift(1)<=upper.shift(1))
        short=(close<lower)&(close.shift(1)>=lower.shift(1))
    elif family=="atr_squeeze_breakout":
        mid=close.rolling(slow,min_periods=slow).mean()
        dev=close.rolling(slow,min_periods=slow).std(ddof=0)
        width=4*dev/mid.replace(0,np.nan)
        squeezed=width<width.rolling(slow,min_periods=slow).quantile(0.25).shift(1)
        upper=d.high.rolling(slow,min_periods=slow).max().shift(1)
        lower=d.low.rolling(slow,min_periods=slow).min().shift(1)
        long=squeezed.shift(1)&(close>upper)&(close.shift(1)<=upper.shift(1))
        short=squeezed.shift(1)&(close<lower)&(close.shift(1)>=lower.shift(1))
    elif family=="obv_trend":
        signed=np.sign(close.diff()).fillna(0.0)*d.volume.astype(float)
        obv=signed.cumsum()
        obv_base=obv.ewm(span=slow,adjust=False,min_periods=slow).mean()
        long=(obv>obv_base)&(obv.shift(1)<=obv_base.shift(1))&(close>ef)
        short=(obv<obv_base)&(obv.shift(1)>=obv_base.shift(1))&(close<ef)
    elif family=="vwap_reclaim":
        typical=(d.high+d.low+close)/3; vol=d.volume.astype(float)
        vw=(typical*vol).rolling(slow,min_periods=slow).sum()/vol.rolling(slow,min_periods=slow).sum().replace(0,np.nan)
        long=(close>vw)&(close.shift(1)<=vw.shift(1))
        short=(close<vw)&(close.shift(1)>=vw.shift(1))
    elif family=="pivot_reversal":
        support=d.low.rolling(slow,min_periods=slow).min().shift(1)
        resistance=d.high.rolling(slow,min_periods=slow).max().shift(1)
        tolerance=atr*0.25
        long=(d.low<=support+tolerance)&(close>support)
        short=(d.high>=resistance-tolerance)&(close<resistance)
    elif family=="ema_pullback":
        long=(ef>es)&(close>ef)&(close.shift(1)<=ef.shift(1))
        short=(ef<es)&(close<ef)&(close.shift(1)>=ef.shift(1))
    elif family=="macd_crossover":
        macd=close.ewm(span=fast,adjust=False,min_periods=fast).mean()-close.ewm(span=slow,adjust=False,min_periods=slow).mean()
        sig=macd.ewm(span=max(2,fast//2),adjust=False,min_periods=max(2,fast//2)).mean()
        long=(macd>sig)&(macd.shift(1)<=sig.shift(1)); short=(macd<sig)&(macd.shift(1)>=sig.shift(1))
    elif family=="bollinger_reversion":
        mid=close.rolling(slow,min_periods=slow).mean(); dev=close.rolling(slow,min_periods=slow).std(ddof=0)
        upper=mid+2*dev; lower=mid-2*dev
        long=(close>lower)&(close.shift(1)<=lower.shift(1)); short=(close<upper)&(close.shift(1)>=upper.shift(1))
    elif family=="inside_bar_breakout":
        inside=(d.high.shift(1)<d.high.shift(2))&(d.low.shift(1)>d.low.shift(2))
        long=inside&(close>d.high.shift(1))&(close.shift(1)<=d.high.shift(1))
        short=inside&(close<d.low.shift(1))&(close.shift(1)>=d.low.shift(1))
    elif family=="engulfing_reversal":
        po=d.open.shift(1); pc=close.shift(1)
        long=(close>d.open)&(pc<po)&(d.open<=pc)&(close>=po)
        short=(close<d.open)&(pc>po)&(d.open>=pc)&(close<=po)
    elif family=="momentum_breakout":
        upper=d.high.rolling(slow,min_periods=slow).max().shift(1); lower=d.low.rolling(slow,min_periods=slow).min().shift(1)
        long=(close>close.shift(fast))&(close>upper)&(close.shift(1)<=upper.shift(1))
        short=(close<close.shift(fast))&(close<lower)&(close.shift(1)>=lower.shift(1))
    elif family=="swing_structure_break":
        upper=d.high.rolling(slow,min_periods=slow).max().shift(1); lower=d.low.rolling(slow,min_periods=slow).min().shift(1)
        long=(close>upper)&(close.shift(1)<=upper.shift(1)); short=(close<lower)&(close.shift(1)>=lower.shift(1))
    elif family=="failed_breakout_reversal":
        upper=d.high.rolling(slow,min_periods=slow).max().shift(2); lower=d.low.rolling(slow,min_periods=slow).min().shift(2)
        long=(d.low.shift(1)<lower)&(close.shift(1)<lower)&(close>lower)&(close>d.open)
        short=(d.high.shift(1)>upper)&(close.shift(1)>upper)&(close<upper)&(close<d.open)
    elif family=="wick_rejection":
        body=(close-d.open).abs().clip(lower=1e-12); lower_wick=np.minimum(d.open,close)-d.low; upper_wick=d.high-np.maximum(d.open,close)
        location=(close-d.low)/(d.high-d.low).replace(0,np.nan)
        long=(lower_wick>=2.0*body)&(location>=0.65)&(close>d.open); short=(upper_wick>=2.0*body)&(location<=0.35)&(close<d.open)
    elif family=="range_expansion_breakout":
        bar_range=d.high-d.low; baseline=bar_range.rolling(fast,min_periods=fast).mean().shift(1)
        long=(bar_range>=1.5*baseline)&(close>d.high.shift(1))&(close.shift(1)<=d.high.shift(1))
        short=(bar_range>=1.5*baseline)&(close<d.low.shift(1))&(close.shift(1)>=d.low.shift(1))
    elif family=="three_bar_momentum":
        upper=d.high.rolling(fast,min_periods=fast).max().shift(1); lower=d.low.rolling(fast,min_periods=fast).min().shift(1)
        rising=(close>close.shift(1))&(close.shift(1)>close.shift(2))&(close.shift(2)>close.shift(3)); falling=(close<close.shift(1))&(close.shift(1)<close.shift(2))&(close.shift(2)<close.shift(3))
        long=rising&(close>upper); short=falling&(close<lower)
    elif family=="volatility_regime_breakout":
        atr_base=atr.rolling(slow,min_periods=slow).median().shift(1)
        upper=d.high.rolling(slow,min_periods=slow).max().shift(1); lower=d.low.rolling(slow,min_periods=slow).min().shift(1)
        active=atr>atr_base
        long=active&(close>upper)&(close.shift(1)<=upper.shift(1)); short=active&(close<lower)&(close.shift(1)>=lower.shift(1))
    elif family=="trend_exhaustion_reversal":
        body=(close-d.open).abs().clip(lower=1e-12); lower_wick=np.minimum(d.open,close)-d.low; upper_wick=d.high-np.maximum(d.open,close)
        long=(close.shift(1)<close.shift(fast+1))&(close.shift(1)<d.open.shift(1))&(close>d.open)&(lower_wick>=body)
        short=(close.shift(1)>close.shift(fast+1))&(close.shift(1)>d.open.shift(1))&(close<d.open)&(upper_wick>=body)
    elif family=="volume_climax_reversal":
        vol_threshold=d.volume.rolling(slow,min_periods=slow).quantile(0.90).shift(1)
        body=(close-d.open).abs().clip(lower=1e-12); lower_wick=np.minimum(d.open,close)-d.low; upper_wick=d.high-np.maximum(d.open,close)
        long=(d.volume>vol_threshold)&(lower_wick>=1.5*body)&(close>d.open); short=(d.volume>vol_threshold)&(upper_wick>=1.5*body)&(close<d.open)
    elif family=="range_compression_breakout":
        bar_range=d.high-d.low; avg_range=bar_range.rolling(fast,min_periods=fast).mean()
        compression_level=avg_range.rolling(slow,min_periods=slow).quantile(0.25).shift(1); compressed=avg_range.shift(1)<compression_level
        upper=d.high.rolling(slow,min_periods=slow).max().shift(1); lower=d.low.rolling(slow,min_periods=slow).min().shift(1)
        long=compressed&(close>upper)&(close.shift(1)<=upper.shift(1)); short=compressed&(close<lower)&(close.shift(1)>=lower.shift(1))
    elif family=="support_resistance_rejection":
        support=d.low.rolling(slow,min_periods=slow).min().shift(1); resistance=d.high.rolling(slow,min_periods=slow).max().shift(1)
        body=(close-d.open).abs().clip(lower=1e-12); lower_wick=np.minimum(d.open,close)-d.low; upper_wick=d.high-np.maximum(d.open,close)
        long=(d.low<=support+atr*0.3)&(close>support)&(lower_wick>=body); short=(d.high>=resistance-atr*0.3)&(close<resistance)&(upper_wick>=body)
    elif family=="micro_double_bottom":
        current_low=d.low.rolling(fast,min_periods=fast).min(); prior_low=d.low.shift(fast).rolling(fast,min_periods=fast).min()
        trigger=d.high.shift(1).rolling(fast,min_periods=fast).max()
        long=((current_low-prior_low).abs()<=atr*0.35)&(d.low<=current_low+atr*0.1)&(close>trigger)&(close>d.open); short=pd.Series(False,index=d.index)
    elif family=="micro_double_top":
        current_high=d.high.rolling(fast,min_periods=fast).max(); prior_high=d.high.shift(fast).rolling(fast,min_periods=fast).max()
        trigger=d.low.shift(1).rolling(fast,min_periods=fast).min()
        short=((current_high-prior_high).abs()<=atr*0.35)&(d.high>=current_high-atr*0.1)&(close<trigger)&(close<d.open); long=pd.Series(False,index=d.index)
    context_filter=spec.get("context_filter","none")
    confirmation_filter=spec.get("confirmation_filter","none")
    if context_filter=="ema_trend":
        long=long&(ef>es); short=short&(ef<es)
    elif context_filter=="adx_strength":
        up=d.high.diff(); down=-d.low.diff()
        plus=up.where((up>down)&(up>0),0.0).ewm(alpha=1/period,adjust=False,min_periods=period).mean()
        minus=down.where((down>up)&(down>0),0.0).ewm(alpha=1/period,adjust=False,min_periods=period).mean()
        pdi=100*plus/atr.replace(0,np.nan); mdi=100*minus/atr.replace(0,np.nan)
        dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
        adx=dx.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
        long=long&(adx>=20)&(pdi>mdi); short=short&(adx>=20)&(mdi>pdi)
    elif context_filter=="volume_confirm":
        baseline=d.volume.rolling(fast,min_periods=fast).mean().shift(1)
        active=d.volume>=baseline*1.5
        long=long&active; short=short&active
    elif context_filter=="vwap_side":
        typical=(d.high+d.low+close)/3; vol=d.volume.astype(float)
        vw=(typical*vol).rolling(slow,min_periods=slow).sum()/vol.rolling(slow,min_periods=slow).sum().replace(0,np.nan)
        long=long&(close>vw); short=short&(close<vw)
    elif context_filter=="low_volatility":
        ratio=atr/close.replace(0,np.nan)
        threshold=ratio.rolling(slow,min_periods=slow).quantile(0.35).shift(1)
        calm=ratio<=threshold
        long=long&calm; short=short&calm
    if confirmation_filter=="candle_direction":
        long=long&(close>d.open); short=short&(close<d.open)
    elif confirmation_filter=="momentum":
        long=long&(close>close.shift(fast)); short=short&(close<close.shift(fast))
    elif confirmation_filter=="rsi_midline":
        delta=close.diff()
        gain=delta.clip(lower=0).ewm(alpha=1/period,adjust=False,min_periods=period).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/period,adjust=False,min_periods=period).mean()
        rsi=100-100/(1+gain/loss.replace(0,np.nan))
        long=long&(rsi>50); short=short&(rsi<50)
    if spec.get("signal_composition","single")!="single":
        secondary_spec={**spec,"family":spec.get("secondary_family","none"),
                       "secondary_family":"none","signal_composition":"single"}
        sec_long,sec_short,_,_,_,_=_independent_signals(d,secondary_spec)
        operation=spec.get("signal_composition")
        if operation=="and":
            long=long&sec_long; short=short&sec_short
        elif operation=="or":
            long=long|sec_long; short=short|sec_short
        elif operation=="xor":
            long=long^sec_long; short=short^sec_short
        else:
            raise ValueError("Unsupported signal composition")
    long=long.fillna(False)
    short=short.fillna(False)
    return long,short,atr,fast,slow,period


def _resolve_bar_exit_independent(side: str, bar_open: float, bar_high: float, bar_low: float, stop: float, target: float):
    """Independent conservative OHLC exit resolution; gap fills take precedence."""
    if side == "LONG":
        if bar_open <= stop: return "SL", bar_open
        if bar_open >= target: return "TP", target
        sl = bar_low <= stop; tp = bar_high >= target
    else:
        if bar_open >= stop: return "SL", bar_open
        if bar_open <= target: return "TP", target
        sl = bar_high >= stop; tp = bar_low <= target
    if sl and tp: raise ValueError("UNRESOLVED_AMBIGUITY: finer data required")
    if sl: return "SL", stop
    if tp: return "TP", target
    return None


def _resolve_execution_interval_independent(side, execution_frame, start_ms, interval_ms, stop, target):
    if interval_ms < 1000 or interval_ms % 1000:
        raise ValueError("Base interval cannot be resolved with 1-second data")
    times=execution_frame["open_time"].to_numpy(dtype=np.int64)
    first=int(np.searchsorted(times,start_ms,side="left")); end_ms=start_ms+interval_ms
    last=int(np.searchsorted(times,end_ms,side="left")); expected=interval_ms//1000
    bars=execution_frame.iloc[first:last]
    if len(bars)!=expected or int(bars.open_time.iloc[0])!=start_ms or int(bars.open_time.iloc[-1])!=end_ms-1000:
        raise ValueError("Missing/incomplete 1-second execution data")
    for bar in bars.itertuples(index=False):
        result=_resolve_bar_exit_independent(side,float(bar.open),float(bar.high),float(bar.low),stop,target)
        if result is not None:
            reason,price=result
            return reason,price,int(bar.open_time)
    return None


def independent_replay(frame: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any], execution_frame=None) -> dict[str, Any]:
    """Recompute the trade ledger independently for deterministic cross-engine checks."""
    d=frame.copy().reset_index(drop=True)
    # Normalize independently: CSV/SQLite can infer OHLCV columns as strings.
    required=["open_time","open","high","low","close","volume"]
    try:
        for column in required:
            d[column]=pd.to_numeric(d[column],errors="raise")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Independent replay requires numeric OHLCV columns: {exc}") from exc
    if d["open_time"].isna().any() or not np.isfinite(d["open_time"].to_numpy(dtype=float)).all():
        raise ValueError("Independent replay found invalid open_time timestamps")
    if not np.equal(d["open_time"].to_numpy(dtype=float),np.floor(d["open_time"].to_numpy(dtype=float))).all():
        raise ValueError("Independent replay requires integer-millisecond open_time timestamps")
    d["open_time"]=d["open_time"].astype("int64")
    exec_df=None
    base_interval=int(d.open_time.diff().dropna().mode().iloc[0]) if len(d)>1 else None
    if execution_frame is not None:
        exec_df=execution_frame.copy().reset_index(drop=True)
        required=["open_time","open","high","low","close","volume"]
        if any(c not in exec_df.columns for c in required):
            raise ValueError("Missing columns in execution data")
        for c in required: exec_df[c]=pd.to_numeric(exec_df[c],errors="coerce")
        if exec_df[required].isna().any().any() or exec_df.open_time.duplicated().any() or not exec_df.open_time.is_monotonic_increasing:
            raise ValueError("Invalid execution data")
        # Gaps are evaluated per held-trade interval so they can be conservatively
        # recorded as losses instead of aborting the entire replay.
        if base_interval is None or base_interval % 1000:
            raise ValueError("Base interval cannot be resolved with 1-second data")
    close=d["close"].astype(float)
    long,short,atr,fast,slow,period=_independent_signals(d,spec)
    rate=(float(cfg["commission_pct_per_side"])+float(cfg["slippage_pct_per_side"]))/100.0
    balance=float(cfg["initial_balance"]); streak=0; ledger=[]; max_streak=0; open_position=None
    i=max(slow,period)
    while i < len(d)-1:
        if not np.isfinite(atr.iloc[i]) or atr.iloc[i] <= 0: i+=1; continue
        side="LONG" if bool(long.iloc[i]) else ("SHORT" if bool(short.iloc[i]) else None)
        if side is None or (spec.get("direction","both")=="long" and side=="SHORT") or (spec.get("direction","both")=="short" and side=="LONG"):
            i+=1; continue
        ei=i+1; entry=float(d.open.iloc[ei]); distance=float(atr.iloc[i]*float(spec["stop_atr"]))
        stop=entry-distance if side=="LONG" else entry+distance
        mode=cfg["risk_mode"]
        base=(balance if mode=="percent_equity" or cfg.get("compound") else float(cfg["initial_balance"]))
        risk=float(cfg["risk_usd"]) if mode=="fixed_usd" else base*float(cfg["risk_pct"])/100
        if risk<=0: i+=1; continue
        # Match the primary engine: never open a trade that risks all remaining equity.
        if balance<=risk+1e-9: break
        dist=entry-stop if side=="LONG" else stop-entry
        qty=risk/(dist+entry*rate+stop*rate)
        if cfg["target_mode"]=="gross_rr":
            target=entry+dist*float(cfg["reward_risk"]) if side=="LONG" else entry-dist*float(cfg["reward_risk"])
        else:
            reward=risk*float(cfg["reward_risk"])
            target=(entry*(1+rate)+reward/qty)/(1-rate) if side=="LONG" else (entry*(1-rate)-reward/qty)/(1+rate)
        if target<=0: i+=1; continue
        ex=None; px=None; reason=None; exit_time=None; resolution_status="resolved_base_ohlc"
        for j in range(ei,len(d)):
            if exec_df is not None:
                try:
                    resolution=_resolve_execution_interval_independent(side,exec_df,int(d.open_time.iloc[j]),base_interval,stop,target)
                except ValueError as exc:
                    # Independent replay must reject unresolved execution; never manufacture a stop fill.
                    raise ValueError(f"EXECUTION_PREFLIGHT_FAILED at {int(d.open_time.iloc[j])}: {exc}") from exc
                if resolution is not None:
                    reason,px,exit_time=resolution; ex=j; resolution_status="resolved_1s_ohlc"; break
            else:
                bar=d.iloc[j]
                try:
                    resolution=_resolve_bar_exit_independent(side,float(bar.open),float(bar.high),float(bar.low),stop,target)
                except ValueError as exc:
                    if "UNRESOLVED_AMBIGUITY" not in str(exc):
                        raise ValueError(f"EXECUTION_PREFLIGHT_FAILED at {int(d.open_time.iloc[j])}: {exc}") from exc
                    try:
                        seconds=download_binance_execution_window(str(cfg.get("symbol","ETHUSDT")),int(d.open_time.iloc[j]),base_interval)
                    except Exception as fetch_exc:
                        raise ValueError(f"UNRESOLVED_AMBIGUITY at {int(d.open_time.iloc[j])}; automatic 1-second resolution failed: {fetch_exc}") from fetch_exc
                    resolution=_resolve_execution_interval_independent(side,seconds,int(d.open_time.iloc[j]),base_interval,stop,target)
                    if resolution is None:
                        raise ValueError(f"1-second data contradicts ambiguous base candle at {int(d.open_time.iloc[j])}")
                    if resolution is not None:
                        reason,px,exit_time=resolution; ex=j; resolution_status="resolved_1s_ohlc_auto"; break
                if resolution is not None:
                    reason,px=resolution; ex=j; exit_time=int(d.open_time.iloc[j]); resolution_status="resolved_base_ohlc"; break
        if ex is None:
            last_price=float(d.close.iloc[-1])
            unrealized_gross=(last_price-entry)*qty*(1 if side=="LONG" else -1)
            open_position={"signal_time":int(d.open_time.iloc[i]),"entry_time":int(d.open_time.iloc[ei]),"direction":side,
                           "entry_price":entry,"stop_loss":stop,"take_profit":target,"quantity":qty,"risk_budget":risk,
                           "last_price":last_price,"unrealized_gross_pnl":unrealized_gross,
                           "unrealized_net_pnl":unrealized_gross-entry*qty*rate-last_price*qty*rate}
            break
        gross=(px-entry)*qty*(1 if side=="LONG" else -1)
        ec=entry*qty*rate; xc=px*qty*rate; net=gross-ec-xc; balance+=net
        streak=0 if reason=="TP" else streak+1
        max_streak=max(max_streak,streak)
        ledger.append({"trade_id":len(ledger)+1,"signal_time":int(d.open_time.iloc[i]),"entry_time":int(d.open_time.iloc[ei]),
          "exit_time":int(exit_time),"direction":side,"signal_price":float(close.iloc[i]),"entry_price":entry,
          "stop_loss":stop,"take_profit":target,"exit_price":float(px),"exit_reason":reason,"quantity":qty,"risk_budget":risk,
          "gross_pnl":gross,"entry_cost":ec,"exit_cost":xc,"fees_slippage":ec+xc,"net_pnl":net,
          "balance_after":balance,"loss_streak_after":streak,"resolution_status":resolution_status,"execution_ambiguity":resolution_status=="unresolved_as_loss"})
        i=ex+1
    return {"ledger":ledger,"max_consecutive_losses":max_streak,"open_position":open_position,"open_positions_at_end":1 if open_position else 0,"open_position_unrealized_net_pnl":open_position["unrealized_net_pnl"] if open_position else 0.0}


def open_position_matches(primary: dict[str, Any], independent: dict[str, Any]) -> bool:
    """Compare end-of-window open-position state independently of closed-trade rows."""
    left=primary.get("open_position"); right=independent.get("open_position")
    if left is None or right is None: return left is None and right is None
    if set(left) != set(right): return False
    for key in left:
        x=left[key]; y=right[key]
        if isinstance(x,(int,float)) and not isinstance(x,bool):
            if not math.isclose(float(x),float(y),rel_tol=1e-10,abs_tol=1e-8): return False
        elif x != y: return False
    return True
