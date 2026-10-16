"""Independent audit engine. Intentionally does not import app.engine."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

FIELDS = ("trade_id","signal_time","entry_time","exit_time","direction","signal_price",
          "entry_price","stop_loss","take_profit","exit_price","exit_reason","quantity",
          "risk_budget","gross_pnl","entry_cost","exit_cost","fees_slippage","net_pnl",
          "balance_after","loss_streak_after")

def independent_replay(frame: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Recompute the trade ledger independently for deterministic cross-engine checks."""
    d=frame.copy().reset_index(drop=True)
    close=d["close"].astype(float); prev=close.shift(1)
    fast=int(spec["fast"]); slow=int(spec["slow"]); period=int(spec["atr_period"])
    ef=close.ewm(span=fast,adjust=False,min_periods=fast).mean()
    es=close.ewm(span=slow,adjust=False,min_periods=slow).mean()
    tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
    long=(ef>es)&(ef.shift(1)<=es.shift(1))
    short=(ef<es)&(ef.shift(1)>=es.shift(1))
    family=spec.get("family","ema_cross")
    lookback=int(spec.get("lookback",20)); rsi_period=int(spec.get("rsi_period",14))
    rsi_low=float(spec.get("rsi_low",30)); rsi_high=float(spec.get("rsi_high",70))
    volume_mult=float(spec.get("volume_mult",1.5)); band_std=float(spec.get("band_std",2.0))
    momentum_threshold=float(spec.get("momentum_threshold",0.0))
    o=d["open"].astype(float); h=d["high"].astype(float); l=d["low"].astype(float); v=d["volume"].astype(float)
    if family=="momentum":
        mom=close.pct_change(lookback); long=mom>momentum_threshold; short=mom < -momentum_threshold
    elif family in {"rsi_reversal","range_reversion"}:
        delta=close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/rsi_period,adjust=False,min_periods=rsi_period).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/rsi_period,adjust=False,min_periods=rsi_period).mean()
        rsi=100-(100/(1+gain/loss.replace(0,np.nan)))
        if family=="rsi_reversal":
            long=(rsi.shift(1)<rsi_low)&(rsi>=rsi_low); short=(rsi.shift(1)>rsi_high)&(rsi<=rsi_high)
        else:
            long=rsi<=rsi_low; short=rsi>=rsi_high
    elif family=="bollinger_breakout":
        mid=close.rolling(lookback,min_periods=lookback).mean()
        sd=close.rolling(lookback,min_periods=lookback).std(ddof=0)
        upper=mid+band_std*sd; lower=mid-band_std*sd
        long=(close>upper)&(prev<=upper.shift(1)); short=(close<lower)&(prev>=lower.shift(1))
    elif family in {"donchian_breakout","channel_breakout","support_resistance","volume_breakout"}:
        ceiling=h.shift(1).rolling(lookback,min_periods=lookback).max()
        floor=l.shift(1).rolling(lookback,min_periods=lookback).min()
        if family=="support_resistance":
            long=(l<=floor*1.002)&(close>o)&(close.shift(1)<=o.shift(1))
            short=(h>=ceiling*0.998)&(close<o)&(close.shift(1)>=o.shift(1))
        elif family=="volume_breakout":
            volbase=v.shift(1).rolling(lookback,min_periods=lookback).mean(); active=v>volbase*volume_mult
            long=(close>ceiling)&(prev<=ceiling)&active; short=(close<floor)&(prev>=floor)&active
        else:
            long=(close>ceiling)&(prev<=ceiling); short=(close<floor)&(prev>=floor)
    elif family in {"candle_engulfing","candle_patterns"}:
        po=o.shift(1); pc=close.shift(1)
        bullish=(close>o)&(pc<po)&(o<=pc)&(close>=po)
        bearish=(close<o)&(pc>po)&(o>=pc)&(close<=po)
        body=(close-o).abs().replace(0,np.nan)
        lower_wick=np.minimum(o,close)-l; upper_wick=h-np.maximum(o,close)
        hammer=(lower_wick>=2*body)&(upper_wick<=body)&(close>o)
        shooting_star=(upper_wick>=2*body)&(lower_wick<=body)&(close<o)
        inside=(h<h.shift(1))&(l>l.shift(1))
        inside_break_up=inside.shift(1).fillna(False)&(close>h.shift(2))
        inside_break_down=inside.shift(1).fillna(False)&(close<l.shift(2))
        if family=="candle_engulfing":
            long=bullish; short=bearish
        else:
            long=bullish|hammer|inside_break_up; short=bearish|shooting_star|inside_break_down
    elif family=="classic_patterns":
        pivot_high=(h.shift(2)>h.shift(3))&(h.shift(2)>h.shift(1))
        pivot_low=(l.shift(2)<l.shift(3))&(l.shift(2)<l.shift(1))
        ph=h.shift(2).where(pivot_high); pl=l.shift(2).where(pivot_low)
        prior_high=ph.shift(1).rolling(lookback,min_periods=2).max()
        prior_low=pl.shift(1).rolling(lookback,min_periods=2).min()
        tol=band_std/100.0
        double_top=pivot_high&(ph>=prior_high*(1-tol))&(ph<=prior_high*(1+tol))
        double_bottom=pivot_low&(pl>=prior_low*(1-tol))&(pl<=prior_low*(1+tol))
        long=double_bottom.shift(1).fillna(False)&(close>o)
        short=double_top.shift(1).fillna(False)&(close<o)
    elif family=="stochastic_reversal":
        low_n=l.rolling(lookback,min_periods=lookback).min()
        high_n=h.rolling(lookback,min_periods=lookback).max()
        k=100*(close-low_n)/(high_n-low_n).replace(0,np.nan)
        dline=k.rolling(max(2,rsi_period//3),min_periods=max(2,rsi_period//3)).mean()
        long=(k.shift(1)<rsi_low)&(k>=rsi_low)&(k>dline)
        short=(k.shift(1)>rsi_high)&(k<=rsi_high)&(k<dline)
    elif family=="vwap_reversion":
        typical=(h+l+close)/3
        vol_sum=v.rolling(lookback,min_periods=lookback).sum()
        vwap=(typical*v).rolling(lookback,min_periods=lookback).sum()/vol_sum.replace(0,np.nan)
        sd=close.rolling(lookback,min_periods=lookback).std(ddof=0)
        long=(close<vwap-band_std*sd)&(close>o)
        short=(close>vwap+band_std*sd)&(close<o)
    elif family=="macd_cross":
        macd=ef-es; signal=macd.ewm(span=max(2,rsi_period//2),adjust=False,min_periods=max(2,rsi_period//2)).mean()
        long=(macd>signal)&(macd.shift(1)<=signal.shift(1)); short=(macd<signal)&(macd.shift(1)>=signal.shift(1))
    elif family=="trend_pullback":
        long=(ef>es)&(l<=ef)&(close>ef)&(close.shift(1)<=ef.shift(1))
        short=(ef<es)&(h>=ef)&(close<ef)&(close.shift(1)>=ef.shift(1))
    # Independent duplicate of the causal confirmation/regime layer.
    filt=spec.get("filter_family","none")
    if filt!="none":
        volbase=v.shift(1).rolling(lookback,min_periods=lookback).mean()
        atrbase=atr.shift(1).rolling(lookback,min_periods=lookback).median()
        delta=close.diff()
        gain=delta.clip(lower=0).ewm(alpha=1/rsi_period,adjust=False,min_periods=rsi_period).mean()
        loss=(-delta.clip(upper=0)).ewm(alpha=1/rsi_period,adjust=False,min_periods=rsi_period).mean()
        rsi=100-(100/(1+gain/loss.replace(0,np.nan)))
        mom=close.pct_change(lookback)
        trend_up=ef>es; trend_down=ef<es
        slope_n=max(2,rsi_period//3)
        slope_up=ef>ef.shift(slope_n); slope_down=ef<ef.shift(slope_n)
        vol_active=v>volbase*volume_mult; atr_active=atr>atrbase
        bullish=close>o; bearish=close<o
        high_prev=h.shift(1).rolling(lookback,min_periods=lookback).max()
        low_prev=l.shift(1).rolling(lookback,min_periods=lookback).min()
        if filt=="trend_alignment": allow_l,allow_s=trend_up,trend_down
        elif filt=="volume_expansion": allow_l=allow_s=vol_active
        elif filt=="atr_expansion": allow_l=allow_s=atr_active
        elif filt=="candle_direction": allow_l,allow_s=bullish,bearish
        elif filt=="rsi_bias": allow_l,allow_s=rsi>=50,rsi<=50
        elif filt=="momentum_alignment": allow_l,allow_s=mom>0,mom<0
        elif filt=="ema_slope": allow_l,allow_s=slope_up,slope_down
        elif filt=="range_guard": allow_l=allow_s=(atr/close.abs().replace(0,np.nan)<0.02)
        elif filt=="breakout_confirmation": allow_l,allow_s=close>high_prev,close<low_prev
        elif filt=="volume_and_trend": allow_l,allow_s=vol_active&trend_up,vol_active&trend_down
        elif filt=="trend_and_momentum": allow_l,allow_s=trend_up&(mom>0),trend_down&(mom<0)
        elif filt=="candle_and_volume": allow_l,allow_s=bullish&vol_active,bearish&vol_active
        elif filt=="low_volatility": allow_l=allow_s=(atr/close.abs().replace(0,np.nan)<0.01)
        else: raise ValueError(f"Unsupported filter family: {filt}")
        long=long & allow_l.fillna(False)
        short=short & allow_s.fillna(False)
    rate=(float(cfg["commission_pct_per_side"])+float(cfg["slippage_pct_per_side"]))/100.0
    balance=float(cfg["initial_balance"]); streak=0; ledger=[]; max_streak=0
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
        if risk<=0 or (mode=="percent_equity" and risk>=balance): i+=1; continue
        dist=entry-stop if side=="LONG" else stop-entry
        qty=risk/(dist+entry*rate+stop*rate)
        if cfg["target_mode"]=="gross_rr":
            target=entry+dist*float(cfg["reward_risk"]) if side=="LONG" else entry-dist*float(cfg["reward_risk"])
        else:
            reward=risk*float(cfg["reward_risk"])
            target=(entry*(1+rate)+reward/qty)/(1-rate) if side=="LONG" else (entry*(1-rate)-reward/qty)/(1+rate)
        if target<=0: i+=1; continue
        ex=None; px=None; reason=None
        for j in range(ei,len(d)):
            bar=d.iloc[j]
            sl=(float(bar.low)<=stop) if side=="LONG" else (float(bar.high)>=stop)
            tp=(float(bar.high)>=target) if side=="LONG" else (float(bar.low)<=target)
            if sl and tp: raise ValueError(f"UNRESOLVED_AMBIGUITY row={j}")
            if sl: ex=j; px=stop; reason="SL"; break
            if tp: ex=j; px=target; reason="TP"; break
        if ex is None: break
        gross=(px-entry)*qty*(1 if side=="LONG" else -1)
        ec=entry*qty*rate; xc=px*qty*rate; net=gross-ec-xc; balance+=net
        streak=0 if reason=="TP" else streak+1
        max_streak=max(max_streak,streak)
        ledger.append({"trade_id":len(ledger)+1,"signal_time":int(d.open_time.iloc[i]),"entry_time":int(d.open_time.iloc[ei]),
          "exit_time":int(d.open_time.iloc[ex]),"direction":side,"signal_price":float(close.iloc[i]),"entry_price":entry,
          "stop_loss":stop,"take_profit":target,"exit_price":float(px),"exit_reason":reason,"quantity":qty,"risk_budget":risk,
          "gross_pnl":gross,"entry_cost":ec,"exit_cost":xc,"fees_slippage":ec+xc,"net_pnl":net,
          "balance_after":balance,"loss_streak_after":streak})
        i=ex+1
    return {"ledger":ledger,"max_consecutive_losses":max_streak}
