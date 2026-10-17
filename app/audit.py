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
    long=long.fillna(False)
    short=short.fillna(False)
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
