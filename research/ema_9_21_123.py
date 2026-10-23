"""Focused EMA 9/21 + post-cross 1-2-3 test. Real Binance USD-M candles; no OOS selection."""
import asyncio, json, math, hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from app.engine import normalize_contract, _size_and_target, _resolve_bar_exit
from app.data import DataIntegrityError
from research.run_factory import get_data, split_reference_period

OUT=Path("research-results"); OUT.mkdir(exist_ok=True)
CONTRACT=normalize_contract({"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","min_win_rate_pct":40,"min_profit_factor":1,"min_tp_per_day":3,"max_consecutive_losses":12,"use_min_win_rate":True,"use_min_profit_factor":True,"use_min_tp_per_day":True,"use_max_consecutive_losses":True,"min_trades_per_day":10,"min_trades_per_month":300,"break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False})
STOPS=("signal_candle","swing_3","swing_5","atr_1.0","atr_1.5","atr_2.0")
ENTRIES=("breakout_123","pullback_reclaim","higher_low_breakout","trend_pullback_reversal","trend_pullback_cross5","trend_pullback_cross10","trend_pullback_cross20","cross_pullback_breakout5","cross_pullback_breakout10","cross_pullback_breakout20","structure_break_after_pullback","range_break_retest","swing_structure_break","squeeze_breakout","swing_failure_reclaim","atr_expansion_breakout")

def signals(df):
    c=df.close.astype(float); o=df.open.astype(float); h=df.high.astype(float); lo=df.low.astype(float)
    e9=c.ewm(span=9,adjust=False,min_periods=9).mean()
    e21=c.ewm(span=21,adjust=False,min_periods=21).mean()
    up=(e9>e21)&(e9.shift(1)<=e21.shift(1))
    dn=(e9<e21)&(e9.shift(1)>=e21.shift(1))
    recent_up=up.rolling(31,min_periods=1).max().shift(1).fillna(0).astype(bool)
    recent_dn=dn.rolling(31,min_periods=1).max().shift(1).fillna(0).astype(bool)
    # Cross bar itself is excluded: setup must form after the cross and within 30 bars.
    up_age=up.astype(int).rolling(31,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age=dn.astype(int).rolling(31,min_periods=1).sum().shift(1).fillna(0)>0
    up_age_5=up.astype(int).rolling(6,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age_5=dn.astype(int).rolling(6,min_periods=1).sum().shift(1).fillna(0)>0
    up_age_10=up.astype(int).rolling(11,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age_10=dn.astype(int).rolling(11,min_periods=1).sum().shift(1).fillna(0)>0
    up_age_20=up.astype(int).rolling(21,min_periods=1).sum().shift(1).fillna(0)>0
    dn_age_20=dn.astype(int).rolling(21,min_periods=1).sum().shift(1).fillna(0)>0
    three_up=(c>c.shift(1))&(c.shift(1)>c.shift(2))&(c.shift(2)>c.shift(3))
    three_dn=(c<c.shift(1))&(c.shift(1)<c.shift(2))&(c.shift(2)<c.shift(3))
    prev3h=h.shift(1).rolling(3,min_periods=3).max()
    prev3l=lo.shift(1).rolling(3,min_periods=3).min()
    result={}
    result["breakout_123"]=(up_age&(c>e21)&three_up&(c>prev3h), dn_age&(c<e21)&three_dn&(c<prev3l))
    result["pullback_reclaim"]=(up_age&(lo<=e21)&(c>e21)&(c>o), dn_age&(h>=e21)&(c<e21)&(c<o))
    result["higher_low_breakout"]=(up_age&(lo.shift(1)<lo.shift(2))&(lo>lo.shift(1))&(c>h.shift(1))&(c>e21),
                                   dn_age&(h.shift(1)>h.shift(2))&(h<h.shift(1))&(c<lo.shift(1))&(c<e21))
    # Additional entry structures: trend pullback/reversal, volatility squeeze breakout,
    # failed-breakout reclaim, and ATR expansion through a prior range boundary.
    touched_up=(lo<=e21).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)
    touched_dn=(h>=e21).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)
    trend_long=(e9>e21)&touched_up&(c>o)&(c>h.shift(1))&(c>e21)
    trend_short=(e9<e21)&touched_dn&(c<o)&(c<lo.shift(1))&(c<e21)
    # New causal EMA-cross -> EMA21 pullback -> local structure breakout variants.
    # The touch must occur before the signal candle; entry remains next candle open.
    break2h=h.shift(1).rolling(2,min_periods=2).max()
    break2l=lo.shift(1).rolling(2,min_periods=2).min()
    result["cross_pullback_breakout5"]=(up_age_5&touched_up&(c>o)&(c>break2h)&(c>e21),
                                          dn_age_5&touched_dn&(c<o)&(c<break2l)&(c<e21))
    result["cross_pullback_breakout10"]=(up_age_10&touched_up&(c>o)&(c>break2h)&(c>e21),
                                           dn_age_10&touched_dn&(c<o)&(c<break2l)&(c<e21))
    result["cross_pullback_breakout20"]=(up_age_20&touched_up&(c>o)&(c>break2h)&(c>e21),
                                           dn_age_20&touched_dn&(c<o)&(c<break2l)&(c<e21))
    result["trend_pullback_reversal"]=(trend_long,trend_short)
    result["trend_pullback_cross5"]=(trend_long&up_age_5,trend_short&dn_age_5)
    result["trend_pullback_cross10"]=(trend_long&up_age_10,trend_short&dn_age_10)
    result["trend_pullback_cross20"]=(trend_long&up_age_20,trend_short&dn_age_20)
    # New market-structure entries: causal swing/range levels are shifted so the
    # current signal candle never participates in its own reference level.
    prior_high=h.shift(1).rolling(20,min_periods=20).max()
    prior_low=lo.shift(1).rolling(20,min_periods=20).min()
    recent_pullback_long=(lo.shift(1).rolling(5,min_periods=5).min()<=e21.shift(1))
    recent_pullback_short=(h.shift(1).rolling(5,min_periods=5).max()>=e21.shift(1))
    result["structure_break_after_pullback"]=((e9>e21)&recent_pullback_long&(c>prior_high)&(c>o),
                                                (e9<e21)&recent_pullback_short&(c<prior_low)&(c<o))
    # Breakout then retest of the prior 10-bar boundary, with a close back
    # through the boundary in the direction of the original break.
    high10=h.shift(1).rolling(10,min_periods=10).max()
    low10=lo.shift(1).rolling(10,min_periods=10).min()
    broke_up=(c.shift(1)>high10.shift(1))
    broke_dn=(c.shift(1)<low10.shift(1))
    result["range_break_retest"]=((broke_up&(lo<=high10)&(c>high10)&(c>o)),
                                  (broke_dn&(h>=low10)&(c<low10)&(c<o)))
    # Confirm a higher-low / lower-high structure before breaking a recent swing.
    low5=lo.shift(1).rolling(5,min_periods=5).min()
    high5=h.shift(1).rolling(5,min_periods=5).max()
    higher_low=(low5>low5.shift(5))
    lower_high=(high5<high5.shift(5))
    result["swing_structure_break"]=((higher_low&(c>high5)&(c>o)),
                                     (lower_high&(c<low5)&(c<o)))
    prior_low=lo.shift(1).rolling(20,min_periods=20).min()
    width=(prior_high-prior_low)/c
    width_rank=width.rolling(240,min_periods=60).rank(pct=True)
    squeeze=width_rank.shift(1)<=0.25
    result["squeeze_breakout"]=(squeeze&(c>prior_high),squeeze&(c<prior_low))
    sweep_low=lo<lo.shift(1).rolling(10,min_periods=10).min()
    sweep_high=h>h.shift(1).rolling(10,min_periods=10).max()
    result["swing_failure_reclaim"]=(sweep_low&(c>lo.shift(1).rolling(10,min_periods=10).min())&(c>o),
                                     sweep_high&(c<h.shift(1).rolling(10,min_periods=10).max())&(c<o))
    atr=pd.concat([h-lo,(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    atr_base=atr.rolling(20,min_periods=20).mean()
    result["atr_expansion_breakout"]=( (atr>atr_base.shift(1)*1.15)&(c>prior_high)&(c>o),
                                        (atr>atr_base.shift(1)*1.15)&(c<prior_low)&(c<o))
    return result,atr

def confirmation_filters(df):
    """Causal directional confirmations; these refine entries without changing risk/exit contract."""
    c=df.close.astype(float); o=df.open.astype(float); h=df.high.astype(float); lo=df.low.astype(float); v=df.volume.astype(float) if "volume" in df.columns else pd.Series(np.nan,index=df.index)
    e21=c.ewm(span=21,adjust=False,min_periods=21).mean()
    tr=pd.concat([h-lo,(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=gain/loss.replace(0,np.nan); rsi=100-(100/(1+rs))
    upmove=h.diff(); downmove=-lo.diff()
    plus_dm=upmove.where((upmove>downmove)&(upmove>0),0.0)
    minus_dm=downmove.where((downmove>upmove)&(downmove>0),0.0)
    plus_di=100*plus_dm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr.replace(0,np.nan)
    minus_di=100*minus_dm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/atr.replace(0,np.nan)
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    adx=dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    volmean=v.rolling(20,min_periods=20).mean()
    atrpct=atr/c
    volrank=atrpct.rolling(240,min_periods=60).rank(pct=True)
    candle_range=(h-lo).replace(0,np.nan)
    body_ratio=(c-o).abs()/candle_range
    close_location=(c-lo)/candle_range
    slope_atr=(e21-e21.shift(5))/atr.replace(0,np.nan)
    not_extended=((c-e21).abs()/atr.replace(0,np.nan))<=1.25
    # Additional causal quality filters: no future bars are used.
    long_body=(body_ratio>=0.50)&(close_location>=0.70)
    short_body=(body_ratio>=0.50)&(close_location<=0.30)
    long_adx=(adx>=22)&(plus_di>minus_di)
    short_adx=(adx>=22)&(minus_di>plus_di)
    long_slope=slope_atr>=0.08
    short_slope=slope_atr<=-0.08
    return {
      "baseline": (pd.Series(True,index=df.index),pd.Series(True,index=df.index)),
      "ema21_slope3": ((e21>e21.shift(3)),(e21<e21.shift(3))),
      "adx18_directional": ((adx>=18)&(plus_di>minus_di),(adx>=18)&(minus_di>plus_di)),
      "rsi_momentum": ((rsi>=50)&(rsi<=72),(rsi<=50)&(rsi>=28)),
      "volume_above_sma20": ((v>volmean),(v>volmean)),
      "atr_midrange": ((volrank>=0.2)&(volrank<=0.8),(volrank>=0.2)&(volrank<=0.8)),
      "slope_plus_adx": ((e21>e21.shift(3))&(adx>=18)&(plus_di>minus_di),(e21<e21.shift(3))&(adx>=18)&(minus_di>plus_di)),
      "strong_candle_close": (long_body,short_body),
      "adx22_directional": (long_adx,short_adx),
      "ema21_slope5_atr": (long_slope,short_slope),
      "rsi_tight_momentum": ((rsi>=52)&(rsi<=68),(rsi<=48)&(rsi>=32)),
      "avoid_extended_entry": (not_extended,not_extended),
      "body_plus_adx22": (long_body&long_adx,short_body&short_adx),
      "slope_adx_candle_stack": (long_slope&long_adx&long_body,short_slope&short_adx&short_body),
      "trend_quality_stack": (long_slope&long_adx&long_body&(rsi>=52)&(rsi<=68)&not_extended,
                              short_slope&short_adx&short_body&(rsi<=48)&(rsi>=32)&not_extended),
    }

def run_one(df, sig, atr, side, stopmode):
    op=df.open.to_numpy(float); hi=df.high.to_numpy(float); lo=df.low.to_numpy(float); cl=df.close.to_numpy(float); ts=df.open_time.to_numpy(np.int64)
    mask=sig.fillna(False).to_numpy(bool); positions=np.flatnonzero(mask); ledger=[]; bal=1000.; streak=0; maxstreak=0; capital=False; unresolved=0
    i0=max(21,14)
    positions=positions[positions>=i0]
    cursor=0
    while cursor<len(positions):
        i=int(positions[cursor]); cursor+=1
        if i+1>=len(df): break
        entry_i=i+1; entry=float(op[entry_i]); side_long=side=="LONG"
        if stopmode=="signal_candle": stop=float(lo[i] if side_long else hi[i])
        elif stopmode=="swing_3": stop=float(np.min(lo[max(0,i-2):i+1]) if side_long else np.max(hi[max(0,i-2):i+1]))
        elif stopmode=="swing_5": stop=float(np.min(lo[max(0,i-4):i+1]) if side_long else np.max(hi[max(0,i-4):i+1]))
        else:
            mult=float(stopmode.split("_")[1]); a=float(atr.iloc[i])
            if not np.isfinite(a) or a<=0: continue
            stop=entry-a*mult if side_long else entry+a*mult
        if (side_long and stop>=entry) or ((not side_long) and stop<=entry): continue
        if bal<=10+1e-9: capital=True; break
        try: qty,tp,er,xr=_size_and_target(entry,stop,side,10.,CONTRACT)
        except ValueError: continue
        reason=price=exit_t=None; exit_i=None
        for j in range(entry_i,len(df)):
            try: resolution=_resolve_bar_exit(side,float(op[j]),float(hi[j]),float(lo[j]),stop,tp)
            except DataIntegrityError:
                # Strict conservative rule: unresolved same-bar SL+TP becomes a stop.
                resolution=("SL",float(stop)); unresolved+=1
            if resolution is not None:
                reason,price=resolution; exit_t=int(ts[j]); exit_i=j; break
        if exit_i is None: break
        gross=(price-entry)*qty*(1 if side_long else -1); ec=entry*qty*er; xc=price*qty*xr; net=gross-ec-xc; bal+=net
        if reason=="TP": streak=0
        else:
            streak+=1; maxstreak=max(maxstreak,streak)
            if loss_cooldown_bars>0 and streak>=cooldown_trigger_losses: cooldown_until=exit_i+loss_cooldown_bars
        ledger.append({"entry_time":int(ts[entry_i]),"exit_time":exit_t,"side":side,"entry":entry,"stop":stop,"target":tp,"exit":price,"reason":reason,"net":net,"costs":ec+xc,"streak":streak})
        cursor=int(np.searchsorted(positions,exit_i+1,side="left"))
    wins=sum(t["reason"]=="TP" and t["net"]>0 for t in ledger); gp=sum(max(0,t["net"]) for t in ledger); gl=abs(sum(min(0,t["net"]) for t in ledger))
    span=max((int(ts[-1])-int(ts[0]))/86400000,1/1440)
    months={}
    for t in ledger:
        month=pd.to_datetime(t["entry_time"],unit="ms",utc=True).strftime("%Y-%m"); months[month]=months.get(month,0)+1
    metrics={"trades":len(ledger),"wins":wins,"win_rate_pct":100*wins/len(ledger) if ledger else 0.,"profit_factor":gp/gl if gl else (None if gp else 0.),"tp_per_day":wins/span,"trades_per_day":len(ledger)/span,"max_consecutive_losses":maxstreak,"net_pnl":sum(t["net"] for t in ledger),"final_balance":bal,"total_costs":sum(t["costs"] for t in ledger),"monthly_trade_counts":months,"capital_exhausted":capital,"unresolved_ohlc_bars_count":unresolved}
    metrics["passes_all_user_targets"]=(metrics["win_rate_pct"]>40 and (metrics["profit_factor"] or 0)>1 and metrics["tp_per_day"]>=3 and metrics["trades_per_day"]>=10 and maxstreak<=12 and not capital and bool(months) and all(v>=300 for v in months.values()))
    return metrics,ledger


def run_combined(df, long_sig, short_sig, atr, stopmode, loss_cooldown_bars=0, cooldown_trigger_losses=2):
    """One shared chronological ledger: LONG and SHORT compete for the same single position."""
    op=df.open.to_numpy(float); hi=df.high.to_numpy(float); lo=df.low.to_numpy(float); ts=df.open_time.to_numpy(np.int64)
    lm=long_sig.fillna(False).to_numpy(bool); sm=short_sig.fillna(False).to_numpy(bool)
    signal_idx=np.flatnonzero(lm | sm); ledger=[]; bal=1000.; streak=0; maxstreak=0; capital=False; unresolved=0; conflicts=0
    i0=21
    signal_idx=signal_idx[signal_idx>=i0]
    cursor=0; cooldown_until=-1
    while cursor<len(signal_idx):
        i=int(signal_idx[cursor]); cursor+=1
        if i <= cooldown_until: continue
        is_long=bool(lm[i]); is_short=bool(sm[i])
        if is_long and is_short:
            conflicts+=1
            continue
        side="LONG" if is_long else "SHORT"; side_long=side=="LONG"
        if i+1>=len(df): break
        entry_i=i+1; entry=float(op[entry_i])
        if stopmode=="signal_candle": stop=float(lo[i] if side_long else hi[i])
        elif stopmode=="swing_3": stop=float(np.min(lo[max(0,i-2):i+1]) if side_long else np.max(hi[max(0,i-2):i+1]))
        elif stopmode=="swing_5": stop=float(np.min(lo[max(0,i-4):i+1]) if side_long else np.max(hi[max(0,i-4):i+1]))
        else:
            mult=float(stopmode.split("_")[1]); a=float(atr.iloc[i])
            if not np.isfinite(a) or a<=0: continue
            stop=entry-a*mult if side_long else entry+a*mult
        if (side_long and stop>=entry) or ((not side_long) and stop<=entry): continue
        if bal<=10+1e-9: capital=True; break
        try: qty,tp,er,xr=_size_and_target(entry,stop,side,10.,CONTRACT)
        except ValueError: continue
        reason=price=exit_t=None; exit_i=None
        for j in range(entry_i,len(df)):
            try: resolution=_resolve_bar_exit(side,float(op[j]),float(hi[j]),float(lo[j]),stop,tp)
            except DataIntegrityError:
                resolution=("SL",float(stop)); unresolved+=1
            if resolution is not None:
                reason,price=resolution; exit_t=int(ts[j]); exit_i=j; break
        if exit_i is None: break
        gross=(price-entry)*qty*(1 if side_long else -1); ec=entry*qty*er; xc=price*qty*xr; net=gross-ec-xc; bal+=net
        if reason=="TP":
            streak=0
        else:
            streak+=1
            maxstreak=max(maxstreak,streak)
            if loss_cooldown_bars>0 and streak>=cooldown_trigger_losses:
                cooldown_until=max(cooldown_until,exit_i+loss_cooldown_bars)
        ledger.append({"entry_time":int(ts[entry_i]),"signal_time":int(ts[i]),"exit_time":exit_t,"side":side,"entry":entry,"stop":stop,"target":tp,"exit":price,"reason":reason,"net":net,"costs":ec+xc,"streak":streak})
        cursor=int(np.searchsorted(signal_idx,exit_i+1,side="left"))
    wins=sum(t["reason"]=="TP" and t["net"]>0 for t in ledger); gp=sum(max(0,t["net"]) for t in ledger); gl=abs(sum(min(0,t["net"]) for t in ledger))
    span=max((int(ts[-1])-int(ts[0]))/86400000,1/1440)
    months={}
    for t in ledger:
        month=pd.to_datetime(t["entry_time"],unit="ms",utc=True).strftime("%Y-%m"); months[month]=months.get(month,0)+1
    metrics={"trades":len(ledger),"long_trades":sum(t["side"]=="LONG" for t in ledger),"short_trades":sum(t["side"]=="SHORT" for t in ledger),
      "wins":wins,"win_rate_pct":100*wins/len(ledger) if ledger else 0.,"profit_factor":gp/gl if gl else (None if gp else 0.),
      "tp_per_day":wins/span,"trades_per_day":len(ledger)/span,"max_consecutive_losses":maxstreak,"net_pnl":sum(t["net"] for t in ledger),
      "final_balance":bal,"total_costs":sum(t["costs"] for t in ledger),"monthly_trade_counts":months,"capital_exhausted":capital,
      "unresolved_ohlc_bars_count":unresolved,"simultaneous_signal_conflicts_skipped":conflicts,"loss_cooldown_bars":loss_cooldown_bars,"cooldown_trigger_losses":cooldown_trigger_losses}
    metrics["passes_all_user_targets"]=(metrics["win_rate_pct"]>40 and (metrics["profit_factor"] or 0)>1 and metrics["tp_per_day"]>=3 and metrics["trades_per_day"]>=10 and maxstreak<=12 and not capital and bool(months) and all(v>=300 for v in months.values()))
    return metrics,ledger

async def main():
    df,manifest=await get_data()
    splits=split_reference_period(df)
    results=[]
    for split in ("train","development"):
        part=splits[split]; sigs,atr=signals(part); filters=confirmation_filters(part)
        for ent in ENTRIES:
            raw_long,raw_short=sigs[ent]
            if ent.startswith("trend_pullback_cross") or ent.startswith("cross_pullback_breakout") or ent in ("structure_break_after_pullback","range_break_retest","swing_structure_break"):
                filter_names=("baseline","volume_above_sma20","rsi_tight_momentum","adx22_directional",
                              "strong_candle_close","avoid_extended_entry","slope_plus_adx","trend_quality_stack","slope_adx_candle_stack")
                active_filters=((name,filters[name]) for name in filter_names)
            else:
                active_filters=filters.items()
            for filter_name,(long_filter,short_filter) in active_filters:
                long_sig=raw_long & long_filter
                short_sig=raw_short & short_filter
                for stop in STOPS:
                    # Targeted cooldown sweep only on the promising structure-break family:
                    # vary trigger loss streak and wait bars, while preserving all trade rules.
                    if ent == "structure_break_after_pullback" and stop == "swing_5":
                        # Follow-up focused on the prior 196-trade / max-loss-streak-11 candidate.
                        # Expand causal entry-quality filters while preserving the shared LONG+SHORT
                        # ledger, fixed net -$10/+35 exits, and the exact user's gates.
                        cooldown_configs=[(0,0)]+[(bars,trigger) for trigger in (1,2,3) for bars in (1,2,3,5,8,13,20,30)]
                    else:
                        cooldown_configs=[(bars,2) for bars in ((0,5,10,20) if ent in ("structure_break_after_pullback","swing_failure_reclaim","range_break_retest") else (0,))]
                    for cooldown,trigger_losses in cooldown_configs:
                        m,ledger=run_combined(part,long_sig,short_sig,atr,stop,loss_cooldown_bars=cooldown,cooldown_trigger_losses=trigger_losses)
                        results.append({"split":split,"entry_method":ent,"confirmation_filter":filter_name,"side":"LONG+SHORT_SHARED_SINGLE_POSITION","stop_method":stop,"metrics":m,
                                        "ledger_sha256":hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(",",":")).encode()).hexdigest()})
                        print("EMA123_COMBINED_RESULT "+json.dumps(results[-1],sort_keys=True),flush=True)
    def rank(x):
        m=x["metrics"]
        # No target relaxed; rank diagnostic misses by closest overall gates, then PnL.
        gaps=sum([max(0,40.0001-m["win_rate_pct"])/40, max(0,1.0001-(m["profit_factor"] or 0)),max(0,3-m["tp_per_day"])/3,max(0,10-m["trades_per_day"])/10,max(0,m["max_consecutive_losses"]-12)/12,1 if m["capital_exhausted"] else 0])
        return (not m["passes_all_user_targets"],gaps,-m["net_pnl"],-m["trades"])
    output={"schema":"ema9_21_post_cross_123_filtered_shared_single_position_v3","data":manifest,"data_scope":"Binance USD-M ETHUSDT 1m Apr-Sep 2026; OOS not evaluated","split_rows":{"train":len(splits["train"]),"development":len(splits["development"]),"oos":len(splits["oos"])},"fixed_contract":CONTRACT,"entry_methods":list(ENTRIES),"confirmation_filters":list(confirmation_filters(splits["development"]).keys()),"stop_methods":list(STOPS),"loss_cooldown_variants":{"legacy":{"trigger_after_consecutive_losses":2,"bars":[0,5,10,20]},"targeted_structure_break_sweep":{"triggers":[1,2,3],"bars":[1,2,3,5,8,13,20,30],"stops":["swing_5"],"filters":"all causal confirmation filters for structure_break_after_pullback"}},"execution_mode":"shared chronological LONG+SHORT ledger; one position across both sides","result_count":len(results),"validated_count":sum(x["metrics"]["passes_all_user_targets"] for x in results),"top_by_split":{},"results":results}
    for split in ("train","development"):
        subset=sorted([x for x in results if x["split"]==split],key=rank)
        output["top_by_split"][split]=subset[:10]
    (OUT/"ema_9_21_123_report.json").write_text(json.dumps(output,indent=2,sort_keys=True))
    print("EMA123_FINAL "+json.dumps({"report":"research-results/ema_9_21_123_report.json","result_count":len(results),"validated_count":output["validated_count"],"top_by_split":{k:[{"entry_method":x["entry_method"],"confirmation_filter":x.get("confirmation_filter","baseline"),"side":x["side"],"stop_method":x["stop_method"],"metrics":x["metrics"]} for x in v[:5]] for k,v in output["top_by_split"].items()}},sort_keys=True),flush=True)

if __name__=="__main__": asyncio.run(main())
