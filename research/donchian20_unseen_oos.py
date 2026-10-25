"""Frozen Donchian-20 meta-label strategy test on untouched OOS (Aug-Sep 2026).
Rules copied from the original 159-trade in-sample experiment: Donchian-20 close breakout,
HistGradientBoosting meta-label using the original OHLCV features, probability threshold 0.25,
ATR(14)*1.5 stop, fixed $10 net risk and net 3.5R target, next-bar-open entry,
one position at a time, and one-bar cooldown after a loss. Model is fitted only on TRAIN.
OOS is read only for the final test; no threshold/parameter selection on OOS.
"""
import asyncio, json, time, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import research.causal_ml_meta_label_search as base
import research.ema_9_21_123 as sm
from research.run_factory import get_data, split_reference_period
from app.engine import normalize_contract

OUT=ROOT/"research-results"; OUT.mkdir(exist_ok=True)
THRESHOLD=0.25
FAMILY="donchian_break_20"

async def main():
    started=time.monotonic()
    sm.CONTRACT=normalize_contract({"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False})
    full,manifest=await get_data()
    splits=split_reference_period(full)
    train=splits["train"].reset_index(drop=True)
    oos=splits["oos"].reset_index(drop=True)
    _,atr_train=base.get_signals(train)
    train_events=base.events(train,atr_train)
    if len(train_events)<100: raise RuntimeError(f"Not enough Train events: {len(train_events)}")
    train_events=train_events.sort_values("idx").reset_index(drop=True)
    cut=int(len(train_events)*0.8)
    fit=train_events.iloc[:cut].copy()
    # Match the original model input schema: OHLCV/time/regime features only (no family/side fields).
    feats=list(base.FEATS)
    model=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=0.06,l2_regularization=2,random_state=42))
    model.fit(fit[feats],fit.label.astype(int))
    sigs,atr_oos=base.get_signals(oos)
    f=base.features(oos,atr_oos)
    raw_long,raw_short=sigs[FAMILY]
    long_mask=np.zeros(len(oos),dtype=bool); short_mask=np.zeros(len(oos),dtype=bool)
    for side,raw,out in (("LONG",raw_long,long_mask),("SHORT",raw_short,short_mask)):
        ix=np.flatnonzero(pd.Series(raw,index=oos.index).fillna(False).to_numpy())
        ix=ix[(ix>=80)&(ix<len(oos)-1)]
        if len(ix):
            prob=model.predict_proba(f.iloc[ix][feats])[:,1]
            out[ix]=prob>=THRESHOLD
    metrics,ledger=sm.run_combined(oos,pd.Series(long_mask,index=oos.index),pd.Series(short_mask,index=oos.index),atr_oos,"atr_1.5",loss_cooldown_bars=1,cooldown_trigger_losses=1)
    report={
      "schema":"donchian20_frozen_unseen_oos_v1",
      "status":"completed",
      "rules_frozen":True,
      "oos_used_for_selection":False,
      "model_fit_split":"TRAIN Apr-Jun 2026 only; first 80% of chronological Train events",
      "test_split":"OOS Aug-Sep 2026 only",
      "threshold":THRESHOLD,
      "family":FAMILY,
      "stop":"ATR(14) * 1.5",
      "target":"true net 3.5R (+$35 net per full TP), fixed net risk $10",
      "contract":{"initial_balance":1000,"compound":False,"commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"one_position_max":True,"entry":"next candle open","exit":"SL or TP only","loss_cooldown_bars":1,"same_bar_policy":"SL priority"},
      "dataset_manifest":manifest,
      "train_rows":len(train),"train_events":len(train_events),"model_fit_events":len(fit),
      "oos_rows":len(oos),"oos_metrics":metrics,"oos_trade_count":len(ledger),
      "ledger":ledger,
      "elapsed_seconds":round(time.monotonic()-started,2)
    }
    path=OUT/"donchian20_unseen_oos_report.json"
    path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    summary={"report":str(path),"elapsed_seconds":report["elapsed_seconds"],"train_events":len(train_events),"fit_events":len(fit),"oos_rows":len(oos),"threshold":THRESHOLD,"metrics":metrics,"trade_count":len(ledger),"rules_frozen":True,"oos_used_for_selection":False}
    print("DONCHIAN20_UNSEEN_OOS "+json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__": asyncio.run(main())
