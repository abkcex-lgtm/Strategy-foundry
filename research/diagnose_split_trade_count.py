"""Fast same-configuration train/development diagnostic for the 360-vs-5 trade discrepancy."""
import asyncio, json
import research.ema_9_21_123 as sm
from app.engine import normalize_contract
from research.run_factory import get_data, split_reference_period

CFG={"symbol":"ETHUSDT","initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,
"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","break_even_enabled":False,
"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False}
async def main():
    sm.CONTRACT=normalize_contract(CFG)
    df, manifest = await get_data()
    splits=split_reference_period(df)
    report={"data_manifest":manifest,"oos_evaluated":False,"configs":{}}
    configs=[
      ("dev_360_candidate","structure_break_pb8_br10","baseline","rsi_tight_momentum","swing_5",0,2),
      ("11_loss_baseline","structure_break_after_pullback","ema21_slope5_atr","ema21_slope5_atr","swing_5",1,1),
    ]
    for split in ("train","development"):
        part=splits[split]
        sigs,atr=sm.signals(part); filters=sm.confirmation_filters(part)
        ts=part.open_time
        split_info={"bars":len(part),"first_utc":str(__import__("pandas").to_datetime(int(ts.iloc[0]),unit="ms",utc=True)),
                    "last_utc":str(__import__("pandas").to_datetime(int(ts.iloc[-1]),unit="ms",utc=True)),"configs":{}}
        for name,entry,lf,sf,stop,bars,trig in configs:
            raw_l,raw_s=sigs[entry]
            lfilter=filters[lf][0].reindex(part.index).fillna(False)
            sfilter=filters[sf][1].reindex(part.index).fillna(False)
            long_sig=raw_l & lfilter; short_sig=raw_s & sfilter
            metrics,ledger=sm.run_combined(part,long_sig,short_sig,atr,stop,loss_cooldown_bars=bars,cooldown_trigger_losses=trig)
            split_info["configs"][name]={"entry":entry,"long_filter":lf,"short_filter":sf,"stop":stop,
              "cooldown_bars":bars,"cooldown_trigger_losses":trig,
              "raw_long_signals":int(raw_l.fillna(False).sum()),"raw_short_signals":int(raw_s.fillna(False).sum()),
              "filtered_long_signals":int(long_sig.fillna(False).sum()),"filtered_short_signals":int(short_sig.fillna(False).sum()),
              "overlapping_filtered_signals":int((long_sig.fillna(False)&short_sig.fillna(False)).sum()),
              "metrics":metrics,"first_trade_utc":str(__import__("pandas").to_datetime(ledger[0]["entry_time"],unit="ms",utc=True)) if ledger else None,
              "last_trade_utc":str(__import__("pandas").to_datetime(ledger[-1]["entry_time"],unit="ms",utc=True)) if ledger else None}
        report["configs"][split]=split_info
        print("SAME_CONFIG_DIAGNOSTIC "+json.dumps({"split":split,**split_info},sort_keys=True),flush=True)
    from pathlib import Path
    p=Path("research-results/same_config_trade_count_diagnostic.json"); p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(report,indent=2,sort_keys=True,default=str))
    print("DIAGNOSTIC_FINAL "+json.dumps({"report":str(p),"oos_evaluated":False},sort_keys=True),flush=True)
if __name__=="__main__": asyncio.run(main())
