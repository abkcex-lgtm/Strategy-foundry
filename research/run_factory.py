from __future__ import annotations
import asyncio, hashlib, json, math, os, sys, time, zipfile, gzip, subprocess
from collections import Counter
from statistics import median
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from app.data import download_binance_month, validate_ohlcv, split_reference_period
from app.engine import search_candidates, backtest, normalize_contract, prepare_backtest_data, _candidate_distance, _capability_evidence, _goal_activity_evidence, _first_exit_candidate
from app.audit import independent_replay, FIELDS, open_position_matches
SYMBOL="ETHUSDT"; YEAR=2026; MONTHS=[4,5,6,7,8,9]
EXPECTED={4:43200,5:44640,6:43200,7:44640,8:44640,9:43200}
REQUESTED=int(os.environ.get("FOUNDRY_CANDIDATES","500000")); SEED=int(os.environ.get("FOUNDRY_SEED","42"))
ALGORITHM=os.environ.get("FOUNDRY_ALGORITHM","surrogate_genetic")
MAX_SECONDS=int(os.environ.get("FOUNDRY_MAX_RUNTIME_SECONDS","19500"))
# Owner-requested quick-screen lane is selected by the workflow commit tag; retain the same data, execution model, and acceptance gates.
# Use nearly the full 8-minute workflow search allowance while reserving time
# for mandatory Development/OOS audits, independent ledger replay, and exports.
# Larger batches avoid repeating per-batch replay/audit overhead every 250 genomes.
if ALGORITHM == "stratified_random": MAX_SECONDS=min(MAX_SECONDS,720)
OUT=ROOT/"research-results"; RUN=ROOT/"research-run"; OUT.mkdir(parents=True,exist_ok=True); RUN.mkdir(parents=True,exist_ok=True)
PROGRESS=OUT/"progress.json"
def digest(b): return hashlib.sha256(b).hexdigest()
def frame_hash(df): return digest(df.to_csv(index=False,float_format="%.12g").encode())
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8"); tmp.replace(path)
def ledger_match(primary,other):
    a=primary["ledger"]; b=other["ledger"]
    if len(a)!=len(b): return False,{"primary_trades":len(a),"independent_trades":len(b),"field":"trade_count"}
    for i,(x,y) in enumerate(zip(a,b),1):
        for field in FIELDS:
            p=x.get(field); q=y.get(field)
            if isinstance(p,(int,float)) and not isinstance(p,bool):
                if not math.isclose(float(p),float(q),rel_tol=1e-10,abs_tol=1e-8): return False,{"trade_id":i,"field":field,"primary":p,"independent":q}
            elif p!=q: return False,{"trade_id":i,"field":field,"primary":p,"independent":q}
    if not open_position_matches(primary,other): return False,{"mismatch":"open_position","primary":primary.get("open_position"),"independent":other.get("open_position")}
    metrics=primary.get("metrics") or {}; contract=primary.get("contract") or {}
    net_values=[float(t.get("net_pnl",0.0)) for t in b]
    gross_profit=sum(max(0.0,v) for v in net_values); gross_loss=abs(sum(min(0.0,v) for v in net_values))
    pf=(gross_profit/gross_loss) if gross_loss else (None if gross_profit else 0.0)
    wins=sum(t.get("exit_reason")=="TP" and float(t.get("net_pnl",0.0))>0 for t in b)
    losses=sum(v<=0 for v in net_values); net_pnl=sum(net_values)
    span_days=float(metrics.get("span_days",0.0) or 0.0)
    independent_streak=max([int(t.get("loss_streak_after",0)) for t in b]+[0])
    other_streak=int(other.get("max_consecutive_losses",independent_streak))
    open_pos=other.get("open_position")
    expected_metrics={
        "trades":len(b),"wins":wins,"losses":losses,
        "win_rate_pct":100.0*wins/len(b) if b else 0.0,
        "profit_factor":pf,"avg_net_tp_per_day":wins/span_days if span_days>0 else 0.0,
        "max_consecutive_losses":other_streak,"net_pnl":net_pnl,
        "final_balance":float(contract.get("initial_balance",0.0))+net_pnl,
        "total_costs":sum(float(t.get("fees_slippage",0.0)) for t in b),
        "unresolved_as_loss_count":sum(bool(t.get("execution_ambiguity",False)) for t in b),
        "open_positions_at_end":1 if open_pos else 0,
        "open_position_unrealized_net_pnl":float(open_pos.get("unrealized_net_pnl",0.0)) if open_pos else 0.0,
    }
    for field,expected in expected_metrics.items():
        actual=metrics.get(field)
        if expected is None:
            if actual is not None: return False,{"mismatch":"aggregate_metric","field":field,"primary":actual,"independent":None}
        elif actual is None or not math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=1e-8):
            return False,{"mismatch":"aggregate_metric","field":field,"primary":actual,"independent":expected}
    if int(metrics.get("max_consecutive_losses",0))!=independent_streak:
        return False,{"mismatch":"ledger_loss_streak","metric":metrics.get("max_consecutive_losses"),"ledger_max":independent_streak}
    return True,{"primary_trades":len(a),"independent_trades":len(b),"ledger_fields_checked":len(FIELDS),"open_position_checked":True,"aggregate_metrics_checked":list(expected_metrics)}
def brief(m):
    keys=("trades","wins","losses","win_rate_pct","profit_factor","avg_net_tp_per_day","max_consecutive_losses","net_pnl","final_balance","total_costs","passes_all_targets","capital_exhausted","monthly_trade_counts","trades_per_day","data_hash_sha256","execution_data_hash_sha256","execution_resolution","unresolved_as_loss_count","open_positions_at_end","open_position_unrealized_net_pnl","span_days","rows")
    return {k:m.get(k) for k in keys}
def capability_pilot_warning(summary, requested):
    """Return an advisory only; a small pilot must never stop the full search."""
    summary = summary or {}
    capability = int(summary.get("capability_evidence_candidates", 0) or 0)
    if capability == 0 and int(requested) >= 250:
        return ("The initial 250-candidate pilot found no candidate with >=0.5 net TP/day, "
                ">=50 trades, and either >=25% win rate or PF >=0.5. This is an advisory "
                "only: continue the requested search because the pilot cannot prove no strategy exists.")
    return None

def activity_gap(metrics):
    """Rank near-misses by actual daily activity shortfall, not lucky PnL."""
    rate=float(metrics.get("trades_per_day",0.0) or 0.0)
    if rate<=0.0:
        span=max(float(metrics.get("span_days",0.0) or 0.0),0.0)
        rate=float(metrics.get("trades",0) or 0)/span if span>0 else 0.0
    return max(0.0,8.0-rate)/8.0

def near_key(x):
    m=x.get("development_metrics") or x.get("metrics") or {}
    train_m=x.get("train_metrics") or x.get("metrics") or {}
    cfg=normalize_contract(x.get("contract") or {})
    dev_net=float(m.get("net_pnl",0.0) or 0.0)
    dev_final=float(m.get("final_balance",cfg["initial_balance"]+dev_net) or 0.0)
    train_net=float(train_m.get("net_pnl",0.0) or 0.0)
    train_final=float(train_m.get("final_balance",cfg["initial_balance"]+train_net) or 0.0)
    risk=float(cfg["risk_usd"] if cfg["risk_mode"]=="fixed_usd" else cfg["initial_balance"]*cfg["risk_pct"]/100.0)
    # Near-miss ordering is evidence-first: audited ledgers, no exhausted capital,
    # daily activity, then target distance. Profitability cannot let a sparse
    # lucky sample outrank a strategy that actually approaches the activity contract.
    return (0 if x.get("development_audit_status")=="MATCH" else 1,
            bool(m.get("capital_exhausted",False)) or bool(train_m.get("capital_exhausted",False)),
            activity_gap(m)+activity_gap(train_m),
            float(x.get("development_distance",x.get("distance",math.inf))),
            dev_final <= risk + 1e-9, train_final <= risk + 1e-9,
            dev_net <= 0.0, train_net <= 0.0,
            -min(dev_net,train_net),
            -min(int(m.get("trades",0)),int(train_m.get("trades",0))))
def trade_diagnostics(ledger, span_days):
    """Compact diagnostics derived from closed trades; ledger remains the source of truth."""
    rows=list(ledger or [])
    span_days=max(float(span_days or 0.0),1.0/1440.0)
    reasons=Counter(str(t.get("exit_reason","UNKNOWN")) for t in rows)
    streak_lengths=[]; current_streak=0
    for trade in rows:
        if trade.get("exit_reason")=="TP" and float(trade.get("net_pnl",0.0))>0:
            if current_streak: streak_lengths.append(current_streak)
            current_streak=0
        else:
            current_streak+=1
    if current_streak: streak_lengths.append(current_streak)
    durations=sorted(max(0.0,(int(t.get("exit_time",0))-int(t.get("entry_time",0)))/60000.0) for t in rows)
    def side_summary(side_rows):
        values=[float(t.get("net_pnl",0.0)) for t in side_rows]
        wins=sum(t.get("exit_reason")=="TP" and float(t.get("net_pnl",0.0))>0 for t in side_rows)
        gp=sum(max(0.0,v) for v in values); gl=abs(sum(min(0.0,v) for v in values))
        pf=gp/gl if gl else (None if gp else 0.0)
        return {"trades":len(side_rows),"wins":wins,"losses":sum(v<=0 for v in values),
                "win_rate_pct":100.0*wins/len(side_rows) if side_rows else 0.0,
                "profit_factor":pf,"net_pnl":sum(values)}
    loss_hist=Counter(streak_lengths)
    wins=sum(t.get("exit_reason")=="TP" and float(t.get("net_pnl",0.0))>0 for t in rows)
    return {"closed_trades":len(rows),"exit_reason_counts":dict(sorted(reasons.items())),
            "direction_breakdown":{side:side_summary([t for t in rows if t.get("direction")==side]) for side in ("LONG","SHORT")},
            "wins":wins,"losses":sum(float(t.get("net_pnl",0.0))<=0 for t in rows),
            "net_pnl":sum(float(t.get("net_pnl",0.0)) for t in rows),
            "total_costs":sum(float(t.get("fees_slippage",0.0)) for t in rows),
            "avg_tp_per_day":wins/span_days,"max_consecutive_losses":max(streak_lengths,default=0),
            "loss_streak_run_count":len(streak_lengths),
            "loss_streak_length_histogram":{str(k):v for k,v in sorted(loss_hist.items())},
            "holding_minutes":{"mean":sum(durations)/len(durations) if durations else 0.0,
                                "median":median(durations) if durations else 0.0,
                                "p95":durations[min(len(durations)-1,int(math.ceil(0.95*len(durations))-1))] if durations else 0.0,
                                "max":max(durations,default=0.0)},
            "trade_ledger_sha256":digest(json.dumps(rows,sort_keys=True,separators=(",",":"),allow_nan=False).encode())}

def benchmark_exit_scan(frame, sample_count=512):
    """Compare the bounded-memory scanner with the pre-optimization reference on TRAIN OHLCV."""
    opens=frame["open"].to_numpy(dtype=float)
    highs=frame["high"].to_numpy(dtype=float)
    lows=frame["low"].to_numpy(dtype=float)
    closes=frame["close"].to_numpy(dtype=float)
    if len(frame)<10:
        raise RuntimeError("Not enough TRAIN rows for exit-scan benchmark")
    starts=np.linspace(0,len(frame)-2,min(int(sample_count),len(frame)-1),dtype=int)
    cases=[]
    for k,start_idx in enumerate(starts):
        anchor=float(closes[start_idx])
        # Benchmark-only levels anchored to observed TRAIN prices; no trade metrics are produced.
        distance=max(anchor*0.0008,0.01)
        side="LONG" if k%2==0 else "SHORT"
        stop,target=(anchor-distance,anchor+distance) if side=="LONG" else (anchor+distance,anchor-distance)
        cases.append((side,int(start_idx),stop,target))
    def reference_scan(side,start_idx,stop,target):
        op=opens[start_idx:]; hi=highs[start_idx:]; lo=lows[start_idx:]
        if side=="LONG":
            touched=(op<=stop)|(op>=target)|(lo<=stop)|(hi>=target)
        else:
            touched=(op>=stop)|(op<=target)|(hi>=stop)|(lo<=target)
        hits=np.flatnonzero(touched)
        return None if hits.size==0 else start_idx+int(hits[0])
    # Keep call order explicit so both implementations receive identical arguments.
    started=time.perf_counter()
    expected=[reference_scan(*case) for case in cases]
    reference_seconds=time.perf_counter()-started
    started=time.perf_counter()
    actual=[_first_exit_candidate(case[0],opens,highs,lows,case[1],case[2],case[3]) for case in cases]
    chunked_seconds=time.perf_counter()-started
    if actual!=expected:
        first=next(i for i,(left,right) in enumerate(zip(expected,actual)) if left!=right)
        raise RuntimeError(f"Chunked exit scan mismatch at benchmark case {first}: reference={expected[first]} optimized={actual[first]}")
    return {
        "status":"PASS",
        "dataset_scope":"TRAIN only",
        "train_rows":len(frame),
        "train_sha256":frame_hash(frame),
        "cases":len(cases),
        "reference_full_tail_seconds":round(reference_seconds,6),
        "chunked_scan_seconds":round(chunked_seconds,6),
        "measured_speedup":round(reference_seconds/max(chunked_seconds,1e-9),3),
        "first_hit_results_match":True,
        "chunk_size":4096,
        "note":"Component microbenchmark on real TRAIN OHLCV; not a full-candidate throughput benchmark.",
    }


def append_unique(dst,src):
    keys={json.dumps(x.get("strategy",{}),sort_keys=True,separators=(",",":")) for x in dst}
    for x in src:
        k=json.dumps(x.get("strategy",{}),sort_keys=True,separators=(",",":"))
        if k not in keys: dst.append(x); keys.add(k)
async def get_data():
    chunks=[]; archives=[]; counts={}
    for month in MONTHS:
        print(f"DATA_START {YEAR}-{month:02d}",flush=True)
        part,h,url=await download_binance_month(SYMBOL,"1m",YEAR,month)
        if len(part)!=EXPECTED[month]: raise RuntimeError(f"Expected {EXPECTED[month]} candles for {YEAR}-{month:02d}, got {len(part)}")
        validate_ohlcv(part,60000,strict_gaps=True); chunks.append(part); counts[f"{YEAR}-{month:02d}"]=len(part)
        archives.append({"month":f"{YEAR}-{month:02d}","url":url,"archive_sha256":h,"rows":len(part)})
        print(f"DATA_OK {YEAR}-{month:02d} rows={len(part)} archive_sha256={h}",flush=True)
    df=pd.concat(chunks,ignore_index=True).sort_values("open_time").reset_index(drop=True)
    integrity=validate_ohlcv(df,60000,strict_gaps=True)
    if len(df)!=263520: raise RuntimeError(f"Expected 263520 total candles, got {len(df)}")
    first=pd.Timestamp("2026-04-01T00:00:00Z").value//1_000_000
    last=pd.Timestamp("2026-09-30T23:59:00Z").value//1_000_000
    if integrity["first_open_time"]!=first or integrity["last_open_time"]!=last: raise RuntimeError("Reference dataset endpoints do not match Apr-Sep 2026")
    csv=RUN/"ETHUSDT-1m-202604-202609.csv"; df.to_csv(csv,index=False,float_format="%.12g")
    with open(csv,"rb") as src,gzip.open(str(csv)+".gz","wb",compresslevel=6) as dst:
        while True:
            chunk=src.read(1024*1024)
            if not chunk: break
            dst.write(chunk)
    return df,{"source":"Binance USD-M Vision monthly archives","symbol":SYMBOL,"interval":"1m","month_rows":counts,"monthly_archives":archives,"validation":integrity,"dataset_sha256":frame_hash(df),"csv_sha256":digest(csv.read_bytes())}
def run(df,data_manifest):
    splits=split_reference_period(df); train,dev,oos=splits["train"],splits["development"],splits["oos"]
    if (len(train),len(dev),len(oos))!=(131040,44640,87840): raise RuntimeError(f"Unexpected split sizes {len(train)}/{len(dev)}/{len(oos)}")
    train_hash=frame_hash(train); dev_hash=frame_hash(dev)
    prepared_train=prepare_backtest_data(train)
    speed_benchmark=benchmark_exit_scan(train)
    print("EXIT_SCAN_BENCHMARK "+json.dumps(speed_benchmark,sort_keys=True),flush=True)
    prepared_dev=prepare_backtest_data(dev)
    train_backtest_hash=prepared_train.data_hash
    dev_backtest_hash=prepared_dev.data_hash
    cfg=normalize_contract({"symbol":SYMBOL,"initial_balance":1000,"risk_mode":"fixed_usd","risk_usd":10,"compound":False,"reward_risk":3.5,"target_mode":"net_rr","commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"max_positions":1,"entry_model":"next_candle_open","exit_model":"sl_tp_only","min_win_rate_pct":40,"min_profit_factor":1,"min_tp_per_day":3,"max_consecutive_losses":12,"use_min_win_rate":True,"use_min_profit_factor":True,"use_min_tp_per_day":True,"use_max_consecutive_losses":True,"min_trades_per_day":8,"min_trades_per_month":300,"break_even_enabled":False,"trailing_stop_enabled":False,"partial_exit_enabled":False,"time_exit_enabled":False})
    start=time.monotonic(); seen=set(); train_pass=[]; train_near=[]; failed=generations=fits=0; failure_reasons={}; family_counts={}; completed=batches=zero=0
    preflight_summary=None; preflight_gate_warning=None
    max_batches=max(5,math.ceil(REQUESTED/1000)*5)
    print(f"SEARCH_START requested={REQUESTED} seed={SEED} algorithm={ALGORITHM} train_rows={len(train)} train_sha256={train_hash}",flush=True)
    while completed<REQUESTED and batches<max_batches and time.monotonic()-start<MAX_SECONDS:
        # Start with a small representative pilot; later batches learn from
        # prior audited candidates instead of restarting an isolated population.
        remaining_seconds=MAX_SECONDS-(time.monotonic()-start)
        if remaining_seconds < 60:
            break
        # Candidate batches are not pre-emptible. Shrink the final batch using
        # observed throughput and reserve time for Development/OOS audits/exports.
        measured_rate=(completed/max(time.monotonic()-start,1.0)) if completed else 2.5
        adaptive_cap=max(32,min(1000,int((remaining_seconds-60)*measured_rate*0.75)))
        batch_cap=adaptive_cap
        size=min(batch_cap,REQUESTED-completed); seed=SEED+batches*1000003
        def train_survival_key(x):
            m=x.get("metrics") or {}
            net=float(m.get("net_pnl",0.0) or 0.0)
            final=float(m.get("final_balance",cfg["initial_balance"]+net) or 0.0)
            risk=float(cfg["risk_usd"] if cfg["risk_mode"]=="fixed_usd" else cfg["initial_balance"]*cfg["risk_pct"]/100.0)
            # TRAIN-only selection: prioritize positive after-cost PnL and survival,
            # then activity evidence and target distance. Never inspect OOS here.
            return (not bool(m.get("passes_all_targets",False)),
                    final <= risk + 1e-9,
                    activity_gap(m),
                    float(x.get("distance",math.inf)),
                    net <= 0.0, -net,
                    -int(m.get("trades",0)))
        prior_pool=sorted(train_pass+train_near,key=train_survival_key)[:192]
        batch_started=time.monotonic()
        def heartbeat(batch_state):
            # Live observability: publish frequent log heartbeats while a batch is
            # still evaluating, not only after the whole batch returns.
            local_done=int(batch_state.get("evaluated_candidates",0))
            global_done=completed+local_done
            state={
                "status":"searching",
                "requested_candidates":REQUESTED,
                "evaluated_unique_candidates":global_done,
                "completed_batches":batches,
                "current_batch":batches+1,
                "current_batch_target":size,
                "current_batch_evaluated":local_done,
                "failed_candidate_evaluations":failed+int(batch_state.get("failed_candidates",0)),
                "candidate_failure_reason_counts":batch_state.get("error_reason_counts",{}),
                "duplicate_genomes_skipped":int(batch_state.get("duplicate_genomes_skipped",0)),
                "elapsed_seconds":round(time.monotonic()-start,2),
                "batch_elapsed_seconds":round(time.monotonic()-batch_started,2),
                "candidates_per_second":round(local_done/max(time.monotonic()-batch_started,1e-9),3),
                "seed":SEED,
                "algorithm":ALGORITHM,
                "train_sha256":train_hash,
                "development_sha256":dev_hash,
                "family_evaluated_counts":{family:family_counts.get(family,0)+int(batch_state.get("family_evaluated_counts",{}).get(family,0)) for family in set(family_counts)|set(batch_state.get("family_evaluated_counts",{}))},
                "heartbeat":"candidate-evaluation",
            }
            write(PROGRESS,state)
            print("SEARCH_HEARTBEAT "+json.dumps(state,sort_keys=True),flush=True)
        result=search_candidates(train,size,seed,cfg,ALGORITHM,progress_callback=heartbeat,
                                 excluded_genome_hashes=seen,prior_candidates=prior_pool,
                                 prepared_context=prepared_train)
        hashes=result.pop("evaluated_genome_hashes",[]); actual=len(hashes); seen.update(hashes); completed+=actual; batches+=1
        failed+=int(result.get("failed_candidates",0)); generations+=int(result.get("generations",0)); fits+=int(result.get("surrogate_model_fits",0))
        for reason,count in result.get("error_reason_counts",{}).items(): failure_reasons[reason]=failure_reasons.get(reason,0)+int(count)
        for fam,n in result.get("family_evaluated_counts",{}).items(): family_counts[fam]=family_counts.get(fam,0)+int(n)
        if result.get("all_passing"): append_unique(train_pass,result.get("results",[]))
        else:
            append_unique(train_near,result.get("results",[]))
            # Rank evidence-eligible/high-activity candidates before lucky tiny samples.
            train_near=sorted(train_near,key=lambda x:(
                not bool((x.get("metrics") or {}).get("passes_all_targets",False)),
                bool((x.get("metrics") or {}).get("capital_exhausted",False)),
                activity_gap(x.get("metrics",{})),
                float(x.get("distance",math.inf)),
                float((x.get("metrics") or {}).get("final_balance",0.0) or 0.0) <= (
                    float(cfg["risk_usd"] if cfg["risk_mode"]=="fixed_usd" else cfg["initial_balance"]*cfg["risk_pct"]/100.0) + 1e-9),
                float((x.get("metrics") or {}).get("net_pnl",0.0) or 0.0) <= 0.0,
                -int((x.get("metrics") or {}).get("trades",0)),
                int(x.get("metrics",{}).get("max_consecutive_losses",10**9))))[:max(5,min(500,batches*5))]
        zero=zero+1 if actual==0 else 0
        if batches==1:
            preflight_summary=result.get("preflight_summary",{})
            promising=int(preflight_summary.get("promising_candidates",0))
            capability=int(preflight_summary.get("capability_evidence_candidates",0))
            preflight_gate_warning=capability_pilot_warning(preflight_summary,REQUESTED)
            print("PREFLIGHT_WARNING "+json.dumps({"blocking":False,
                                                  "capability_evidence_candidates":capability,
                                                  "strictly_promising_candidates":promising,
                                                  "summary":preflight_summary,
                                                  "warning":preflight_gate_warning},
                                                 sort_keys=True),flush=True)
        p={"status":"searching","requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,"completed_batches":batches,"failed_candidate_evaluations":failed,"candidate_failure_reason_counts":failure_reasons,"elapsed_seconds":round(time.monotonic()-start,2),"seed":SEED,"algorithm":ALGORITHM,"train_sha256":train_hash,"development_sha256":dev_hash,"train_passing_pool":len(train_pass),"train_near_pool":len(train_near),"last_batch_unique":actual,"preflight_summary":preflight_summary,"preflight_warning":preflight_gate_warning,"prior_candidates_used":result.get("prior_candidates_used",0),"indicator_cache_stats":result.get("indicator_cache_stats")}
        write(PROGRESS,p); print("SEARCH_PROGRESS "+json.dumps(p,sort_keys=True),flush=True)
        # Never terminate discovery merely because several batches have no passing candidate.
        # This was the root cause of the previous run stopping after only 1,000 candidates.
        # Continue until the explicit runtime/candidate budget is reached; report partial searches honestly.
    search_complete=completed>=REQUESTED
    if not search_complete:
        write(PROGRESS,{"status":"partial_search_reported","requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,"completed_batches":batches,"failed_candidate_evaluations":failed,"reason":"Runtime budget or candidate-space exhaustion; any result will be explicitly marked incomplete."})
        print(f"SEARCH_PARTIAL evaluated={completed} requested={REQUESTED}; evaluating auditable candidates found so far without claiming exhaustive search",flush=True)
    pool=[]; append_unique(pool,train_pass); append_unique(pool,train_near)
    if not pool: raise RuntimeError("No auditable candidates returned")
    print(f"DEV_START candidate_pool={len(pool)}; OOS is not inspected before selection freezes",flush=True)
    dev_pass=[]; dev_near=[]; audit_failures=[]
    for idx,item in enumerate(pool,1):
        try:
            c=normalize_contract(item.get("contract") or cfg); primary=backtest(dev,item["strategy"],c,_prepared_context=prepared_dev); second=independent_replay(dev,item["strategy"],c)
            match,evidence=ledger_match(primary,second); distance,miss,distances=_candidate_distance(primary["metrics"],c)
            item["train_metrics"]=item.get("metrics",{}); item["development_metrics"]=brief(primary["metrics"]); item["development_distance"]=float(distance)
            item["development_failed_conditions"]=miss; item["development_criterion_distances"]=distances; item["development_audit_status"]="MATCH" if match else "MISMATCH"; item["development_audit_evidence"]=evidence
            item["development_data_sha256"]=primary["metrics"]["data_hash_sha256"]; item["development_execution_data_sha256"]=primary["metrics"].get("execution_data_hash_sha256")
            item["development_passes_all_targets"]=not miss and match; item["_dev_ledger"]=primary["ledger"]; item["_dev_span_days"]=primary["metrics"]["span_days"]
            if not match: audit_failures.append({"split":"development","strategy":item["strategy"],"evidence":evidence})
            elif miss: dev_near.append(item)
            else: dev_pass.append(item)
        except Exception as exc:
            item["development_audit_status"]="FAILED"; item["development_error"]=str(exc); audit_failures.append({"split":"development","strategy":item["strategy"],"error":str(exc)})
        if idx%10==0: print(f"DEV_PROGRESS {idx}/{len(pool)}",flush=True)
    dev_pass=sorted(dev_pass,key=near_key); dev_near=sorted(dev_near,key=near_key)
    frozen=dev_pass+dev_near[:5] if dev_pass else dev_near[:5]
    if not frozen: raise RuntimeError("No candidate survived Development audit; no strategy certificate")
    print(f"SELECTION_FROZEN dev_pass={len(dev_pass)} dev_near={len(dev_near)} oos_candidates={len(frozen)}",flush=True)
    # OOS metrics are computed only after the Development candidate set is frozen.
    oos_hash=frame_hash(oos); prepared_oos=prepare_backtest_data(oos); oos_backtest_hash=prepared_oos.data_hash; oos_results=[]
    for idx,item in enumerate(frozen,1):
        try:
            c=normalize_contract(item.get("contract") or cfg); primary=backtest(oos,item["strategy"],c,_prepared_context=prepared_oos); second=independent_replay(oos,item["strategy"],c)
            match,evidence=ledger_match(primary,second); distance,miss,distances=_candidate_distance(primary["metrics"],c)
            item["oos_metrics"]=brief(primary["metrics"]); item["oos_data_sha256"]=primary["metrics"]["data_hash_sha256"]; item["oos_execution_data_sha256"]=primary["metrics"].get("execution_data_hash_sha256")
            item["oos_audit_status"]="MATCH" if match else "MISMATCH"; item["oos_audit_evidence"]=evidence; item["oos_failed_conditions"]=miss; item["oos_distance"]=float(distance); item["_oos_ledger"]=primary["ledger"]; item["_oos_span_days"]=primary["metrics"]["span_days"]
            item["oos_passes_all_targets"]=not miss and match
        except Exception as exc:
            item["oos_audit_status"]="FAILED"; item["oos_error"]=str(exc); item["oos_passes_all_targets"]=False
        oos_results.append(item); print(f"OOS_PROGRESS {idx}/{len(frozen)}",flush=True)
    validated=[x for x in oos_results if x.get("development_passes_all_targets") and x.get("development_audit_status")=="MATCH" and x.get("oos_passes_all_targets")]
    def activity_eligible(item):
        # Enforce the agreed minimum activity using the user-approved minimum of 300 closed trades in each calendar month.
        # All immutable performance gates remain enforced by _candidate_distance and
        # independent replay; this gate only enforces the agreed daily trade frequency.
        for key in ("train_metrics","development_metrics","oos_metrics"):
            metrics=item.get(key) or {}
            span=float(metrics.get("span_days",0.0) or 0.0)
            trades=int(metrics.get("trades",0) or 0)
            rate=float(metrics.get("trades_per_day", trades/span if span>0 else 0.0) or 0.0)
            if rate<8.0:
                return False
        return True
    validated=[x for x in validated if activity_eligible(x)]
    activity_eligible_near=sorted([x for x in oos_results if activity_eligible(x)],key=near_key)[:5]
    # Never hide all real, audited results just because none reaches the activity
    # floor. Return up to five honest near-misses with their actual failed gates.
    # These are explicitly uncertified; this does not relax any acceptance rule.
    audited_near=sorted([
        x for x in oos_results
        if x.get("development_audit_status")=="MATCH" and x.get("oos_audit_status")=="MATCH"
    ],key=near_key)[:5]
    eligible_near=activity_eligible_near or audited_near or sorted(oos_results,key=near_key)[:5]
    delivery=validated if validated else eligible_near
    if validated:
        outcome="VALIDATED_STRATEGIES_FOUND" if search_complete else "VALIDATED_STRATEGIES_FOUND_SEARCH_INCOMPLETE"
    elif delivery:
        if activity_eligible_near:
            outcome="NO_STRATEGY_PASSED_ALL_REQUIRED_SPLITS" if search_complete else "INCOMPLETE_SEARCH_NO_CERTIFIED_STRATEGY"
        else:
            outcome="NO_CANDIDATE_MET_DAILY_ACTIVITY_FLOOR" if search_complete else "INCOMPLETE_SEARCH_NO_ACTIVITY_ELIGIBLE_CANDIDATE"
    else:
        outcome="NO_CANDIDATE_MET_DAILY_ACTIVITY_FLOOR" if search_complete else "INCOMPLETE_SEARCH_NO_ACTIVITY_ELIGIBLE_CANDIDATE"
        # Preserve a truthful, downloadable report instead of exporting low-activity
        # candidates that the user explicitly excluded.
        full_hash=frame_hash(df)
        write(OUT/"report.json",{
            "schema_version":1,"outcome":outcome,"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,"candidate_search_complete":search_complete,
            "failed_candidate_evaluations":failed,"candidate_failure_reason_counts":failure_reasons,
            "completed_batches":batches,"family_evaluated_counts":family_counts,"seed":SEED,"algorithm":ALGORITHM,
            "contract":cfg,"target_criteria":{"win_rate_pct_strictly_greater_than":40,"profit_factor_strictly_greater_than":1,
            "avg_net_tp_per_day_at_least":3,"max_consecutive_losses_at_most":12,"min_trades_per_day":8,"risk_usd_fixed":10,"initial_balance":1000,"net_reward_risk":3.5},
            "data":data_manifest,"splits":{"train":{"rows":len(train),"sha256":train_hash},
            "development":{"rows":len(dev),"sha256":dev_hash},"oos":{"rows":len(oos),"sha256":oos_hash}},
            "validated_strategy_count":len(validated),"returned_candidate_count":0,"results":[],
            "activity_eligible_near_miss_count":len(activity_eligible_near),
            "note":"If no strategy meets every required gate, up to five actual OOS-evaluated candidates are returned. Candidates that miss the >=8 trades/day floor or fail independent Development/OOS ledger reconciliation are explicitly uncertified; audit status/evidence and failed conditions are included. No acceptance threshold is relaxed.",
            "reproducibility":"Candidate selection and primary/independent ledger audits ran before this activity gate."})
        with zipfile.ZipFile(OUT/"validated_strategies.zip","w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("README.txt","No strategies exported: no candidate met the agreed daily activity floor across Train, Development, and OOS.")
        write(PROGRESS,{"status":"completed" if search_complete else "partial_search_reported","outcome":outcome,
            "requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,"validated_strategy_count":len(validated),
            "returned_candidate_count":0,"report":"research-results/report.json","strategy_zip":"research-results/validated_strategies.zip"})
        print("FINAL_REPORT "+json.dumps({"outcome":outcome,"requested_candidates":REQUESTED,
            "evaluated_unique_candidates":completed,"validated_strategy_count":len(validated),"returned_candidate_count":0,
            "full_dataset_sha256":full_hash,"report_path":"research-results/report.json","zip_path":"research-results/validated_strategies.zip"},sort_keys=True),flush=True)
        return
    # Near-misses are not certificates. Preserve their real per-split metrics and audit
    # evidence without running expensive full-dataset standalone certification for them.
    # Full-data replay/export remains mandatory for candidates that pass every gate.
    if not validated:
        full_hash=frame_hash(df)
        results=[{k:v for k,v in item.items() if not k.startswith("_")} for item in delivery]
        report={
            "schema_version":1,"outcome":outcome,
            "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,
            "candidate_search_complete":search_complete,"failed_candidate_evaluations":failed,
            "candidate_failure_reason_counts":failure_reasons,"completed_batches":batches,
            "surrogate_model_fits":fits,"generations":generations,"family_evaluated_counts":family_counts,
            "speed_benchmark":speed_benchmark,"candidate_search_budget_seconds":MAX_SECONDS,
            "seed":SEED,"algorithm":ALGORITHM,"preflight_summary":preflight_summary,
            "preflight_warning":preflight_gate_warning,"contract":cfg,
            "target_criteria":{"win_rate_pct_strictly_greater_than":40,"profit_factor_strictly_greater_than":1,
                "avg_net_tp_per_day_at_least":3,"max_consecutive_losses_at_most":12,
                "min_trades_per_day":8.0,"min_trades_per_month":300,"risk_usd_fixed":10,"initial_balance":1000,"net_reward_risk":3.5,
                "commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,
                "entry_model":"next_candle_open","exit_model":"SL/TP only","max_positions":1,
                "loss_streak_resets_only_on_full_TP":True},
            "data":data_manifest,"splits":{
                "train":{"rows":len(train),"sha256":train_hash,"used_for_candidate_generation":True},
                "development":{"rows":len(dev),"sha256":dev_hash,"used_for_selection_and_near_miss_ranking":True},
                "oos":{"rows":len(oos),"sha256":oos_hash,"used_only_after_selection_frozen":True}},
            "development_passing_count":len(dev_pass),"development_near_miss_count":len(dev_near),
            "oos_evaluated_count":len(oos_results),"validated_strategy_count":0,
            "returned_candidate_count":len(delivery),
            "activity_eligible_near_miss_count":len(activity_eligible_near),
            "near_miss_shortfall":max(0,5-len(delivery)),
            "development_audit_failures":audit_failures,"exports":[],
            "results":results,
            "note":"No candidate passed all acceptance gates in this bounded search. Returned candidates are uncertified near-misses with per-split metrics, failed conditions, and independent audit statuses/evidence. Full-data standalone certification is intentionally not run for near-misses; no acceptance threshold was relaxed.",
            "selection_policy":"Train creates candidates; Development freezes selection; OOS is inspected only after selection is frozen. All required performance/activity gates and independent ledger audits remain mandatory for certification.",
            "reproducibility":"Dataset/split hashes, strategy specs, per-split metrics, and primary-vs-independent audit evidence are included. Full-data standalone exports are only created for strategies that pass every gate."
        }
        write(OUT/"report.json",report)
        with zipfile.ZipFile(OUT/"validated_strategies.zip","w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("README.txt","No validated strategy was found. This archive contains the exact Binance data manifest; see report.json for up to five real, uncertified near-misses, their per-split metrics, failed gates, and independent audit evidence. No standalone strategy is certified or exported.")
            z.writestr("market_data_manifest.json",json.dumps(data_manifest,ensure_ascii=False,sort_keys=True,indent=2))
        write(PROGRESS,{"status":"completed" if search_complete else "partial_search_reported",
            "outcome":outcome,"candidate_search_complete":search_complete,
            "requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,
            "validated_strategy_count":0,"returned_candidate_count":len(delivery),
            "report":"research-results/report.json","strategy_zip":"research-results/validated_strategies.zip"})
        print("FINAL_REPORT "+json.dumps({"outcome":outcome,"requested_candidates":REQUESTED,
            "evaluated_unique_candidates":completed,"validated_strategy_count":0,
            "returned_candidate_count":len(delivery),"full_dataset_sha256":full_hash,
            "report_path":"research-results/report.json","zip_path":"research-results/validated_strategies.zip"},sort_keys=True),flush=True)
        return
    full_hash=frame_hash(df); prepared_full=prepare_backtest_data(df); template=(ROOT/"app"/"standalone_template.py").read_text(encoding="utf-8")
    zip_path=OUT/"validated_strategies.zip"; exports=[]
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr("README.txt","Strategy Foundry reproducible export. Read each manifest. Standalone Python verifies exact full-data and run SHA-256; unresolved SL/TP candles trigger exact 1-second Binance USD-M fetch, and incomplete resolution fails closed. No live orders.")
        z.writestr("market_data_manifest.json",json.dumps(data_manifest,ensure_ascii=False,sort_keys=True,indent=2))
        for idx,item in enumerate(delivery,1):
            c=normalize_contract(item.get("contract") or cfg)
            train_full=backtest(train,item["strategy"],c,_prepared_context=prepared_train)
            full=backtest(df,item["strategy"],c,_prepared_context=prepared_full); second=independent_replay(df,item["strategy"],c)
            match,evidence=ledger_match(full,second)
            item["trade_diagnostics"]={
                "train":trade_diagnostics(train_full["ledger"],train_full["metrics"]["span_days"]),
                "development":trade_diagnostics(item.get("_dev_ledger",[]),item.get("_dev_span_days",0)),
                "oos":trade_diagnostics(item.get("_oos_ledger",[]),item.get("_oos_span_days",0)),
                "full":trade_diagnostics(full["ledger"],full["metrics"]["span_days"])
            }
            item["full_data_audit_status"]="MATCH" if match else "MISMATCH"; item["full_data_audit_evidence"]=evidence
            if not match: continue
            source=template.replace("__SPEC__",repr(full["strategy"])).replace("__CONTRACT__",repr(full["contract"])).replace("__DATA_HASH__",full["metrics"]["data_hash_sha256"]).replace("__RUN_HASH__",full["run_hash_sha256"])
            name=f"strategy_{idx:03d}.py"; z.writestr(name,source)
            z.writestr(f"strategy_{idx:03d}_train_ledger.json",json.dumps(train_full["ledger"],ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
            z.writestr(f"strategy_{idx:03d}_development_ledger.json",json.dumps(item.get("_dev_ledger",[]),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
            z.writestr(f"strategy_{idx:03d}_full_ledger.json",json.dumps(full["ledger"],ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
            z.writestr(f"strategy_{idx:03d}_oos_ledger.json",json.dumps(item.get("_oos_ledger",[]),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
            manifest={"strategy":item["strategy"],"contract":full["contract"],"full_dataset_sha256":full_hash,"train_sha256":train_hash,"development_sha256":dev_hash,"oos_sha256":oos_hash,"full_run_metrics":brief(full["metrics"]),"full_run_hash_sha256":full["run_hash_sha256"],"full_data_independent_audit":{"status":"MATCH",**evidence},"development_metrics":item.get("development_metrics"),"development_audit_status":item.get("development_audit_status"),"development_passes_all_targets":item.get("development_passes_all_targets"),"oos_metrics":item.get("oos_metrics"),"oos_audit_status":item.get("oos_audit_status"),"oos_passes_all_targets":item.get("oos_passes_all_targets"),"oos_failed_conditions":item.get("oos_failed_conditions",[]),"trade_diagnostics":item.get("trade_diagnostics",{})}
            z.writestr(f"strategy_{idx:03d}_manifest.json",json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
            exports.append({"file":name,"strategy":item["strategy"],"full_run_metrics":brief(full["metrics"]),"full_run_hash_sha256":full["run_hash_sha256"],"full_data_audit_status":"MATCH"})
    results=[]
    for item in delivery:
        results.append({k:v for k,v in item.items() if not k.startswith("_")})
    report={"schema_version":1,"outcome":outcome,"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,"candidate_search_complete":search_complete,"failed_candidate_evaluations":failed,"candidate_failure_reason_counts":failure_reasons,"completed_batches":batches,"surrogate_model_fits":fits,"generations":generations,"family_evaluated_counts":family_counts,"speed_benchmark":speed_benchmark,"candidate_search_budget_seconds":MAX_SECONDS,"invalid_stop_proposals_removed":True,"parallel_worker_limit":(max(1,min(8,os.cpu_count() or 1)) if ALGORITHM=="stratified_random" else 1),"indicator_cache_stats":{"hits":prepared_train.indicator_cache_hits,"misses":prepared_train.indicator_cache_misses,"entries":len(prepared_train.indicator_cache),"limit":prepared_train.indicator_cache_limit},"seed":SEED,"algorithm":ALGORITHM,"preflight_summary":preflight_summary,"preflight_warning":preflight_gate_warning,"capability_gate":{"min_avg_tp_per_day":0.5,"min_trades":50,"min_win_rate_pct":25,"min_profit_factor":0.5,"blocking":False,"policy":"advisory only; all candidate families remain eligible and pilot results never terminate the requested search"},"contract":cfg,"target_criteria":{"win_rate_pct_strictly_greater_than":40,"profit_factor_strictly_greater_than":1,"avg_net_tp_per_day_at_least":3,"max_consecutive_losses_at_most":12,"min_trades_per_day":8.0,"min_trades_per_month":300,"risk_usd_fixed":10,"initial_balance":1000,"net_reward_risk":3.5,"commission_pct_per_side":0.04,"slippage_pct_per_side":0.02,"entry_model":"next_candle_open","exit_model":"SL/TP only","max_positions":1,"loss_streak_resets_only_on_full_TP":True},"data":data_manifest,"splits":{"train":{"rows":len(train),"sha256":train_hash,"backtest_sha256":train_backtest_hash,"used_for_candidate_generation":True},"development":{"rows":len(dev),"sha256":dev_hash,"backtest_sha256":dev_backtest_hash,"used_for_selection_and_near_miss_ranking":True},"oos":{"rows":len(oos),"sha256":oos_hash,"backtest_sha256":oos_backtest_hash,"used_only_after_selection_frozen":True}},"development_passing_count":len(dev_pass),"development_near_miss_count":len(dev_near),"oos_evaluated_count":len(oos_results),"validated_strategy_count":len(validated),"returned_candidate_count":len(delivery),"activity_eligible_near_miss_count":len(activity_eligible_near),"near_miss_shortfall":max(0,5-len(delivery)) if not validated else 0,"development_audit_failures":audit_failures,"exports":exports,"results":results,"selection_policy":"Train creates candidates; win rate, profit factor, TP/day, max consecutive losses, >=8 trades/day and independent ledger match are mandatory eligibility gates; four active performance criteria have equal-weight normalized distance; Development freezes selection; OOS is only evaluated after selection is frozen and never used to rank/select.","reproducibility":"Primary and independent ledgers compared field-by-field at rel_tol=1e-10 and abs_tol=1e-8; standalone exports include exact hashes and auto-fetch 1-second windows only for ambiguous candles."}
    write(OUT/"report.json",report); write(PROGRESS,{"status":"completed" if search_complete else "partial_search_reported","candidate_search_complete":search_complete,"outcome":outcome,"candidate_search_complete":search_complete,"requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,"validated_strategy_count":len(validated),"returned_candidate_count":len(delivery),"report":"research-results/report.json","strategy_zip":"research-results/validated_strategies.zip"})
    print("FINAL_REPORT "+json.dumps({"outcome":outcome,"requested_candidates":REQUESTED,"evaluated_unique_candidates":completed,"validated_strategy_count":len(validated),"returned_candidate_count":len(delivery),"full_dataset_sha256":full_hash,"train_sha256":train_hash,"development_sha256":dev_hash,"oos_sha256":oos_hash,"report_path":"research-results/report.json","zip_path":"research-results/validated_strategies.zip"},sort_keys=True),flush=True)
def main():
    write(PROGRESS,{"status":"downloading_data","requested_candidates":REQUESTED,"seed":SEED,"algorithm":ALGORITHM})
    df,manifest=asyncio.run(get_data()); write(PROGRESS,{"status":"data_validated","requested_candidates":REQUESTED,"dataset_sha256":manifest["dataset_sha256"],"rows":len(df)})
    run(df,manifest)
if __name__=="__main__": main()
