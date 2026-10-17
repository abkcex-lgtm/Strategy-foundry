from __future__ import annotations
import asyncio,io,csv,hashlib,json,math,uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File,Query
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .engine import backtest,search_candidates,StrategySpec,normalize_contract,_candidate_distance
from .audit import independent_replay, FIELDS
from .data import download_binance_month,validate_ohlcv,split_reference_period
from .persistence import load_datasets,load_dataset_metadata,load_runs,save_dataset,save_run,list_history,save_profile,list_profiles,save_search,list_search_history,load_search_result,save_job,load_jobs,save_job_genome_hashes,load_job_genome_hashes,save_job_batch_checkpoint,storage_status

ROOT=Path(__file__).resolve().parent

def _period_label(frame:pd.DataFrame, note:str='') -> str:
    if frame.empty:
        return 'empty split'
    first=pd.to_datetime(int(frame.open_time.iloc[0]),unit='ms',utc=True).isoformat()
    last=pd.to_datetime(int(frame.open_time.iloc[-1]),unit='ms',utc=True).isoformat()
    suffix=f' ({note})' if note else ''
    return f'{first} to {last}{suffix}'

app=FastAPI(title='Strategy Foundry API',version='0.7.7',description='Server-side strategy research and simulation only; never sends live orders.')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
DATASETS:dict[str,pd.DataFrame]=load_datasets()
DATASET_METADATA:dict[str,dict[str,Any]]=load_dataset_metadata()
RUNS:dict[str,dict[str,Any]]=load_runs()
RUN_DATASET:dict[str,str]={}

class BacktestRequest(BaseModel):
    strategy:dict[str,Any]=Field(default_factory=dict)
    contract:dict[str,Any]=Field(default_factory=dict)
class SearchRequest(BaseModel):
    candidates:int=Field(default=500,ge=1,le=500000)
    seed:int=42
    algorithm:str="surrogate_genetic"  # hybrid_genetic | stratified_random
    search_space:dict[str,Any]=Field(default_factory=dict)
    contract:dict[str,Any]=Field(default_factory=dict)
class DownloadRequest(BaseModel):
    symbol:str='ETHUSDT'
    interval:str='1m'
    start_year:int=2026
    start_month:int=4
    end_year:int=2026
    end_month:int=9
class ProfileRequest(BaseModel):
    profile_name:str=Field(min_length=1,max_length=80)
    config:dict[str,Any]=Field(default_factory=dict)

class AIChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=8000)
    history:list[dict[str,str]]=Field(default_factory=list)

@app.get('/',response_class=HTMLResponse)
def home(): return (ROOT/'static'/'index.html').read_text(encoding='utf-8')
@app.get('/api/health')
def health(): return {'status':'ok','service':'strategy-foundry','version':'0.7.7','execution':'server-side','live_trading':False,'persistent_store':'sqlite','storage':storage_status(),'loaded_datasets':len(DATASETS),'loaded_runs':len(RUNS),'features':{'tracked_binance_download_jobs':True,'tracked_search_jobs':True,'persistent_search_history':True,'download_progress_endpoint':'/api/jobs/{job_id}','search_job_endpoint':'/api/search-jobs/{dataset_id}','search_history_endpoint':'/api/search-history','persistent_job_status':True,'batch_search_max_candidates':500000,'global_genome_deduplication':True,'batch_checkpoint_resume':True,'question_first_assistant_routing':True}}
@app.get('/api/contract')
def contract():
    from .engine import normalize_contract
    return {'contract':normalize_contract(),'editable':True,'notes':['این فقط یک پروفایل اولیه و قابل‌ویرایش است، نه قرارداد قفل‌شده.','سفارش واقعی ارسال نمی‌شود.','داده مصنوعی بازار ساخته نمی‌شود.','ابهام حل‌نشدهٔ SL/TP باعث توقف می‌شود.']}
@app.post('/api/settings/validate')
def validate_settings(req:BacktestRequest):
    from .engine import normalize_contract
    try:
        cfg=normalize_contract(req.contract)
        return {'valid':True,'contract':cfg,'effective_cost_pct_per_side':cfg['commission_pct_per_side']+cfg['slippage_pct_per_side']}
    except Exception as exc: raise HTTPException(422,f'Settings rejected: {exc}') from exc
@app.get('/api/profiles')
def get_profiles(): return list_profiles()
@app.post('/api/profiles')
def put_profile(req:ProfileRequest):
    from .engine import normalize_contract
    try: cfg=normalize_contract(req.config)
    except Exception as exc: raise HTTPException(422,f'Profile rejected: {exc}') from exc
    save_profile(req.profile_name,cfg)
    return {'saved':True,'profile_name':req.profile_name,'config':cfg}
@app.get('/api/history')
def history(limit:int=Query(default=100,ge=1,le=500)): return list_history(limit)

@app.get('/api/ai/status')
def ai_status():
    from .ai_router import provider_status
    return provider_status()

@app.post('/api/ai/chat')
async def ai_chat(req:AIChatRequest):
    from .ai_router import chat
    return await chat(req.message,req.history)

@app.post('/api/data/upload')
async def upload_csv(file:UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith('.csv'): raise HTTPException(400,'فقط فایل CSV کندل پذیرفته می‌شود.')
    try:
        raw=await file.read(); df=pd.read_csv(io.BytesIO(raw))
        aliases={'timestamp':'open_time','time':'open_time','open time':'open_time'}
        df=df.rename(columns={c:aliases.get(c.lower(),c.lower()) for c in df.columns})
        if 'open_time' in df and pd.api.types.is_datetime64_any_dtype(df['open_time']): df['open_time']=df['open_time'].astype('int64')//1000000
        report=validate_ohlcv(df,60000 if '1m' in file.filename.lower() else None,strict_gaps=('1m' in file.filename.lower()))
        csv_hash=hashlib.sha256(raw).hexdigest()
        key=str(report['rows'])+'-'+str(int(df.open_time.iloc[0]))+'-'+csv_hash[:8]
        metadata={'source':'user_csv','sha256_csv':csv_hash}
        DATASETS[key]=df; DATASET_METADATA[key]=metadata; save_dataset(key,df,metadata)
        return {'dataset_id':key,'report':report,'sha256_csv':csv_hash}
    except Exception as exc: raise HTTPException(422,f'بررسی داده رد شد: {exc}') from exc

DOWNLOAD_JOBS:dict[str,dict[str,Any]]={}

async def _run_download_job(job_id:str,req:DownloadRequest):
    job=DOWNLOAD_JOBS[job_id]
    job.update({'status':'running','kind':'data_download','message':'در حال اتصال به آرشیو رسمی Binance','completed_months':0,'total_months':(req.end_year*12+req.end_month-1)-(req.start_year*12+req.start_month-1)+1,'rows_downloaded':0}); save_job(job_id, 'data_download', job)
    try:
        start=req.start_year*12+req.start_month-1
        end=req.end_year*12+req.end_month-1
        chunks=[]; hashes=[]
        for idx in range(start,end+1):
            year,month=idx//12,idx%12+1
            job.update({'message':f'دریافت و بررسی {year}-{month:02d}','current_month':f'{year}-{month:02d}'})
            df,h,url=await download_binance_month(req.symbol,req.interval,year,month)
            chunks.append(df); hashes.append({'sha256':h,'url':url})
            job.update({'completed_months':len(chunks),'rows_downloaded':sum(len(part) for part in chunks)}); save_job(job_id, 'data_download', job)
        merged=pd.concat(chunks,ignore_index=True).sort_values('open_time').reset_index(drop=True)
        job.update({'message':'اعتبارسنجی نهایی ترتیب زمانی، شکاف‌ها و OHLCV'})
        report=validate_ohlcv(merged,60000 if req.interval=='1m' else None,strict_gaps=(req.interval=='1m'))
        key=f'{req.symbol.upper()}-{req.interval}-{req.start_year}{req.start_month:02d}-{req.end_year}{req.end_month:02d}'
        metadata={'source':'binance_vision','symbol':req.symbol.upper(),'interval':req.interval,'monthly_archive_hashes':hashes}
        save_dataset(key,merged,metadata)
        DATASETS[key]=merged; DATASET_METADATA[key]=metadata
        result={'dataset_id':key,'symbol':req.symbol.upper(),'interval':req.interval,'report':report,'monthly_archive_hashes':hashes,'rows':len(merged),'splits':{k:len(v) for k,v in split_reference_period(merged).items()}}
        job.update({'status':'completed','message':'داده دریافت، اعتبارسنجی و ذخیره شد','result':result,'rows_downloaded':len(merged),'completed_months':job['total_months']}); save_job(job_id, 'data_download', job)
    except Exception as exc:
        job.update({'status':'failed','message':f'دریافت/اعتبارسنجی کامل نشد؛ بک‌تست شروع نشد: {exc}','error':str(exc)}); save_job(job_id, 'data_download', job)

@app.post('/api/data/binance')
async def download_data(req:DownloadRequest):
    if req.interval not in {'1m','3m','5m','15m','30m','1h','2h','4h','1d'}: raise HTTPException(400,'بازهٔ زمانی پشتیبانی‌نشده')
    if not req.symbol.isalnum() or not 1<=req.start_month<=12 or not 1<=req.end_month<=12: raise HTTPException(400,'نماد یا ماه نامعتبر است.')
    start=req.start_year*12+req.start_month-1; end=req.end_year*12+req.end_month-1
    if req.start_year<2017 or req.end_year<2017 or end<start or end-start>24: raise HTTPException(400,'بازهٔ ماه‌ها نامعتبر است یا بیش از 25 ماه است.')
    job_id=uuid.uuid4().hex
    DOWNLOAD_JOBS[job_id]={'job_id':job_id,'status':'queued','message':'درخواست در صف دریافت قرار گرفت','completed_months':0,'total_months':end-start+1,'rows_downloaded':0}
    asyncio.create_task(_run_download_job(job_id,req))
    return {'job_id':job_id,'status':'queued','message':'درخواست دریافت در سرور ثبت شد؛ وضعیت را از مسیر پیگیری بررسی کنید.'}

SEARCH_JOBS:dict[str,dict[str,Any]]={}
# Restore visible job history after process restart. In-flight batches are not resumed;
# only atomically completed batches can be resumed from their persisted checkpoint.
for _job_id, _job in load_jobs().items():
    if _job.get("kind") == "data_download":
        DOWNLOAD_JOBS[_job_id] = _job
    else:
        SEARCH_JOBS[_job_id] = _job
    if _job.get("status") in {"queued", "running"}:
        _job.update({"status": "interrupted", "message": "سرور دوباره راه‌اندازی شد؛ عملیات از آخرین دستهٔ کامل‌شده قابل ادامه است."})
        save_job(_job_id, _job.get("kind", "strategy_search"), _job)


@app.get('/api/jobs/{job_id}')
def get_job(job_id:str):
    job=DOWNLOAD_JOBS.get(job_id) or SEARCH_JOBS.get(job_id)
    if job is None: raise HTTPException(404,'شناسهٔ عملیات پیدا نشد؛ ممکن است تاریخچهٔ قدیمی پاک شده باشد یا میزبان ذخیره‌سازی پایدار نداشته باشد.')
    return job


@app.get('/api/datasets')
def datasets():
    out=[]
    for key,frame in DATASETS.items():
        split_counts={name:int(len(part)) for name,part in split_reference_period(frame).items()}
        data_hash=hashlib.sha256(frame.to_csv(index=False,float_format='%.12g').encode()).hexdigest()
        out.append({'dataset_id':key,'rows':len(frame),'first_open_time':int(frame.open_time.iloc[0]),'last_open_time':int(frame.open_time.iloc[-1]),'splits':split_counts,'data_hash_sha256':data_hash,'metadata':DATASET_METADATA.get(key,{})})
    return out

@app.get('/api/datasets/{dataset_id}/candles')
def get_candles(dataset_id:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=500,ge=1,le=2000)):
    df=DATASETS.get(dataset_id)
    if df is None: raise HTTPException(404,'Dataset پیدا نشد.')
    part=df.iloc[offset:offset+limit]
    return {'dataset_id':dataset_id,'offset':offset,'total_rows':len(df),'candles':part[['open_time','open','high','low','close','volume']].to_dict(orient='records')}

def _store_result(result:dict[str,Any],dataset_id:str):
    run_id=result['run_hash_sha256'][:16]
    result['run_id']=run_id
    result['dataset_id']=dataset_id
    RUNS[run_id]=result; RUN_DATASET[run_id]=dataset_id
    save_run(run_id,dataset_id,result)
    return run_id

@app.post('/api/backtest/{dataset_id}')
def run_backtest(dataset_id:str,req:BacktestRequest):
    df=DATASETS.get(dataset_id)
    if df is None: raise HTTPException(404,'Dataset پیدا نشد؛ ابتدا داده را دریافت کنید.')
    try:
        result=backtest(df,req.strategy or None,req.contract or None); run_id=_store_result(result,dataset_id)
        return {'run_id':run_id,**result}
    except Exception as exc: raise HTTPException(422,f'اجرای بک‌تست رد شد: {exc}') from exc

def _rank_near_misses_on_development(result:dict[str,Any],splits:dict[str,pd.DataFrame],req:SearchRequest):
    """Rank a frozen TRAIN near-miss pool on DEVELOPMENT; never inspect OOS."""
    if result.get('all_passing') or not result.get('results'):
        return
    development=splits.get('development',pd.DataFrame())
    if len(development)<100:
        result['near_miss_ranking_policy']='TRAIN-only fallback: DEVELOPMENT has fewer than 100 rows; OOS was not read.'
        return
    for item in result['results']:
        train_metrics=item.get('metrics',{})
        train_distance=float(item.get('distance',1e30))
        item['train_metrics']=train_metrics
        item['train_distance']=train_distance
        try:
            cfg=normalize_contract(item.get('contract') or result.get('contract') or req.contract)
            dev_run=backtest(development,item['strategy'],cfg)
            distance,failed,distances=_candidate_distance(dev_run['metrics'],cfg)
            item['metrics']=dev_run['metrics']
            item['metrics_split']='development'
            item['distance']=float(distance)
            item['failed_filters']=failed
            item['filter_distances']=distances
            item['development_evaluation_status']='COMPLETED'
            item['development_data_hash_sha256']=dev_run['metrics']['data_hash_sha256']
        except Exception as exc:
            # Keep the TRAIN metrics visible, but rank an unevaluable candidate after
            # all candidates with a valid DEVELOPMENT evaluation.
            item['metrics']=train_metrics
            item['metrics_split']='train'
            item['distance']=1e30
            item['development_evaluation_status']='PREFLIGHT_FAILED'
            item['development_evaluation_reason']=str(exc)
    def key(item):
        metrics=item.get('metrics',{})
        pf=metrics.get('profit_factor')
        return (
            0 if item.get('development_evaluation_status')=='COMPLETED' else 1,
            float(item.get('distance',1e30)),
            int(metrics.get('max_consecutive_losses',10**9)),
            -float(metrics.get('win_rate_pct',0)),
            -float(metrics.get('avg_net_tp_per_day',0)),
            -float(pf if pf is not None else 1e9),
        )
    result['results']=sorted(result['results'],key=key)[:5]
    result['near_miss_ranking_policy']='Candidate generation and pass detection use TRAIN; when no TRAIN candidate passes, near misses are ranked on DEVELOPMENT after the pool is frozen. OOS is not read.'


def _perform_search(dataset_id:str,req:SearchRequest,progress_callback=None,excluded_genome_hashes=None,include_genome_hashes=False,save_result=True):
    df=DATASETS.get(dataset_id)
    if df is None: raise ValueError('ابتدا داده را دریافت کنید.')
    splits=split_reference_period(df); train=splits['train']
    if len(train)<100: raise ValueError('دادهٔ Train برای بازهٔ Apr-Jun 2026 کافی نیست؛ جست‌وجو آغاز نشد.')
    result=search_candidates(train,req.candidates,req.seed,req.contract,req.algorithm,req.search_space,progress_callback,excluded_genome_hashes)
    _rank_near_misses_on_development(result,splits,req)
    train_hash=hashlib.sha256(train.to_csv(index=False,float_format='%.12g').encode()).hexdigest()
    development_hash=hashlib.sha256(splits['development'].to_csv(index=False,float_format='%.12g').encode()).hexdigest()
    manifest={'dataset_id':dataset_id,'train_data_hash_sha256':train_hash,
              'development_data_hash_sha256':development_hash,
              'near_miss_ranking_policy':result.get('near_miss_ranking_policy','Passing strategies are selected on TRAIN only; OOS is not read.'),
              'candidates':req.candidates,'seed':req.seed,'contract':result.get('contract',{})}
    manifest_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True,allow_nan=False).encode()).hexdigest()
    search_id=uuid.uuid4().hex[:16]
    result.update({'dataset_id':dataset_id,'search_id':search_id,'search_manifest':manifest,'search_manifest_sha256':manifest_hash,
      'search_period':_period_label(splits['train'],'TRAIN candidate generation and pass detection'),
      'development_period':_period_label(splits['development'],'near-miss ranking only when no TRAIN candidate passes'),
      'oos_period':_period_label(splits['oos'],'untouched evaluation; not used for selection'),
      'selection_policy':'Candidate generation and pass detection use TRAIN. If no TRAIN candidate passes, the frozen near-miss pool is ranked on DEVELOPMENT. OOS is not read for candidate selection.'})
    genome_hashes=result.pop('evaluated_genome_hashes',[])
    if save_result: save_search(search_id,dataset_id,manifest,result)
    if include_genome_hashes: result['_internal_evaluated_genome_hashes']=genome_hashes
    return result

@app.post('/api/search/{dataset_id}')
def search(dataset_id:str,req:SearchRequest):
    if req.candidates>10000: raise HTTPException(422,'برای جست‌وجوی بیش از ۱۰٬۰۰۰ نامزد از اجرای پس‌زمینه‌ای استفاده کنید تا جست‌وجو دسته‌ای و قابل‌بازیابی باشد.')
    if dataset_id not in DATASETS: raise HTTPException(404,'ابتدا داده را دریافت کنید.')
    try: return _perform_search(dataset_id,req)
    except ValueError as exc:
        if 'ابتدا داده' in str(exc) or 'Train برای بازه' in str(exc): raise HTTPException(422,str(exc)) from exc
        raise HTTPException(422,str(exc)) from exc

async def _run_search_job(job_id:str,dataset_id:str,req:SearchRequest):
    job=SEARCH_JOBS[job_id]
    job.update({'status':'running','kind':'strategy_search','dataset_id':dataset_id,
                'requested_candidates':req.candidates,
                'message':'جست‌وجوی دسته‌ای؛ فقط TRAIN برای انتخاب استفاده می‌شود.'})
    save_job(job_id, 'strategy_search', job)
    try:
        # Checkpoints are committed only at batch boundaries. A process restart
        # resumes from the last completed batch and deterministically reuses its seed.
        seen=load_job_genome_hashes(job_id)
        completed=int(job.get('completed_candidates',0))
        batch_index=int(job.get('completed_batches',0))
        passing=list(job.get('checkpoint_passing_results',[]))
        near=list(job.get('checkpoint_near_misses',[]))
        family_counts=dict(job.get('checkpoint_family_counts',{}))
        total_failed=int(job.get('checkpoint_failed_candidates',0))
        total_generations=int(job.get('checkpoint_generations',0))
        total_model_fits=int(job.get('checkpoint_surrogate_model_fits',0))
        max_attempts=max(5,math.ceil(req.candidates/10000)*5)
        while completed<req.candidates and batch_index<max_attempts:
            batch_size=min(10000,req.candidates-completed)
            batch_seed=int(req.seed+batch_index*1000003)
            batch_req=req.copy(update={'candidates':batch_size,'seed':batch_seed})
            batch_base=completed
            def on_progress(progress):
                progress_count=int(progress.get('evaluated_candidates',0))
                if progress_count%250!=0 and progress_count!=batch_size: return
                job.update(progress)
                job.update({'status':'running','requested_candidates':req.candidates,
                            'evaluated_candidates':batch_base+progress_count,
                            'completed_candidates':batch_base,
                            'completed_batches':batch_index,
                            'current_batch':batch_index+1,
                            'message':f'دستهٔ {batch_index+1}؛ ارزیابی نامزدهای یکتا؛ انتخاب فقط از TRAIN'})
                save_job(job_id,'strategy_search',job)
            chunk=await asyncio.to_thread(_perform_search,dataset_id,batch_req,on_progress,seen,True,False)
            hashes=chunk.pop('_internal_evaluated_genome_hashes',[])
            seen.update(hashes)
            actual=len(hashes)
            completed+=actual
            batch_index+=1
            total_failed+=int(chunk.get('failed_candidates',0))
            total_generations+=int(chunk.get('generations',0))
            total_model_fits+=int(chunk.get('surrogate_model_fits',0))
            for family,count in chunk.get('family_evaluated_counts',{}).items():
                family_counts[family]=family_counts.get(family,0)+int(count)
            if chunk.get('all_passing'):
                known={json.dumps(x.get('strategy',{}),sort_keys=True,separators=(',',':')) for x in passing}
                for item in chunk.get('results',[]):
                    key=json.dumps(item.get('strategy',{}),sort_keys=True,separators=(',',':'))
                    if key not in known:
                        passing.append(item); known.add(key)
            else:
                known={json.dumps(x.get('strategy',{}),sort_keys=True,separators=(',',':')) for x in near}
                for item in chunk.get('results',[]):
                    key=json.dumps(item.get('strategy',{}),sort_keys=True,separators=(',',':'))
                    if key not in known:
                        near.append(item); known.add(key)
                def near_key(item):
                    m=item.get('metrics',{})
                    pf=m.get('profit_factor')
                    return (float(item.get('distance',math.inf)),
                            int(m.get('max_consecutive_losses',10**9)),
                            -float(m.get('win_rate_pct',0)),
                            -float(m.get('avg_net_tp_per_day',0)),
                            -float(pf if pf is not None else 1e9))
                near=sorted(near,key=near_key)[:5]
            job.update({'status':'running','requested_candidates':req.candidates,
                        'evaluated_candidates':completed,'completed_candidates':completed,
                        'completed_batches':batch_index,'current_batch':batch_index+1,
                        'unique_genomes_seen':len(seen),
                        'checkpoint_passing_results':passing,
                        'checkpoint_near_misses':near,
                        'checkpoint_family_counts':family_counts,
                        'checkpoint_failed_candidates':total_failed,
                        'checkpoint_generations':total_generations,
                        'checkpoint_surrogate_model_fits':total_model_fits,
                        'message':f'دستهٔ {batch_index} تکمیل شد؛ {completed:,} نامزد یکتا ارزیابی شد.'})
            save_job_batch_checkpoint(job_id,'strategy_search',job,hashes)
            if actual==0 and batch_index>=max_attempts:
                break
        all_passing=bool(passing)
        selected=passing if all_passing else near[:5]
        splits=split_reference_period(DATASETS[dataset_id]); train=splits['train']
        development=splits['development']
        manifest={'dataset_id':dataset_id,
                  'train_data_hash_sha256':hashlib.sha256(train.to_csv(index=False,float_format='%.12g').encode()).hexdigest(),
                  'development_data_hash_sha256':hashlib.sha256(development.to_csv(index=False,float_format='%.12g').encode()).hexdigest(),
                  'near_miss_ranking_policy':'TRAIN pass detection; DEVELOPMENT ranks frozen near-miss pool only when no TRAIN candidate passes; OOS untouched.',
                  'requested_candidates':req.candidates,'evaluated_unique_candidates':completed,
                  'completed_batches':batch_index,'seed':req.seed,'algorithm':req.algorithm,
                  'contract':selected[0].get('contract',{}) if selected else req.contract,
                  'globally_deduplicated':True}
        manifest_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True,allow_nan=False).encode()).hexdigest()
        search_id=uuid.uuid4().hex[:16]
        final_result={'requested_candidates':req.candidates,'evaluated_candidates':completed,
                      'unique_genomes_seen':len(seen),'completed_batches':batch_index,
                      'failed_candidates':total_failed,'family_evaluated_counts':family_counts,
                      'generations':total_generations,'surrogate_model_fits':total_model_fits,
                      'seed':req.seed,'algorithm':req.algorithm,'contract':manifest['contract'],
                      'all_passing':all_passing,'results':selected,
                      'requested_near_misses':0 if all_passing else 5,
                      'returned_near_misses':0 if all_passing else len(selected),
                      'near_miss_shortfall':max(0,5-len(selected)) if not all_passing else 0,
                      'search_id':search_id,'dataset_id':dataset_id,
                      'search_manifest':manifest,'search_manifest_sha256':manifest_hash,
                      'search_period':_period_label(splits['train'],'TRAIN only'),
                      'development_period':_period_label(splits['development'],'near-miss ranking only when no TRAIN candidate passes'),
                      'oos_period':_period_label(splits['oos'],'not used in search'),
                      'selection_policy':'Candidates are globally deduplicated and pass detection uses TRAIN. If no TRAIN candidate passes, the frozen near-miss pool is ranked on DEVELOPMENT. OOS is not read for candidate selection.',
                      'near_miss_ranking_policy':manifest['near_miss_ranking_policy'],
                      'warning':('Completed all requested globally unique candidate evaluations; all reported passing candidates were independently replayed.' if completed>=req.candidates else
                                 'INCOMPLETE SEARCH: candidate space/attempt limit prevented the requested unique evaluation count. Do not treat this as a completed 500k search.') }
        save_search(search_id,dataset_id,manifest,final_result)
        status='completed' if completed>=req.candidates else 'completed_with_shortfall'
        job.update({'status':status,'message':'جست‌وجو پایان یافت.' if status=='completed' else 'جست‌وجو ناقص ماند؛ تعداد نامزد یکتای درخواستی تکمیل نشد.',
                    'result':final_result,'evaluated_candidates':completed,'completed_candidates':completed,
                    'requested_candidates':req.candidates,'unique_genomes_seen':len(seen),
                    'checkpoint_passing_results':passing,'checkpoint_near_misses':near,
                    'search_id':search_id})
        save_job(job_id,'strategy_search',job)
    except Exception as exc:
        job.update({'status':'failed','message':f'جست‌وجو کامل نشد: {exc}','error':str(exc)})
        save_job(job_id,'strategy_search',job)

@app.get('/api/search-history')
def search_history(limit:int=Query(default=30,ge=1,le=200)):
    return list_search_history(limit)

@app.get('/api/search-history/{search_id}')
def get_search_history_item(search_id:str):
    result=load_search_result(search_id)
    if result is None: raise HTTPException(404,'جست‌وجوی ذخیره‌شده پیدا نشد.')
    return result

@app.get('/api/search-jobs')
def list_search_jobs(limit:int=Query(default=20,ge=1,le=100)):
    visible={'queued','running','interrupted','failed','completed_with_shortfall'}
    rows=[{k:v for k,v in job.items() if k not in {'result','checkpoint_passing_results','checkpoint_near_misses','checkpoint_family_counts','checkpoint_failed_candidates','checkpoint_generations','checkpoint_surrogate_model_fits'}}
          for job in SEARCH_JOBS.values() if job.get('status') in visible]
    rows.sort(key=lambda item:item.get('job_id',''),reverse=True)
    return rows[:limit]

@app.post('/api/search-jobs/{dataset_id}')
async def start_search_job(dataset_id:str,req:SearchRequest):
    if dataset_id not in DATASETS: raise HTTPException(404,'ابتدا داده را دریافت کنید.')
    job_id=uuid.uuid4().hex
    SEARCH_JOBS[job_id]={'job_id':job_id,'kind':'strategy_search','status':'queued',
        'dataset_id':dataset_id,'request':req.dict(),
        'message':'جست‌وجوی دسته‌ای در صف سرور قرار گرفت','requested_candidates':req.candidates,
        'evaluated_candidates':0,'completed_candidates':0,'completed_batches':0,
        'checkpoint_passing_results':[],'checkpoint_near_misses':[],'algorithm':req.algorithm}
    save_job(job_id,'strategy_search',SEARCH_JOBS[job_id])
    asyncio.create_task(_run_search_job(job_id,dataset_id,req))
    return {'job_id':job_id,'status':'queued','message':'جست‌وجوی دسته‌ای در سرور ثبت شد؛ وضعیت را پیگیری کنید.'}

@app.post('/api/search-job/{job_id}/resume')
async def resume_search_job(job_id:str):
    job=SEARCH_JOBS.get(job_id)
    if job is None: raise HTTPException(404,'جست‌وجوی ذخیره‌شده پیدا نشد.')
    if job.get('status') not in {'interrupted','failed'}:
        raise HTTPException(409,'فقط جست‌وجوی متوقف‌شده یا ناموفق قابل ادامه است.')
    dataset_id=job.get('dataset_id')
    if dataset_id not in DATASETS: raise HTTPException(409,'دادهٔ اصلی در ذخیره‌سازی موجود نیست؛ ادامهٔ جست‌وجو امن نیست.')
    try: req=SearchRequest(**job['request'])
    except Exception as exc: raise HTTPException(409,f'تنظیمات ذخیره‌شده قابل بازیابی نیست: {exc}') from exc
    job.update({'status':'queued','message':'ادامه از آخرین دستهٔ کامل‌شده','error':None})
    save_job(job_id,'strategy_search',job)
    asyncio.create_task(_run_search_job(job_id,dataset_id,req))
    return {'job_id':job_id,'status':'queued','completed_batches':job.get('completed_batches',0),'evaluated_candidates':job.get('completed_candidates',0)}

@app.post('/api/evaluate/{dataset_id}')
def evaluate_strategy(dataset_id:str,req:BacktestRequest):
    df=DATASETS.get(dataset_id)
    if df is None: raise HTTPException(404,'ابتدا داده را دریافت کنید.')
    try:
        splits=split_reference_period(df)
        # Freeze one effective strategy and contract before evaluating any split.
        # OOS results are reported after freezing; they never feed selection logic.
        spec=StrategySpec(**req.strategy) if req.strategy else StrategySpec()
        spec.validate()
        cfg=normalize_contract(req.contract)
        full_hash=hashlib.sha256(df.to_csv(index=False,float_format='%.12g').encode()).hexdigest()
        split_manifest={}
        for key,part in splits.items():
            split_hash=hashlib.sha256(part.to_csv(index=False,float_format='%.12g').encode()).hexdigest()
            split_manifest[key]={
                'rows':int(len(part)),
                'data_hash_sha256':split_hash,
                'period':_period_label(part,key.upper()),
            }
        frozen_strategy=asdict(spec)
        frozen_strategy_hash=hashlib.sha256(json.dumps(frozen_strategy,sort_keys=True,allow_nan=False).encode()).hexdigest()
        contract_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True,allow_nan=False).encode()).hexdigest()
        manifest={
            'schema_version':1,
            'dataset_id':dataset_id,
            'full_data_hash_sha256':full_hash,
            'splits':split_manifest,
            'frozen_strategy':frozen_strategy,
            'frozen_strategy_sha256':frozen_strategy_hash,
            'effective_contract':cfg,
            'contract_sha256':contract_hash,
            'selection_policy':'Strategy and contract are frozen before split evaluation; OOS is evaluation-only and must not be used for selection or tuning.',
            'oos_used_for_selection':False,
        }
        manifest_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True,allow_nan=False).encode()).hexdigest()
        report={
            'dataset_id':dataset_id,
            'dataset_hash_sha256':full_hash,
            'strategy':frozen_strategy,
            'strategy_sha256':frozen_strategy_hash,
            'contract':cfg,
            'contract_sha256':contract_hash,
            'evaluation_manifest':manifest,
            'evaluation_manifest_sha256':manifest_hash,
            'periods':{},
            'oos_used_for_selection':False,
            'selection_policy':manifest['selection_policy'],
            'note':'این گزارش اجرای ثابت‌شده روی هر بخش را ثبت می‌کند؛ استفادهٔ بعدی از OOS برای انتخاب یا تنظیم استراتژی، استقلال آزمون را تضعیف می‌کند.',
        }
        for key,part in splits.items():
            split_hash=split_manifest[key]['data_hash_sha256']
            if len(part)<100:
                report['periods'][key]={'status':'PREFLIGHT_FAILED','rows':len(part),'data_hash_sha256':split_hash,'period':split_manifest[key]['period'],'reason':'Insufficient rows for split evaluation'}
                continue
            try:
                result=backtest(part,frozen_strategy,cfg)
                result['evaluation_split']=key
                run_id=_store_result(result,dataset_id)
                report['periods'][key]={
                    'status':'COMPLETED',
                    'run_id':run_id,
                    'rows':len(part),
                    'data_hash_sha256':split_hash,
                    'period':split_manifest[key]['period'],
                    'metrics':result['metrics'],
                    'trade_count':len(result['ledger']),
                }
            except Exception as exc:
                report['periods'][key]={'status':'PREFLIGHT_FAILED','rows':len(part),'data_hash_sha256':split_hash,'period':split_manifest[key]['period'],'reason':str(exc)}
        report['certified']=False
        report['certification_note']='Evaluation manifest and split metrics are reproducibility evidence, not production certification or a profitability guarantee.'
        return report
    except Exception as exc: raise HTTPException(422,f'Split evaluation failed: {exc}') from exc

@app.get('/api/runs/{run_id}')
def get_run(run_id:str):
    if run_id not in RUNS: raise HTTPException(404,'Run پیدا نشد.')
    return RUNS[run_id]

@app.get('/api/runs/{run_id}/audit')
def audit_run(run_id:str):
    r=RUNS.get(run_id)
    if not r: raise HTTPException(404,'Run پیدا نشد.')
    dataset_id=r.get('dataset_id') or RUN_DATASET.get(run_id)
    df=DATASETS.get(dataset_id) if dataset_id else None
    if df is None: raise HTTPException(409,'دادهٔ اصلی این اجرا در ذخیره‌سازی موجود نیست؛ تطبیق مستقل انجام نشد.')
    try:
        scope=r.get('evaluation_split')
        audit_frame=split_reference_period(df).get(scope) if scope in {'train','development','oos'} else df
        if audit_frame is None or audit_frame.empty:
            raise HTTPException(409,'بازهٔ دادهٔ لازم برای ممیزی این اجرا موجود نیست.')
        full_hash=hashlib.sha256(audit_frame.to_csv(index=False,float_format='%.12g').encode()).hexdigest()
        if full_hash != r.get('metrics',{}).get('data_hash_sha256'):
            raise HTTPException(409,'هش داده با اجرای ذخیره‌شده تطبیق ندارد؛ ممیزی مستقل متوقف شد.')
        second=independent_replay(audit_frame,r['strategy'],r['contract'])
        first=r['ledger']; other=second['ledger']
        if len(first)!=len(other):
            return {'status':'MISMATCH','engine_a_trades':len(first),'engine_b_trades':len(other),'matched':False}
        numeric=('signal_price','entry_price','stop_loss','take_profit','exit_price','quantity','risk_budget','gross_pnl','entry_cost','exit_cost','fees_slippage','net_pnl','balance_after')
        mismatches=[]
        for idx,(a,b) in enumerate(zip(first,other),1):
            bad=[k for k in FIELDS if (k not in numeric and a.get(k)!=b.get(k)) or (k in numeric and not math.isclose(float(a.get(k,0)),float(b.get(k,0)),rel_tol=1e-10,abs_tol=1e-8))]
            if bad: mismatches.append({'trade_id':idx,'fields':bad})
            if len(mismatches)>=20: break
        ok=not mismatches and second['max_consecutive_losses']==r['metrics']['max_consecutive_losses']
        return {'status':'MATCH' if ok else 'MISMATCH','matched':ok,'engine_a_trades':len(first),'engine_b_trades':len(other),
                'mismatches':mismatches,'method':'independent replay; exact discrete fields + tight numeric tolerance',
                'certified':False,'note':'تطبیق موتور، به‌تنهایی صحت داده یا سودآوری آینده را تضمین نمی‌کند.'}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(422,f'تطبیق مستقل متوقف شد: {exc}') from exc

@app.get('/api/runs/{run_id}/chart')
def run_chart(run_id:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=500,ge=1,le=2000),trade_id:int|None=Query(default=None,ge=1)):
    r=RUNS.get(run_id)
    if not r: raise HTTPException(404,'Run پیدا نشد.')
    dataset_id=r.get('dataset_id') or RUN_DATASET.get(run_id); df=DATASETS.get(dataset_id) if dataset_id else None
    if df is None: raise HTTPException(409,'دادهٔ اصلی در دسترس نیست.')
    # Optional ledger-driven focus: locate the trade's entry candle and center it in the visible window.
    if trade_id is not None:
        trade=next((t for t in r.get('ledger',[]) if int(t.get('trade_id',-1))==trade_id),None)
        if trade is None: raise HTTPException(404,'معاملهٔ درخواستی در دفتر این اجرا پیدا نشد.')
        try:
            entry_time=int(trade['entry_time'])
            position=int(df['open_time'].searchsorted(entry_time))
            offset=max(0,min(max(0,len(df)-limit),position-limit//2))
        except Exception as exc:
            raise HTTPException(422,'موقعیت معامله برای فوکوس نمودار قابل محاسبه نیست.') from exc
    offset=max(0,min(offset,max(0,len(df)-limit)))
    part=df.iloc[offset:offset+limit]
    candles=part[['open_time','open','high','low','close','volume']].to_dict(orient='records')
    if candles:
        window_start=int(candles[0]['open_time']); window_end=int(candles[-1]['open_time'])
        markers=[{k:t.get(k) for k in ('trade_id','signal_time','entry_time','exit_time','direction','entry_price','stop_loss','take_profit','exit_price','exit_reason','net_pnl')}
                 for t in r['ledger'] if int(t['entry_time'])<=window_end and int(t['exit_time'])>=window_start]
    else: markers=[]
    return {'run_id':run_id,'offset':offset,'total_rows':len(df),'candles':candles,'trades':markers,'focused_trade_id':trade_id}

@app.get('/api/runs/{run_id}/ledger.csv')
def export_ledger(run_id:str):
    r=RUNS.get(run_id)
    if not r: raise HTTPException(404,'Run پیدا نشد.')
    out=io.StringIO(); fields=list(r['ledger'][0].keys()) if r['ledger'] else ['trade_id','entry_time','direction','entry_price','stop_loss','take_profit','exit_price','exit_reason','net_pnl']
    writer=csv.DictWriter(out,fieldnames=fields); writer.writeheader(); writer.writerows(r['ledger'])
    return StreamingResponse(iter([out.getvalue()]),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename={run_id}-ledger.csv'})

@app.get('/api/runs/{run_id}/python')
def export_python(run_id:str):
    r=RUNS.get(run_id)
    if not r: raise HTTPException(404,'Run پیدا نشد.')
    template=(ROOT/'standalone_template.py').read_text(encoding='utf-8')
    script=template.replace('__SPEC__',repr(r['strategy'])).replace('__CONTRACT__',repr(r['contract'])).replace('__DATA_HASH__',r['metrics']['data_hash_sha256']).replace('__RUN_HASH__',r['run_hash_sha256'])
    return StreamingResponse(iter([script]),media_type='text/x-python',headers={'Content-Disposition':f'attachment; filename={run_id}-replay.py'})
