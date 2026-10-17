from __future__ import annotations
import asyncio,io,csv,hashlib,math,uuid
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File,Query
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .engine import backtest,search_candidates
from .audit import independent_replay, FIELDS
from .data import download_binance_month,validate_ohlcv,split_reference_period
from .persistence import load_datasets,load_runs,save_dataset,save_run,list_history,save_profile,list_profiles

ROOT=Path(__file__).resolve().parent
app=FastAPI(title='Strategy Foundry API',version='0.2.0',description='Server-side strategy research and simulation only; never sends live orders.')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
DATASETS:dict[str,pd.DataFrame]=load_datasets()
RUNS:dict[str,dict[str,Any]]=load_runs()
RUN_DATASET:dict[str,str]={}

class BacktestRequest(BaseModel):
    strategy:dict[str,Any]=Field(default_factory=dict)
    contract:dict[str,Any]=Field(default_factory=dict)
class SearchRequest(BaseModel):
    candidates:int=Field(default=500,ge=1,le=10000)
    seed:int=42
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
def health(): return {'status':'ok','service':'strategy-foundry','version':'0.3.0','execution':'server-side','live_trading':False,'persistent_store':'sqlite','loaded_datasets':len(DATASETS),'loaded_runs':len(RUNS),'features':{'tracked_binance_download_jobs':True,'download_progress_endpoint':'/api/jobs/{job_id}','question_first_assistant_routing':True}}
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
        key=str(report['rows'])+'-'+str(int(df.open_time.iloc[0]))+'-'+hashlib.sha256(raw).hexdigest()[:8]
        DATASETS[key]=df; save_dataset(key,df,{'source':'user_csv','sha256_csv':hashlib.sha256(raw).hexdigest()})
        return {'dataset_id':key,'report':report,'sha256_csv':hashlib.sha256(raw).hexdigest()}
    except Exception as exc: raise HTTPException(422,f'بررسی داده رد شد: {exc}DOWNLOAD_JOBS:dict[str,dict[str,Any]]={}

async def _run_download_job(job_id:str,req:DownloadRequest):
    job=DOWNLOAD_JOBS[job_id]
    job.update({'status':'running','message':'در حال اتصال به آرشیو رسمی Binance','completed_months':0,'total_months':(req.end_year*12+req.end_month-1)-(req.start_year*12+req.start_month-1)+1,'rows_downloaded':0})
    try:
        start=req.start_year*12+req.start_month-1
        end=req.end_year*12+req.end_month-1
        chunks=[]; hashes=[]
        for idx in range(start,end+1):
            year,month=idx//12,idx%12+1
            job.update({'message':f'دریافت و بررسی {year}-{month:02d}','current_month':f'{year}-{month:02d}'})
            df,h,url=await download_binance_month(req.symbol,req.interval,year,month)
            chunks.append(df); hashes.append({'sha256':h,'url':url})
            job.update({'completed_months':len(chunks),'rows_downloaded':sum(len(part) for part in chunks)})
        merged=pd.concat(chunks,ignore_index=True).sort_values('open_time').reset_index(drop=True)
        job.update({'message':'اعتبارسنجی نهایی ترتیب زمانی، شکاف‌ها و OHLCV'})
        report=validate_ohlcv(merged,60000 if req.interval=='1m' else None,strict_gaps=(req.interval=='1m'))
        key=f'{req.symbol.upper()}-{req.interval}-{req.start_year}{req.start_month:02d}-{req.end_year}{req.end_month:02d}'
        metadata={'source':'binance_vision','symbol':req.symbol.upper(),'interval':req.interval,'monthly_archive_hashes':hashes}
        save_dataset(key,merged,metadata)
        DATASETS[key]=merged
        result={'dataset_id':key,'symbol':req.symbol.upper(),'interval':req.interval,'report':report,'monthly_archive_hashes':hashes,'rows':len(merged),'splits':{k:len(v) for k,v in split_reference_period(merged).items()}}
        job.update({'status':'completed','message':'داده دریافت، اعتبارسنجی و ذخیره شد','result':result,'rows_downloaded':len(merged),'completed_months':job['total_months']})
    except Exception as exc:
        job.update({'status':'failed','message':f'دریافت/اعتبارسنجی کامل نشد؛ بک‌تست شروع نشد: {exc}','error':str(exc)})

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

@app.get('/api/jobs/{job_id}')
def get_job(job_id:str):
    job=DOWNLOAD_JOBS.get(job_id)
    if job is None: raise HTTPException(404,'شناسهٔ عملیات پیدا نشد؛ ممکن است سرور پس از شروع مجدد حافظهٔ عملیات را پاک کرده باشد.')
    return job

') from exc

@app.get('/api/datasets')
def datasets():
    out=[]
    for key,frame in DATASETS.items():
        split_counts={name:int(len(part)) for name,part in split_reference_period(frame).items()}
        out.append({'dataset_id':key,'rows':len(frame),'first_open_time':int(frame.open_time.iloc[0]),'last_open_time':int(frame.open_time.iloc[-1]),'splits':split_counts})
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

@app.post('/api/search/{dataset_id}')
def search(dataset_id:str,req:SearchRequest):
    df=DATASETS.get(dataset_id)
    if df is None: raise HTTPException(404,'ابتدا داده را دریافت کنید.')
    try:
        splits=split_reference_period(df); train=splits['train']
        if len(train)<100: raise HTTPException(422,'دادهٔ Train برای بازهٔ Apr-Jun 2026 کافی نیست؛ جست‌وجو آغاز نشد.')
        result=search_candidates(train,req.candidates,req.seed,req.contract)
        result.update({'dataset_id':dataset_id,'search_period':'2026-04-01 to 2026-07-01 UTC (TRAIN only)',
          'development_period':'2026-07-01 to 2026-08-01 UTC (not used in search)',
          'oos_period':'2026-08-01 to 2026-10-01 UTC (not used in search)'})
        return result
    except HTTPException: raise
    except Exception as exc: raise HTTPException(422,str(exc)) from exc

@app.post('/api/evaluate/{dataset_id}')
def evaluate_strategy(dataset_id:str,req:BacktestRequest):
    df=DATASETS.get(dataset_id)
    if df is None: raise HTTPException(404,'ابتدا داده را دریافت کنید.')
    try:
        splits=split_reference_period(df)
        report={'dataset_id':dataset_id,'strategy':req.strategy,'periods':{},
          'note':'هر بار استفاده از OOS برای انتخاب/تنظیم استراتژی، استقلال مجموعهٔ آزمون را تضعیف می‌کند.'}
        for key,part in splits.items():
            if len(part)<100:
                report['periods'][key]={'status':'PREFLIGHT_FAILED','rows':len(part),'reason':'Insufficient rows for the fixed reference period'}
                continue
            try:
                result=backtest(part,req.strategy or None,req.contract or None)
                run_id=_store_result(result,dataset_id)
                report['periods'][key]={'status':'COMPLETED','run_id':run_id,'metrics':result['metrics'],'trade_count':len(result['ledger'])}
            except Exception as exc:
                report['periods'][key]={'status':'PREFLIGHT_FAILED','rows':len(part),'reason':str(exc)}
        report['certified']=False
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
        full_hash=hashlib.sha256(df.to_csv(index=False,float_format='%.12g').encode()).hexdigest()
        if full_hash != r.get('metrics',{}).get('data_hash_sha256'):
            raise HTTPException(409,'این اجرا روی یک بخش Train/Dev/OOS انجام شده، نه کل دیتاست؛ برای جلوگیری از تطبیق اشتباه، audit مستقل روی همین قطعه در این نسخه در دسترس نیست.')
        second=independent_replay(df,r['strategy'],r['contract'])
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
    except Exception as exc: raise HTTPException(422,f'تطبیق مستقل متوقف شد: {exc}') from exc

@app.get('/api/runs/{run_id}/chart')
def run_chart(run_id:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=500,ge=1,le=2000)):
    r=RUNS.get(run_id)
    if not r: raise HTTPException(404,'Run پیدا نشد.')
    dataset_id=r.get('dataset_id') or RUN_DATASET.get(run_id); df=DATASETS.get(dataset_id) if dataset_id else None
    if df is None: raise HTTPException(409,'دادهٔ اصلی در دسترس نیست.')
    part=df.iloc[offset:offset+limit]
    candles=part[['open_time','open','high','low','close','volume']].to_dict(orient='records')
    times={int(c['open_time']) for c in candles}
    markers=[{k:t.get(k) for k in ('trade_id','signal_time','entry_time','exit_time','direction','entry_price','stop_loss','take_profit','exit_price','exit_reason','net_pnl')}
             for t in r['ledger'] if int(t['entry_time']) in times or int(t['exit_time']) in times]
    return {'run_id':run_id,'offset':offset,'total_rows':len(df),'candles':candles,'trades':markers}

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
