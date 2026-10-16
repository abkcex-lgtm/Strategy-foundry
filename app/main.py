from __future__ import annotations
import io,csv,hashlib
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .engine import backtest,search_candidates,CONTRACT
from .data import download_binance_month,validate_ohlcv,split_reference_period

ROOT=Path(__file__).resolve().parent
app=FastAPI(title='Strategy Foundry API',version='0.1.0',description='Research/backtest only; no live orders.')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
DATASETS:dict[str,pd.DataFrame]={}; RUNS:dict[str,dict[str,Any]]={}
class BacktestRequest(BaseModel):
    strategy:dict[str,Any]=Field(default_factory=dict)
    contract:dict[str,Any]=Field(default_factory=dict)
class DownloadRequest(BaseModel):
    symbol:str='ETHUSDT'; interval:str='1m'; start_year:int=2026; start_month:int=4; end_year:int=2026; end_month:int=9
@app.get('/',response_class=HTMLResponse)
def home(): return (ROOT/'static'/'index.html').read_text(encoding='utf-8')
@app.get('/api/health')
def health(): return {'status':'ok','service':'strategy-foundry','version':'0.1.0','execution':'server-side','live_trading':False}
@app.get('/api/contract')
def contract(): return {'contract':CONTRACT,'notes':['No live orders','No synthetic candles','Ambiguous SL/TP fails closed','Train/dev/OOS kept separate']}
@app.post('/api/data/upload')
async def upload_csv(file:UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith('.csv'): raise HTTPException(400,'فقط فایل CSV کندل پذیرفته می‌شود.')
    try:
        raw=await file.read(); df=pd.read_csv(io.BytesIO(raw))
        aliases={'timestamp':'open_time','time':'open_time','open time':'open_time'}
        df=df.rename(columns={c:aliases.get(c.lower(),c.lower()) for c in df.columns})
        if 'open_time' in df and pd.api.types.is_datetime64_any_dtype(df['open_time']): df['open_time']=df['open_time'].astype('int64')//1000000
        report=validate_ohlcv(df,60000 if '1m' in file.filename.lower() else None)
        key=str(report['rows'])+'-'+str(int(df.open_time.iloc[0])); DATASETS[key]=df
        return {'dataset_id':key,'report':report,'sha256_csv':hashlib.sha256(raw).hexdigest()}
    except Exception as exc: raise HTTPException(422,f'بررسی داده رد شد: {exc}') from exc
@app.post('/api/data/binance')
async def download_data(req:DownloadRequest):
    if req.interval not in {'1m','3m','5m','15m','30m','1h','2h','4h','1d'}: raise HTTPException(400,'بازهٔ زمانی پشتیبانی‌نشده')
    start=req.start_year*12+req.start_month-1; end=req.end_year*12+req.end_month-1
    if not 1<=req.start_month<=12 or not 1<=req.end_month<=12 or end<start or end-start>24: raise HTTPException(400,'بازهٔ ماه‌ها نامعتبر است یا بیش از 25 ماه است.')
    chunks=[]; hashes=[]
    try:
        for idx in range(start,end+1):
            df,h,url=await download_binance_month(req.symbol,req.interval,idx//12,idx%12+1); chunks.append(df); hashes.append({'sha256':h,'url':url})
        merged=pd.concat(chunks,ignore_index=True).sort_values('open_time').reset_index(drop=True)
        report=validate_ohlcv(merged,60000 if req.interval=='1m' else None)
        key=f'{req.symbol.upper()}-{req.interval}-{req.start_year}{req.start_month:02d}-{req.end_year}{req.end_month:02d}'; DATASETS[key]=merged
        return {'dataset_id':key,'symbol':req.symbol.upper(),'interval':req.interval,'report':report,'monthly_archive_hashes':hashes,'rows':len(merged),'splits':{k:len(v) for k,v in split_reference_period(merged).items()}}
    except Exception as exc: raise HTTPException(422,f'دریافت/اعتبارسنجی کامل نشد؛ بک‌تست شروع نشد: {exc}') from exc
@app.get('/api/datasets')
def datasets(): return [{'dataset_id':k,'rows':len(v),'first_open_time':int(v.open_time.iloc[0]),'last_open_time':int(v.open_time.iloc[-1])} for k,v in DATASETS.items()]
@app.post('/api/backtest/{dataset_id}')
def run_backtest(dataset_id:str,req:BacktestRequest):
    df=DATASETS.get(dataset_id)
    if df is None: raise HTTPException(404,'Dataset پیدا نشد؛ ابتدا داده را دریافت کنید.')
    try:
        result=backtest(df,req.strategy or None,req.contract or None); run_id=result['run_hash_sha256'][:16]; RUNS[run_id]=result
        return {'run_id':run_id,**result}
    except Exception as exc: raise HTTPException(422,f'اجرای بک‌تست رد شد: {exc}') from exc
@app.post('/api/search/{dataset_id}')
def search(dataset_id:str,candidates:int=500,seed:int=42):
    df=DATASETS.get(dataset_id)
    if df is None: raise HTTPException(404,'ابتدا داده را دریافت کنید.')
    try:
        splits=split_reference_period(df); train=splits['train']
        if len(train)<100: raise HTTPException(422,'دادهٔ Train برای بازهٔ قرارداد Apr-Jun 2026 کافی نیست؛ جست‌وجو آغاز نشد.')
        result=search_candidates(train,candidates,seed)
        result['dataset_id']=dataset_id
        result['search_period']='2026-04-01 to 2026-07-01 UTC (TRAIN only)'
        result['development_period']='2026-07-01 to 2026-08-01 UTC (not used in search)'
        result['oos_period']='2026-08-01 to 2026-10-01 UTC (not used in search)'
        return result
    except HTTPException: raise
    except Exception as exc: raise HTTPException(422,str(exc)) from exc

@app.post('/api/evaluate/{dataset_id}')
def evaluate_strategy(dataset_id:str,req:BacktestRequest):
    df=DATASETS.get(dataset_id)
    if df is None: raise HTTPException(404,'ابتدا داده را دریافت کنید.')
    try:
        splits=split_reference_period(df); report={'dataset_id':dataset_id,'strategy':req.strategy,'periods':{},'note':'Evaluate only a frozen strategy. Repeated OOS selection contaminates the holdout.'}
        for key,part in splits.items():
            if len(part)<100:
                report['periods'][key]={'status':'PREFLIGHT_FAILED','rows':len(part),'reason':'Insufficient rows for the fixed reference period'}
                continue
            try:
                result=backtest(part,req.strategy or None,req.contract or None)
                run_id=result['run_hash_sha256'][:16]; RUNS[run_id]=result
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
    script='import sys, pandas as pd\nfrom app.engine import backtest\nSPEC = '+repr(r['strategy'])+'\nEXPECTED_DATA_SHA256 = '+repr(r['metrics']['data_hash_sha256'])+'\nif __name__ == "__main__":\n df=pd.read_csv(sys.argv[1])\n result=backtest(df,SPEC)\n print(result["metrics"])\n if result["metrics"]["data_hash_sha256"] != EXPECTED_DATA_SHA256: raise SystemExit("DATA HASH MISMATCH")\n'
    return StreamingResponse(iter([script]),media_type='text/x-python',headers={'Content-Disposition':f'attachment; filename={run_id}-replay.py'})
