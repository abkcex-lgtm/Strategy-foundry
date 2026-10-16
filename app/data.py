from __future__ import annotations
import hashlib, io, zipfile
from typing import Any
import httpx
import pandas as pd

class DataIntegrityError(ValueError):
    pass

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _drop_binance_header(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept Binance Vision CSVs both with and without a column-header row."""
    if not frame.empty and str(frame.iloc[0, 0]).strip().lower() in {"open_time", "open time", "timestamp"}:
        return frame.iloc[1:].reset_index(drop=True)
    return frame

def validate_ohlcv(frame: pd.DataFrame, interval_ms: int | None = 60000, strict_gaps: bool = False) -> dict[str, Any]:
    required = ['open_time','open','high','low','close','volume']
    missing = [c for c in required if c not in frame.columns]
    if missing: raise DataIntegrityError(f'Missing columns: {missing}')
    df = frame.copy()
    if df.empty: raise DataIntegrityError('Dataset is empty')
    for c in required: df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[required].isna().any().any(): raise DataIntegrityError('Missing/non-numeric OHLCV values')
    if df.open_time.duplicated().any(): raise DataIntegrityError('Duplicate open_time')
    if not df.open_time.is_monotonic_increasing: raise DataIntegrityError('Candles out of order')
    if (df.open_time.diff().dropna() <= 0).any(): raise DataIntegrityError('Non-increasing timestamps')
    if (df[['open','high','low','close']] <= 0).any().any(): raise DataIntegrityError('Prices must be positive')
    if (df.volume < 0).any(): raise DataIntegrityError('Negative volume')
    if (df.high < df[['open','close','low']].max(axis=1)).any(): raise DataIntegrityError('Invalid high')
    if (df.low > df[['open','close','high']].min(axis=1)).any(): raise DataIntegrityError('Invalid low')
    gaps=[]; gap_count=0
    if interval_ms:
        deltas=df.open_time.diff().iloc[1:].to_numpy()
        bad_positions=[i+1 for i,delta in enumerate(deltas) if delta != interval_ms]
        gap_count=len(bad_positions)
        gaps=[{'row':int(i),'delta_ms':int(df.open_time.iloc[i]-df.open_time.iloc[i-1])} for i in bad_positions[:100]]
        if strict_gaps and gap_count:
            raise DataIntegrityError(f'Timestamp gaps detected: {gap_count}; first gaps: {gaps[:5]}')
    return {'rows':int(len(df)), 'first_open_time':int(df.open_time.iloc[0]), 'last_open_time':int(df.open_time.iloc[-1]), 'gaps':gaps, 'gap_count':gap_count}

async def download_binance_month(symbol: str, interval: str, year: int, month: int):
    symbol=symbol.upper()
    if not symbol.isalnum() or not 1 <= month <= 12 or year < 2017: raise ValueError('Invalid symbol/date')
    url=f'https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip'
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        response=await client.get(url); response.raise_for_status()
    raw=response.content
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names=[n for n in zf.namelist() if n.lower().endswith('.csv')]
            if not names: raise DataIntegrityError('Archive has no CSV')
            with zf.open(names[0]) as fp: df=_drop_binance_header(pd.read_csv(fp, header=None))
    except zipfile.BadZipFile as exc: raise DataIntegrityError('Invalid Binance ZIP archive') from exc
    names=['open_time','open','high','low','close','volume','close_time','quote_volume','trade_count','taker_buy_base','taker_buy_quote','ignore']
    if df.shape[1] < 6: raise DataIntegrityError('Unexpected kline columns')
    df=df.iloc[:,:min(12,df.shape[1])]; df.columns=names[:df.shape[1]]
    df=df[['open_time','open','high','low','close','volume']]
    validate_ohlcv(df, 60000 if interval == '1m' else None, strict_gaps=(interval == '1m'))
    return df, sha256_bytes(raw), url

def split_reference_period(df: pd.DataFrame):
    ts=pd.to_datetime(df.open_time, unit='ms', utc=True)
    return {'train':df[(ts >= '2026-04-01') & (ts < '2026-07-01')].copy(), 'development':df[(ts >= '2026-07-01') & (ts < '2026-08-01')].copy(), 'oos':df[(ts >= '2026-08-01') & (ts < '2026-10-01')].copy()}
