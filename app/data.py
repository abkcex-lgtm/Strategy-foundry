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
    """Return chronological train/dev/OOS splits without ever shuffling candles.

    Preserve the agreed Apr-Sep 2026 reference windows when the dataset fully
    covers that period. For other date ranges, use a deterministic 60/20/20
    chronological fallback so unrelated symbols, intervals, and historical
    windows are still searchable rather than silently producing empty splits.
    """
    if df.empty:
        return {'train': df.copy(), 'development': df.copy(), 'oos': df.copy()}
    ts = pd.to_datetime(df.open_time, unit='ms', utc=True)
    start = ts.iloc[0]
    end = ts.iloc[-1]
    reference_start = pd.Timestamp('2026-04-01', tz='UTC')
    reference_end = pd.Timestamp('2026-10-01', tz='UTC')
    if start <= reference_start and end >= pd.Timestamp('2026-09-30 23:59:00', tz='UTC'):
        return {
            'train': df[(ts >= '2026-04-01') & (ts < '2026-07-01')].copy(),
            'development': df[(ts >= '2026-07-01') & (ts < '2026-08-01')].copy(),
            'oos': df[(ts >= '2026-08-01') & (ts < '2026-10-01')].copy(),
        }
    # Use row boundaries instead of randomized splits: future candles never
    # appear in training when the dataset does not match the reference window.
    n = len(df)
    train_end = max(1, int(n * 0.60))
    dev_end = max(train_end + 1, int(n * 0.80)) if n >= 3 else n
    dev_end = min(n, dev_end)
    return {
        'train': df.iloc[:train_end].copy(),
        'development': df.iloc[train_end:dev_end].copy(),
        'oos': df.iloc[dev_end:].copy(),
    }


from functools import lru_cache

@lru_cache(maxsize=2048)
def _cached_binance_execution_rows(symbol: str, start_ms: int, seconds: int):
    url = "https://fapi.binance.com/fapi/v1/klines"
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            response = client.get(url, params={
                "symbol": symbol, "interval": "1s", "startTime": start_ms,
                "endTime": start_ms + seconds * 1000 - 1, "limit": min(1500, seconds),
            })
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        raise DataIntegrityError(f"Binance USD-M 1-second execution fetch failed for {symbol} at {start_ms}: {exc}") from exc
    if not isinstance(payload, list) or len(payload) > seconds:
        got = len(payload) if isinstance(payload, list) else "non-list response"
        raise DataIntegrityError(f"Invalid Binance 1-second execution window for {symbol} at {start_ms}: expected at most {seconds} trade-backed candles, got {got}")
    rows = []
    try:
        for item in payload:
            if not isinstance(item, list) or len(item) < 6:
                raise ValueError("Unexpected Binance kline response shape")
            rows.append((int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])))
    except (TypeError, ValueError, IndexError) as exc:
        raise DataIntegrityError(f"Invalid Binance 1-second execution data: {exc}") from exc
    frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    if frame.empty:
        # No trade-backed one-second klines means no traded price movement in
        # the requested interval; preserve the empty result, never synthesize OHLC.
        return tuple()
    validate_ohlcv(frame, 1000, strict_gaps=False)
    times = frame["open_time"].to_numpy(dtype="int64")
    if (times < start_ms).any() or (times >= start_ms + seconds * 1000).any():
        raise DataIntegrityError("Binance 1-second execution timestamps fall outside the requested interval")
    return tuple(tuple(row) for row in frame.itertuples(index=False, name=None))


def download_binance_execution_window(symbol: str, start_ms: int, interval_ms: int) -> pd.DataFrame:
    """Fetch every verified 1-second USD-M candle for one ambiguous base candle."""
    symbol = str(symbol).upper()
    if not symbol.isalnum():
        raise DataIntegrityError("Invalid symbol for Binance execution resolution")
    if interval_ms <= 0 or interval_ms % 1000:
        raise DataIntegrityError("Execution interval must be a positive whole number of seconds")
    if start_ms < 1_000_000_000_000:
        raise DataIntegrityError(f"Unresolved same-candle SL/TP ambiguity at implausible timestamp {start_ms}; automatic market-data resolution refused")
    seconds = interval_ms // 1000
    if seconds > 1500:
        raise DataIntegrityError(f"Unresolved same-candle SL/TP ambiguity at {start_ms}; automatic 1-second window exceeds 1500 seconds")
    rows = _cached_binance_execution_rows(symbol, int(start_ms), int(seconds))
    return pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
