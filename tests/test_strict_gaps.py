import pandas as pd
import pytest

from app.data import DataIntegrityError, _drop_binance_header, validate_ohlcv


def test_binance_csv_header_row_is_removed():
    frame = pd.DataFrame([
        ["open_time", "open", "high", "low", "close", "volume"],
        ["1775001600000", "2104.48", "2105.45", "2102.28", "2103.21", "6955.005"],
        ["1775001660000", "2103.21", "2103.48", "2099.19", "2102.18", "4824.334"],
    ])
    parsed = _drop_binance_header(frame)
    assert len(parsed) == 2
    assert parsed.iloc[0, 0] == "1775001600000"
    parsed.columns = ["open_time", "open", "high", "low", "close", "volume"]
    report = validate_ohlcv(parsed, 60000, strict_gaps=True)
    assert report["rows"] == 2
    assert report["gap_count"] == 0


def test_binance_csv_without_header_keeps_first_candle():
    frame = pd.DataFrame([
        ["1775001600000", "2104.48", "2105.45", "2102.28", "2103.21", "6955.005"],
        ["1775001660000", "2103.21", "2103.48", "2099.19", "2102.18", "4824.334"],
    ])
    parsed = _drop_binance_header(frame)
    assert len(parsed) == 2
    assert parsed.iloc[0, 0] == "1775001600000"


def test_strict_validation_rejects_missing_candles():
    frame = pd.DataFrame({
        "open_time": [0, 60000, 180000],
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100.5, 101.5, 102.5],
        "volume": [1, 1, 1],
    })
    with pytest.raises(DataIntegrityError, match="Timestamp gaps detected: 1"):
        validate_ohlcv(frame, 60000, strict_gaps=True)


def test_gap_report_counts_more_than_first_hundred_details():
    rows = []
    for i in range(150):
        rows.append({"open_time": i * 120000, "open": 100+i, "high": 101+i, "low": 99+i, "close": 100.5+i, "volume": 1})
    report = validate_ohlcv(pd.DataFrame(rows), 60000)
    assert report["gap_count"] == 149
    assert len(report["gaps"]) == 100
