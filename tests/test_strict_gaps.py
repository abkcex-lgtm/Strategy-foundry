import pandas as pd
import pytest

from app.data import DataIntegrityError, validate_ohlcv


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
