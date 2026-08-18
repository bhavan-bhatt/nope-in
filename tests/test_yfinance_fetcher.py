"""Tests for yfinance fetcher (mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nope_in.data.fetchers.yfinance_fetcher import download_index_ohlcv


@pytest.fixture
def mock_yfinance_history():
    dates = pd.bdate_range("2023-01-02", periods=10)
    history = pd.DataFrame(
        {
            "Open": range(100, 110),
            "High": range(101, 111),
            "Low": range(99, 109),
            "Close": range(100, 110),
            "Volume": [1000] * 10,
        },
        index=dates,
    )
    history.index.name = "Date"
    return history


def test_download_index_ohlcv_cache(tmp_path, mock_yfinance_history):
    output = tmp_path / "nifty.parquet"

    with patch("nope_in.data.fetchers.yfinance_fetcher.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_yfinance_history
        df1 = download_index_ohlcv("^NSEI", "2023-01-01", "2023-01-31", output)

    assert output.exists()
    assert "close" in df1.columns

    # Second call reads cache
    df2 = download_index_ohlcv("^NSEI", "2023-01-01", "2023-01-31", output)
    assert len(df2) == len(df1)
