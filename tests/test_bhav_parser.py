"""Tests for bhav copy parser."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nope_in.data.fetchers.bhav_copy_fetcher import (
    merge_bhav_to_parquet,
    parse_bhavcopy_bytes,
    validate_bhav_schema,
)

FIXTURES = Path(__file__).parent / "fixtures" / "bhav_copy"


def test_parse_legacy_optidx():
    content = (FIXTURES / "legacy_optidx.csv").read_bytes()
    df = parse_bhavcopy_bytes(content, pd.Timestamp("2023-01-05"))
    assert len(df) == 3
    assert set(df["symbol"]) == {"NIFTY", "BANKNIFTY"}
    assert set(df["option_type"]) == {"CE", "PE"}
    assert df["settle_price"].min() > 0


def test_parse_udiff_ido():
    content = (FIXTURES / "udiff_ido.csv").read_bytes()
    df = parse_bhavcopy_bytes(content, pd.Timestamp("2024-01-05"))
    assert len(df) == 2
    assert df["symbol"].unique().tolist() == ["NIFTY"]
    assert df["underlying_value"].notna().all()


def test_validate_bhav_schema_drops_invalid():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-05")] * 3,
            "symbol": ["NIFTY", "NIFTY", "NIFTY"],
            "expiry_date": [pd.Timestamp("2023-01-12")] * 3,
            "strike": [18000.0, 0.0, 18000.0],
            "option_type": ["CE", "PE", "XX"],
            "settle_price": [164.0, 126.0, -1.0],
        }
    )
    clean = validate_bhav_schema(df)
    assert len(clean) == 1


def test_merge_bhav_to_parquet(tmp_path):
    bhav_dir = tmp_path / "bhav_copy" / "2023" / "01"
    bhav_dir.mkdir(parents=True)
    content = (FIXTURES / "legacy_optidx.csv").read_bytes()
    df = parse_bhavcopy_bytes(content, pd.Timestamp("2023-01-05"))
    out_csv = bhav_dir / "fo05JAN2023bhav.csv"
    df.to_csv(out_csv, index=False)

    output = tmp_path / "options_chain.parquet"
    merged = merge_bhav_to_parquet(bhav_dir.parent, output, symbols=["NIFTY", "BANKNIFTY"])
    assert len(merged) == 3
    assert output.exists()


def test_get_nse_trading_dates():
    from nope_in.data.fetchers.bhav_copy_fetcher import get_nse_trading_dates

    dates = get_nse_trading_dates(date(2023, 1, 1), date(2023, 1, 7))
    assert date(2023, 1, 1) not in dates  # Sunday
    assert date(2023, 1, 2) in dates  # Monday
