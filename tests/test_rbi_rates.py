"""Tests for RBI rates fetcher."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nope_in.data.fetchers.rbi_rates_fetcher import (
    build_rates_parquet,
    load_rbi_tbill_from_csv,
)

SEED = Path(__file__).resolve().parents[1] / "data" / "manual" / "rbi_rates_seed.csv"


def test_load_rbi_tbill_from_csv():
    df = load_rbi_tbill_from_csv(SEED)
    assert "tbill_91d" in df.columns
    assert df["tbill_91d"].max() <= 1.0  # converted to decimal


def test_build_rates_parquet(tmp_path):
    out = tmp_path / "rates.parquet"
    df = build_rates_parquet(SEED, out, "2023-01-01", "2023-01-31")
    assert len(df) > 0
    assert out.exists()
    assert df["tbill_91d"].notna().all()
