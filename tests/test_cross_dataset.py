"""Tests for cross-dataset validation."""

from __future__ import annotations

import pandas as pd

from nope_in.data.validators.schema_validator import (
    validate_cross_dataset_consistency,
    validate_options_chain,
    validate_rates,
)


def test_validate_options_chain_pass():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=5, freq="B"),
            "symbol": ["NIFTY"] * 5,
            "expiry_date": pd.Timestamp("2023-01-12"),
            "strike": [18000.0] * 5,
            "option_type": ["CE"] * 5,
            "settle_price": [164.0] * 5,
        }
    )
    result = validate_options_chain(df)
    assert result.passed


def test_validate_cross_dataset():
    opt_dates = pd.bdate_range("2023-01-02", periods=5)
    options = pd.DataFrame({"date": opt_dates, "settle_price": [1.0] * 5})
    underlying = pd.DataFrame({"date": opt_dates, "close": [18000.0] * 5})
    vix = pd.DataFrame({"date": opt_dates, "vix_close": [14.0] * 5})

    result = validate_cross_dataset_consistency(options, underlying, vix)
    assert result.passed


def test_validate_rates():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-01", periods=10),
            "tbill_91d": [0.067] * 10,
        }
    )
    result = validate_rates(df, "2023-01-01", "2023-01-31")
    assert result.passed
