"""Tests for IV surface construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nope_in.features.vol_surface import assign_moneyness_bucket, construct_daily_surface, sabr_interpolate


def test_assign_moneyness_bucket():
    m = np.array([0.91, 1.00, 1.09, 1.50])
    buckets = assign_moneyness_bucket(m, tolerance=0.02)
    assert buckets[0] == 0.90
    assert buckets[1] == 1.00
    assert buckets[2] == 1.10
    assert np.isnan(buckets[3])


def test_sabr_interpolate_fallback():
    ks = np.array([20000.0, 22000.0])
    ivs = np.array([0.18, 0.16])
    targets = np.array([21000.0])
    out = sabr_interpolate(ks, ivs, targets, F=22000.0, T=0.1)
    assert len(out) == 1
    assert np.isfinite(out[0])


def test_construct_daily_surface():
    rows = []
    spot = 22000.0
    for i, strike in enumerate([19800, 20900, 22000, 23100, 24200]):
        rows.append(
            {
                "date": pd.Timestamp("2023-06-01"),
                "symbol": "NIFTY",
                "expiry_date": pd.Timestamp("2023-06-08"),
                "strike": float(strike),
                "option_type": "CE",
                "settle_price": 100.0,
                "underlying_value": spot,
                "spot": spot,
                "moneyness": spot / strike,
                "dte": 7,
                "T": 7 / 365,
                "implied_vol": 0.15 + i * 0.005,
            }
        )
    df = pd.DataFrame(rows)
    grid = construct_daily_surface(df, pd.Timestamp("2023-06-01").date(), "NIFTY")
    assert grid.shape == (4, 5)
    assert grid.loc["W1", 1.00] == pytest.approx(0.16, abs=0.01)
