"""Comprehensive BSM pricer unit tests — the most critical module in NOPE-IN."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nope_in.features.bsm import (
    bsm_d1_d2,
    bsm_greeks,
    bsm_price,
    compute_bsm_residual,
    implied_vol_from_price,
    interpolate_rate_to_dte,
    validate_put_call_parity,
)

# Reference parameters (NIFTY-like)
S = 22000.0
K = 22000.0
T = 30 / 365.0
R = 0.065
SIGMA = 0.15
Q = 0.013


def _manual_call_price(s, k, t, r, sigma, q=Q):
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def _manual_put_price(s, k, t, r, sigma, q=Q):
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)


class TestBSMPrice:
    def test_atm_call_matches_manual(self):
        price = bsm_price(S, K, T, R, SIGMA, "CE", q=Q)
        assert abs(price - _manual_call_price(S, K, T, R, SIGMA)) < 1e-6

    def test_atm_put_matches_manual(self):
        price = bsm_price(S, K, T, R, SIGMA, "PE", q=Q)
        assert abs(price - _manual_put_price(S, K, T, R, SIGMA)) < 1e-6

    def test_put_call_parity(self):
        call = bsm_price(S, K, T, R, SIGMA, "CE", q=Q)
        put = bsm_price(S, K, T, R, SIGMA, "PE", q=Q)
        forward = S * np.exp(-Q * T) - K * np.exp(-R * T)
        assert abs(call - put - forward) < 1e-4

    def test_expiry_returns_intrinsic(self):
        call = bsm_price(S, K - 100, 0.0, R, SIGMA, "CE")
        put = bsm_price(S, K + 100, 0.0, R, SIGMA, "PE")
        assert call == pytest.approx(S - (K - 100))
        assert put == pytest.approx((K + 100) - S)

    def test_zero_vol_returns_intrinsic(self):
        call = bsm_price(S, K - 500, T, R, 0.0, "CE")
        assert call == pytest.approx(max(S - (K - 500), 0.0))

    def test_vectorised(self):
        strikes = np.array([21000.0, 22000.0, 23000.0])
        prices = bsm_price(S, strikes, T, R, SIGMA, "CE")
        assert len(prices) == 3
        assert all(p > 0 for p in prices)


class TestBSMD1D2:
    def test_atm_d1_d2_relationship(self):
        d1, d2 = bsm_d1_d2(S, K, T, R, SIGMA, q=Q)
        assert d2 == pytest.approx(d1 - SIGMA * math.sqrt(T), rel=1e-6)

    def test_invalid_spot_raises(self):
        with pytest.raises(ValueError):
            bsm_d1_d2(-1, K, T, R, SIGMA)


class TestBSMGreeks:
    def test_delta_bounds(self):
        g = bsm_greeks(S, K, T, R, SIGMA, "CE", q=Q)
        assert 0 < g["delta"] < 1

        g_put = bsm_greeks(S, K, T, R, SIGMA, "PE", q=Q)
        assert -1 < g_put["delta"] < 0

    def test_gamma_positive(self):
        g = bsm_greeks(S, K, T, R, SIGMA, "CE")
        assert g["gamma"] > 0

    def test_theta_negative(self):
        g = bsm_greeks(S, K, T, R, SIGMA, "CE")
        assert g["theta"] < 0

    def test_vega_positive(self):
        g = bsm_greeks(S, K, T, R, SIGMA, "CE")
        assert g["vega"] > 0

    def test_all_greeks_present(self):
        g = bsm_greeks(S, K, T, R, SIGMA, "CE")
        assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho", "vanna", "volga"}


class TestImpliedVol:
    def test_roundtrip_atm_call(self):
        true_price = bsm_price(S, K, T, R, SIGMA, "CE", q=Q)
        iv = implied_vol_from_price(true_price, S, K, T, R, "CE", q=Q)
        assert abs(iv - SIGMA) < 1e-4

    def test_roundtrip_otm_put(self):
        k = S * 0.90
        true_price = bsm_price(S, k, T, R, SIGMA, "PE", q=Q)
        iv = implied_vol_from_price(true_price, S, k, T, R, "PE", q=Q)
        assert abs(iv - SIGMA) < 1e-3

    def test_zero_price_returns_nan(self):
        iv = implied_vol_from_price(0.0, S, K, T, R, "CE")
        assert np.isnan(iv)

    def test_expiry_returns_nan(self):
        iv = implied_vol_from_price(100.0, S, K, 0.0, R, "CE")
        assert np.isnan(iv)

    def test_iv_speed_1000_contracts(self):
        n = 1000
        prices = bsm_price(S, K, T, R, SIGMA, "CE") * np.ones(n)
        import time

        t0 = time.perf_counter()
        ivs = implied_vol_from_price(prices, S, K, T, R, "CE")
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0  # generous bound; spec target is 500ms
        assert np.allclose(ivs, SIGMA, atol=1e-3)


class TestRateInterpolation:
    def test_flat_rate(self):
        r = interpolate_rate_to_dte(91, 0.06, 0.06, 0.06)
        assert r == pytest.approx(0.06)

    def test_interpolation_midpoint(self):
        r = interpolate_rate_to_dte(136.5, 0.06, 0.07, 0.08)
        assert 0.06 < r < 0.07


class TestComputeBSMResidual:
    @pytest.fixture
    def sample_chain(self):
        rows = []
        for strike in [21000, 22000, 23000]:
            for opt in ["CE", "PE"]:
                price = bsm_price(S, strike, T, R, SIGMA, opt, q=Q)
                rows.append(
                    {
                        "date": pd.Timestamp("2023-06-01"),
                        "symbol": "NIFTY",
                        "expiry_date": pd.Timestamp("2023-07-01"),
                        "strike": float(strike),
                        "option_type": opt,
                        "settle_price": price,
                        "underlying_value": S,
                        "contracts": 100,
                        "open_interest": 500,
                    }
                )
        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_rates(self):
        return pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-06-01")],
                "tbill_91d": [R],
                "tbill_182d": [R + 0.001],
                "tbill_364d": [R + 0.002],
                "yield_spread_2y10y": [0.01],
            }
        )

    def test_enriches_columns(self, sample_chain, sample_rates):
        result = compute_bsm_residual(sample_chain, sample_rates)
        for col in ["implied_vol", "atm_iv", "bsm_price", "delta", "vega", "bsm_error", "moneyness"]:
            assert col in result.columns

    def test_bsm_error_near_zero_at_atm(self, sample_chain, sample_rates):
        """ATM rows match when settle is priced at flat vol (same as ATM IV input)."""
        result = compute_bsm_residual(sample_chain, sample_rates)
        atm = result[result["strike"] == 22000]
        assert atm["bsm_error"].abs().max() < 0.05

    def test_bsm_error_nonzero_with_smile(self, sample_chain, sample_rates):
        """OTM premium vs flat ATM-vol BSM produces meaningful bsm_error."""
        chain = sample_chain.copy()
        chain.loc[chain["strike"] == 23000, "settle_price"] *= 1.10
        result = compute_bsm_residual(chain, sample_rates)
        otm = result[result["strike"] == 23000]
        assert otm["bsm_error"].abs().max() > 1.0
        assert otm["bsm_error"].mean() > 0

    def test_filters_zero_liquidity(self, sample_chain, sample_rates):
        sample_chain.loc[0, "contracts"] = 0
        sample_chain.loc[0, "open_interest"] = 0
        result = compute_bsm_residual(sample_chain, sample_rates)
        assert len(result) < len(sample_chain)


class TestPutCallParityValidation:
    def test_no_violations_for_consistent_chain(self):
        rows = []
        for strike in [21000, 22000]:
            cp = bsm_price(S, strike, T, R, SIGMA, "CE", q=Q)
            pp = bsm_price(S, strike, T, R, SIGMA, "PE", q=Q)
            rows.extend(
                [
                    {
                        "date": pd.Timestamp("2023-06-01"),
                        "symbol": "NIFTY",
                        "expiry_date": pd.Timestamp("2023-07-01"),
                        "strike": float(strike),
                        "option_type": "CE",
                        "settle_price": cp,
                        "underlying_value": S,
                    },
                    {
                        "date": pd.Timestamp("2023-06-01"),
                        "symbol": "NIFTY",
                        "expiry_date": pd.Timestamp("2023-07-01"),
                        "strike": float(strike),
                        "option_type": "PE",
                        "settle_price": pp,
                        "underlying_value": S,
                    },
                ]
            )
        df = pd.DataFrame(rows)
        rates = pd.DataFrame({"date": [pd.Timestamp("2023-06-01")], "tbill_91d": [R], "tbill_182d": [R], "tbill_364d": [R]})
        enriched = compute_bsm_residual(df, rates)
        violations = validate_put_call_parity(enriched, tolerance_pct=0.05)
        assert len(violations) == 0
