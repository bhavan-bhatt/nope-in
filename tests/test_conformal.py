"""Unit tests for conformal prediction utilities."""

from __future__ import annotations

import numpy as np

from nope_in.models.conformal import (
    calibrate_global,
    calibrate_mondrian,
    compute_conformity_scores,
    evaluate_coverage,
)


class TestConformityScores:
    def test_absolute_error(self):
        y_true = np.array([10.0, 12.0, 8.0])
        y_pred = np.array([9.0, 15.0, 8.5])
        scores = compute_conformity_scores(y_true, y_pred)
        np.testing.assert_allclose(scores, [1.0, 3.0, 0.5])


class TestCalibrateGlobal:
    def test_quantile_positive(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(100, 5, size=500)
        y_pred = y_true + rng.normal(0, 2, size=500)
        q = calibrate_global(y_true, y_pred, alpha=0.10)
        assert q > 0

    def test_coverage_on_calibration_like_data(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(50, 10, size=1000)
        noise = rng.normal(0, 1.5, size=1000)
        y_pred = y_true + noise
        q = calibrate_global(y_true, y_pred, alpha=0.10)
        metrics = evaluate_coverage(y_true, y_pred, q)
        assert metrics["coverage"] >= 0.88


class TestCalibrateMondrian:
    def test_sparse_regime_fallback(self):
        rng = np.random.default_rng(2)
        n = 300
        y_true = rng.normal(100, 5, size=n)
        y_pred = y_true + rng.normal(0, 2, size=n)
        regimes = np.array([0] * 280 + [3] * 20)
        quantiles = calibrate_mondrian(y_true, y_pred, regimes, n_regimes=4, min_samples=50)
        assert quantiles[3] == quantiles["global"]
        assert all(k in quantiles for k in range(4))


class TestEvaluateCoverage:
    def test_by_moneyness_and_regime(self):
        y_true = np.array([10.0, 11.0, 12.0, 13.0])
        y_pred = np.array([10.0, 11.5, 12.0, 14.0])
        q = 1.0
        buckets = np.array(["ATM", "ITM", "OTM", "DOTM"])
        regimes = np.array([0, 1, 2, 3])
        metrics = evaluate_coverage(y_true, y_pred, q, regime_labels=regimes, moneyness_labels=buckets)
        assert "coverage_by_moneyness" in metrics
        assert "coverage_by_regime" in metrics
        assert metrics["mean_interval_width"] == 2.0
