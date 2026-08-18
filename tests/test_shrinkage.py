"""Unit tests for bucket-wise calibrated shrinkage (the do-no-harm guardrail)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nope_in.evaluation.shrinkage import (
    apply_bucket_shrinkage,
    bucket_labels,
    fit_bucket_shrinkage,
)


class TestFitBucketShrinkage:
    def test_perfect_prediction_gives_full_weight(self):
        true = np.array([1.0, 2.0, 3.0, 4.0] * 100)
        pred = true.copy()
        buckets = np.array(["A"] * 400)
        sm = fit_bucket_shrinkage(true, pred, buckets, min_samples=10)
        assert sm.factors["A"] == 1.0

    def test_uncorrelated_prediction_shrinks_toward_zero(self):
        rng = np.random.default_rng(0)
        true = rng.normal(0, 10, 1000)
        pred = rng.normal(0, 10, 1000)  # independent noise, no real signal
        buckets = np.array(["A"] * 1000)
        sm = fit_bucket_shrinkage(true, pred, buckets, min_samples=10)
        assert abs(sm.factors["A"]) < 0.3

    def test_small_bucket_falls_back_to_global(self):
        rng = np.random.default_rng(1)
        true = rng.normal(0, 5, 500)
        pred = 0.8 * true
        buckets = np.array(["BIG"] * 490 + ["TINY"] * 10)
        sm = fit_bucket_shrinkage(true, pred, buckets, min_samples=50)
        assert "TINY" not in sm.factors
        assert "BIG" in sm.factors

    def test_shrinkage_never_exceeds_one_or_goes_negative(self):
        rng = np.random.default_rng(2)
        true = rng.normal(0, 10, 500)
        pred = 3.0 * true + rng.normal(0, 20, 500)  # overshooting prediction
        buckets = np.array(["A"] * 500)
        sm = fit_bucket_shrinkage(true, pred, buckets, min_samples=10)
        assert 0.0 <= sm.factors["A"] <= 1.0


class TestApplyBucketShrinkage:
    def test_shrunk_sse_never_worse_than_bsm_or_raw_on_calibration_data(self):
        rng = np.random.default_rng(3)
        n = 2000
        true_correction = rng.normal(0, 10, n)
        # Model is good in bucket A, noise in bucket B.
        buckets = rng.choice(["A", "B"], size=n)
        pred_correction = np.where(
            buckets == "A",
            0.9 * true_correction + rng.normal(0, 1, n),
            rng.normal(0, 15, n),
        )
        bsm_price = rng.uniform(50, 200, n)
        settle = bsm_price + true_correction

        sm = fit_bucket_shrinkage(true_correction, pred_correction, buckets, min_samples=50)
        shrunk_price = apply_bucket_shrinkage(bsm_price, pred_correction, buckets, sm)

        bsm_sse = float(np.sum((settle - bsm_price) ** 2))
        raw_sse = float(np.sum((settle - (bsm_price + pred_correction)) ** 2))
        shrunk_sse = float(np.sum((settle - shrunk_price) ** 2))

        assert shrunk_sse <= bsm_sse + 1e-6
        assert shrunk_sse <= raw_sse + 1e-6


class TestBucketLabels:
    def test_builds_composite_key(self):
        metadata = pd.DataFrame(
            {
                "moneyness_bucket": ["ATM", "OTM"],
                "option_type": ["ce", "PE"],
                "dte": [3.0, 25.0],
            }
        )
        labels = bucket_labels(metadata)
        assert labels[0] == "ATM|dte_lt5|CE"
        assert labels[1] == "OTM|dte_gt20|PE"
