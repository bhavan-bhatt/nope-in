"""
Bucket-wise calibrated shrinkage — the "do no harm" guardrail for NOPE-IN.

PURPOSE:
    NOPE-IN predicts a BSM-error correction. In some segments (currently:
    OTM, ATM, long-dated, PE — see artifacts/evaluation_results.json) the raw
    correction increases RMSE relative to plain BSM even though it improves
    MAE/MAPE globally, because the training loss optimises a robust,
    median-seeking objective while RMSE punishes the mean-squared tail.

    This module fits a single scalar shrinkage factor lambda in [0, 1] per
    (moneyness_bucket, dte_segment, option_type) bucket on the VALIDATION
    split, then applies it to predictions on the test split:

        shrunk_correction = lambda_bucket * raw_correction
        shrunk_price      = bsm_price + shrunk_correction

    lambda is the closed-form least-squares scalar regression of the true
    correction on the predicted correction, clipped to [0, 1]:

        lambda* = clip( sum(pred * true) / sum(pred * pred), 0, 1 )

    Clipping to [0, 1] means lambda=0 exactly reproduces the BSM baseline in
    that bucket and lambda=1 reproduces the raw model — the blend can never
    move further from the truth (in a least-squares sense) than either
    endpoint. On the validation set this guarantees the shrunk model's SSE is
    <= min(BSM SSE, raw NOPE-IN SSE) in every calibrated bucket. It is not a
    guarantee on the held-out test set (that would require the bucket's error
    distribution to be stationary from val to test), but with buckets defined
    on stable structural properties (moneyness, DTE, option side) rather than
    on time, this generalises far better than trusting the raw correction
    everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DTE_SEGMENTS = ("dte_lt5", "dte_5to20", "dte_gt20")


def _dte_segment(dte: np.ndarray) -> np.ndarray:
    seg = np.full(dte.shape, "dte_gt20", dtype=object)
    seg[dte < 5] = "dte_lt5"
    seg[(dte >= 5) & (dte <= 20)] = "dte_5to20"
    return seg


def bucket_labels(metadata: pd.DataFrame) -> np.ndarray:
    """Build shrinkage bucket keys from an eval-metadata frame (see metrics.build_eval_metadata)."""
    moneyness = metadata["moneyness_bucket"].astype(str).to_numpy()
    option_type = metadata["option_type"].astype(str).str.upper().to_numpy()
    dte = metadata["dte"].to_numpy(dtype=float)
    dte_seg = _dte_segment(dte)
    return np.array(
        [f"{m}|{d}|{o}" for m, d, o in zip(moneyness, dte_seg, option_type)],
        dtype=object,
    )


@dataclass
class ShrinkageMap:
    factors: dict[str, float]
    global_factor: float
    min_samples: int

    def lookup(self, bucket: str) -> float:
        return self.factors.get(bucket, self.global_factor)

    def to_dict(self) -> dict:
        return {
            "global_factor": self.global_factor,
            "min_samples": self.min_samples,
            "factors": self.factors,
        }


def fit_bucket_shrinkage(
    correction_true: np.ndarray,
    correction_pred: np.ndarray,
    buckets: np.ndarray,
    min_samples: int = 300,
) -> ShrinkageMap:
    """
    Fit the optimal least-squares shrinkage factor per bucket on held-out data.

    Buckets with fewer than ``min_samples`` observations fall back to the
    global factor to avoid overfitting noisy small-sample estimates.
    """
    correction_true = np.asarray(correction_true, dtype=float)
    correction_pred = np.asarray(correction_pred, dtype=float)

    def _scalar_shrink(true_: np.ndarray, pred_: np.ndarray) -> float:
        denom = float(np.sum(pred_ * pred_))
        if denom <= 1e-12:
            return 0.0
        lam = float(np.sum(pred_ * true_) / denom)
        return float(np.clip(lam, 0.0, 1.0))

    global_factor = _scalar_shrink(correction_true, correction_pred)

    factors: dict[str, float] = {}
    for bucket in np.unique(buckets):
        mask = buckets == bucket
        if mask.sum() < min_samples:
            continue
        factors[str(bucket)] = _scalar_shrink(correction_true[mask], correction_pred[mask])

    return ShrinkageMap(factors=factors, global_factor=global_factor, min_samples=min_samples)


def apply_bucket_shrinkage(
    bsm_price: np.ndarray,
    correction_pred: np.ndarray,
    buckets: np.ndarray,
    shrink_map: ShrinkageMap,
) -> np.ndarray:
    """Apply calibrated per-bucket shrinkage to raw corrections and return shrunk prices."""
    lam = np.array([shrink_map.lookup(str(b)) for b in buckets], dtype=float)
    return np.asarray(bsm_price, dtype=float) + lam * np.asarray(correction_pred, dtype=float)
