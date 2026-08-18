"""
Conformal Prediction Calibration for NOPE-IN.

Split conformal and Mondrian (regime-conditional) prediction intervals.
"""

from __future__ import annotations

import numpy as np


def compute_conformity_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Absolute conformity scores r_i = |y_i - ŷ_i|."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_true - y_pred)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    n = scores.size
    level = min(1.0, (n + 1) * (1.0 - alpha) / n)
    return float(np.quantile(scores, level, method="higher"))


def calibrate_global(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Global conformal quantile for (1-alpha) marginal coverage."""
    scores = compute_conformity_scores(y_true, y_pred)
    return _conformal_quantile(scores, alpha)


def calibrate_mondrian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_labels: np.ndarray,
    n_regimes: int = 4,
    alpha: float = 0.10,
    min_samples: int = 50,
) -> dict[int, float]:
    """
    Regime-conditional conformal quantiles with global fallback for sparse regimes.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    regime_labels = np.asarray(regime_labels, dtype=int)

    global_q = calibrate_global(y_true, y_pred, alpha=alpha)
    quantiles: dict[int, float] = {"global": global_q}

    for regime in range(n_regimes):
        mask = regime_labels == regime
        if mask.sum() < min_samples:
            quantiles[regime] = global_q
            continue
        scores = compute_conformity_scores(y_true[mask], y_pred[mask])
        q = _conformal_quantile(scores, alpha)
        quantiles[regime] = global_q if not np.isfinite(q) else q

    return quantiles


def evaluate_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: float | dict[int, float],
    regime_labels: np.ndarray | None = None,
    moneyness_labels: np.ndarray | None = None,
) -> dict[str, float | dict[str, float]]:
    """
    Evaluate empirical coverage and interval width on a test set.

    ``q`` may be a scalar (global) or regime-id → quantile mapping.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if isinstance(q, dict):
        lower = np.empty_like(y_pred)
        upper = np.empty_like(y_pred)
        regimes = np.asarray(regime_labels if regime_labels is not None else 0, dtype=int)
        fallback = q.get("global", q.get(0, 0.0))
        for regime_id, quantile in q.items():
            if regime_id == "global":
                continue
            mask = regimes == int(regime_id)
            if not np.any(mask):
                continue
            lower[mask] = y_pred[mask] - quantile
            upper[mask] = y_pred[mask] + quantile
        unfilled = ~np.isfinite(lower)
        if unfilled.any():
            lower[unfilled] = y_pred[unfilled] - fallback
            upper[unfilled] = y_pred[unfilled] + fallback
    else:
        lower = y_pred - q
        upper = y_pred + q
        fallback = float(q)

    covered = (y_true >= lower) & (y_true <= upper)
    mean_width = float(np.mean(upper - lower))

    result: dict[str, float | dict[str, float]] = {
        "coverage": float(np.mean(covered)),
        "mean_interval_width": mean_width,
    }

    if moneyness_labels is not None:
        labels = np.asarray(moneyness_labels)
        by_moneyness: dict[str, float] = {}
        for bucket in ("DITM", "ITM", "ATM", "OTM", "DOTM"):
            mask = labels == bucket
            if mask.any():
                by_moneyness[bucket] = float(np.mean(covered[mask]))
        result["coverage_by_moneyness"] = by_moneyness

    if regime_labels is not None:
        regimes = np.asarray(regime_labels, dtype=int)
        by_regime: dict[str, float] = {}
        for regime in np.unique(regimes):
            mask = regimes == regime
            if mask.any():
                by_regime[f"regime_{int(regime)}"] = float(np.mean(covered[mask]))
        result["coverage_by_regime"] = by_regime

    if isinstance(q, dict):
        result["global_quantile"] = fallback
    else:
        result["global_quantile"] = float(q)

    return result
