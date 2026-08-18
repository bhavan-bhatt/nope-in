"""
Custom Loss Functions for NOPE-IN.

ATM-weighted Huber loss for residual BSM error prediction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _huber_elementwise(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    return F.huber_loss(y_pred, y_true, reduction="none", delta=delta)


def atm_weighted_huber_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    moneyness: torch.Tensor,
    delta: float = 0.5,
    atm_sigma: float = 0.05,
    atm_weight: float = 5.0,
) -> torch.Tensor:
    """
    ATM-weighted Huber loss on BSM error predictions.

    ``moneyness`` should be log(S/K) with ATM centred at 0. If values look like
    S/K ratios (near 1.0), they are converted automatically.
    """
    log_m = moneyness
    if torch.isfinite(log_m).all() and log_m.abs().mean() > 0.5:
        log_m = torch.log(torch.clamp(moneyness, min=1e-6))

    weights = 1.0 + atm_weight * torch.exp(-0.5 * log_m.pow(2) / (atm_sigma**2))
    huber = _huber_elementwise(y_pred, y_true, delta)
    return (weights * huber).mean()


def magnitude_weighted_huber_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta: float = 8.0,
    weight_power: float = 1.0,
    weight_scale: float = 15.0,
    min_weight: float = 1.0,
) -> torch.Tensor:
    """
    Huber loss upweighted by target magnitude.

    The BSM-error target is heteroscedastic: near-zero for weekly/ATM contracts,
    tens of rupees for long-dated/DITM/deep-skew contracts. A flat small-delta
    Huber loss degenerates to plain L1 across that whole range and trains the
    network towards the conditional *median* rather than the mean — good for
    MAE, bad for RMSE (which is what BSM is actually benchmarked on). Weighting
    each sample by its own true-error magnitude keeps the loss quadratic-ish
    (via a larger ``delta``) while forcing gradient budget onto the large-error
    tail instead of being dominated by the abundant near-zero examples.
    """
    huber = _huber_elementwise(y_pred, y_true, delta)
    weights = min_weight + (y_true.abs() / weight_scale).pow(weight_power)
    return (weights * huber).mean()


def vega_normalised_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    vega: torch.Tensor,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Vega-normalised MSE — auxiliary vol-space loss."""
    denom = torch.clamp(vega.abs(), min=epsilon)
    return torch.mean(((y_pred - y_true) / denom) ** 2)
