"""Unit tests for NOPE-IN training losses."""

from __future__ import annotations

import torch

from nope_in.training.losses import (
    atm_weighted_huber_loss,
    magnitude_weighted_huber_loss,
    vega_normalised_loss,
)


class TestATMWeightedHuberLoss:
    def test_atm_weighted_higher_than_otm(self):
        y_pred = torch.zeros(4)
        y_true = torch.ones(4)
        m_atm = torch.tensor([1.0, 1.0, 0.85, 0.85])
        loss = atm_weighted_huber_loss(y_pred, y_true, m_atm, delta=0.5, atm_weight=5.0)
        assert loss.item() > 0

        loss_atm = atm_weighted_huber_loss(
            torch.zeros(2),
            torch.ones(2),
            torch.tensor([1.0, 1.0]),
        )
        loss_otm = atm_weighted_huber_loss(
            torch.zeros(2),
            torch.ones(2),
            torch.tensor([0.75, 0.75]),
        )
        assert loss_atm > loss_otm

    def test_zero_error_zero_loss(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        loss = atm_weighted_huber_loss(y, y, torch.tensor([1.0, 1.01, 0.99]))
        assert loss.item() == 0.0


class TestMagnitudeWeightedHuberLoss:
    def test_zero_error_zero_loss(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        loss = magnitude_weighted_huber_loss(y, y)
        assert loss.item() == 0.0

    def test_large_target_weighted_more_than_small(self):
        # Same absolute prediction error (2.0), but one target is much larger
        # in magnitude — the loss should punish the large-target miss harder.
        small_target_loss = magnitude_weighted_huber_loss(
            torch.tensor([0.0]), torch.tensor([2.0]), delta=8.0, weight_scale=15.0
        )
        large_target_loss = magnitude_weighted_huber_loss(
            torch.tensor([58.0]), torch.tensor([60.0]), delta=8.0, weight_scale=15.0
        )
        assert large_target_loss.item() > small_target_loss.item()

    def test_behaves_quadratically_below_delta(self):
        # With a large delta, small errors should follow the MSE-like (not L1) regime.
        loss_1 = magnitude_weighted_huber_loss(torch.tensor([0.0]), torch.tensor([1.0]), delta=8.0)
        loss_2 = magnitude_weighted_huber_loss(torch.tensor([0.0]), torch.tensor([2.0]), delta=8.0)
        # Quadratic doubling of error should roughly quadruple loss magnitude
        # (allowing for the magnitude-weighting term), not merely double it.
        assert loss_2.item() > 3.0 * loss_1.item()


class TestVegaNormalisedLoss:
    def test_scales_by_vega(self):
        y_pred = torch.tensor([2.0, 4.0])
        y_true = torch.tensor([0.0, 0.0])
        vega = torch.tensor([1.0, 2.0])
        loss = vega_normalised_loss(y_pred, y_true, vega)
        assert torch.isclose(loss, torch.tensor(4.0))
