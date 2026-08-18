"""Unit tests for NOPE-IN training losses."""

from __future__ import annotations

import torch

from nope_in.training.losses import atm_weighted_huber_loss, vega_normalised_loss


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


class TestVegaNormalisedLoss:
    def test_scales_by_vega(self):
        y_pred = torch.tensor([2.0, 4.0])
        y_true = torch.tensor([0.0, 0.0])
        vega = torch.tensor([1.0, 2.0])
        loss = vega_normalised_loss(y_pred, y_true, vega)
        assert torch.isclose(loss, torch.tensor(4.0))
