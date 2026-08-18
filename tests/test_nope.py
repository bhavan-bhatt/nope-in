"""Unit tests for NOPE-IN model architecture."""

from __future__ import annotations

import torch

from nope_in.models.nope import ExpertNetwork, NOPEIndia, SharedTrunk


class TestSharedTrunk:
    def test_output_shape(self):
        trunk = SharedTrunk(input_dim=35, hidden_dim=128)
        x = torch.randn(16, 35)
        out = trunk(x)
        assert out.shape == (16, 128)

    def test_gradient_flow(self):
        trunk = SharedTrunk(input_dim=35, hidden_dim=64)
        x = torch.randn(8, 35, requires_grad=True)
        out = trunk(x).sum()
        out.backward()
        assert x.grad is not None


class TestExpertNetwork:
    def test_scalar_output(self):
        expert = ExpertNetwork(input_dim=256)
        out = expert(torch.randn(32, 256))
        assert out.shape == (32, 1)


class TestNOPEIndia:
    def test_forward_shapes(self):
        regime_indices = [100, 101, 102, 103]
        model = NOPEIndia(input_dim=109, hidden_dim=256, regime_feature_indices=regime_indices)
        x = torch.randn(8, 109)
        x[:, regime_indices] = torch.softmax(torch.randn(8, 4), dim=1)
        bsm = torch.randn(8)
        out = model(x, bsm)
        assert out["bsm_error_pred"].shape == (8,)
        assert out["final_price"].shape == (8,)
        assert out["expert_outputs"].shape == (8, 4)
        assert out["gate_weights"].shape == (8, 4)
        assert torch.allclose(out["gate_weights"].sum(dim=1), torch.ones(8), atol=1e-5)

    def test_final_price_is_bsm_plus_error(self):
        model = NOPEIndia(input_dim=20, hidden_dim=32, n_experts=4, regime_feature_indices=[16, 17, 18, 19])
        x = torch.randn(4, 20)
        x[:, 16:20] = 0.25
        bsm = torch.tensor([10.0, 20.0, 30.0, 40.0])
        out = model(x, bsm)
        assert torch.allclose(out["final_price"], bsm + out["bsm_error_pred"])

    def test_parameter_count_order_of_magnitude(self):
        model = NOPEIndia(input_dim=107, hidden_dim=256, n_experts=4, regime_feature_indices=[103, 104, 105, 106])
        n_params = model.count_parameters()
        assert 200_000 <= n_params <= 350_000

    def test_predict_with_uncertainty(self):
        model = NOPEIndia(input_dim=20, hidden_dim=32, n_experts=4, regime_feature_indices=[16, 17, 18, 19])
        x = torch.randn(6, 20)
        x[:, 16:20] = torch.softmax(torch.randn(6, 4), dim=1)
        bsm = torch.randn(6)
        quantiles = {0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0, "global": 2.0}
        out = model.predict_with_uncertainty(x, bsm, quantiles)
        assert out["lower_90"].shape == (6,)
        assert out["upper_90"].shape == (6,)
        assert (out["upper_90"] >= out["final_price"]).all()
        assert (out["lower_90"] <= out["final_price"]).all()
