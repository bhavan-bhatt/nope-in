"""
NOPE-IN — Neural Options Pricing Engine (India).
Core Model Architecture.

Mixture-of-Experts (MoE) residual learning network: predicts BSM pricing error
and adds it to the analytical BSM price for the final option price.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedTrunk(nn.Module):
    """
    Shared feature extraction network producing a hidden representation for all experts.

    LayerNorm → Linear → GELU → Dropout → Linear → GELU with a residual skip connection.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.fc1, self.fc2, self.proj):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        h = self.norm(x)
        h = self.dropout(F.gelu(self.fc1(h)))
        h = F.gelu(self.fc2(h))
        return residual + h


class ExpertNetwork(nn.Module):
    """
    One regime-specific expert: maps trunk output to a scalar BSM error estimate.

    Architecture: Linear(256, 128) → GELU → Linear(128, 64) → GELU → Linear(64, 1)
    """

    def __init__(self, input_dim: int = 256, name: str = ""):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, trunk_out: torch.Tensor) -> torch.Tensor:
        return self.net(trunk_out)


class NOPEIndia(nn.Module):
    """
    Regime-conditioned Mixture-of-Experts residual pricing network.

    Gating weights come from pre-trained HMM regime posterior probabilities
    embedded in the feature vector — they are not learned by the network.
    """

    def __init__(
        self,
        input_dim: int = 107,
        hidden_dim: int = 256,
        n_experts: int = 4,
        dropout: float = 0.1,
        regime_feature_indices: list[int] | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.regime_indices = regime_feature_indices

        self.trunk = SharedTrunk(input_dim, hidden_dim, dropout)
        self.experts = nn.ModuleList(
            [ExpertNetwork(hidden_dim, name=f"expert_{i}") for i in range(n_experts)]
        )

    def _extract_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.regime_indices is not None:
            gate = x[:, self.regime_indices]
        else:
            gate = x[:, -self.n_experts :]
        gate = torch.clamp(gate, min=0.0)
        gate_sum = gate.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return gate / gate_sum

    def forward(
        self,
        x: torch.Tensor,
        bsm_price: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        trunk_out = self.trunk(x)

        expert_outputs = torch.cat(
            [expert(trunk_out) for expert in self.experts],
            dim=1,
        )

        gate_weights = self._extract_gate_weights(x)
        bsm_error_pred = (expert_outputs * gate_weights).sum(dim=1)
        final_price = bsm_price + bsm_error_pred

        return {
            "bsm_error_pred": bsm_error_pred,
            "final_price": final_price,
            "expert_outputs": expert_outputs,
            "gate_weights": gate_weights,
        }

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        bsm_price: torch.Tensor,
        conformal_quantiles: dict[str, torch.Tensor] | dict[int, float],
    ) -> dict[str, torch.Tensor]:
        out = self.forward(x, bsm_price)
        gate_weights = out["gate_weights"]
        regime_id = gate_weights.argmax(dim=1)

        final_price = out["final_price"]
        lower = torch.empty_like(final_price)
        upper = torch.empty_like(final_price)
        filled = torch.zeros_like(final_price, dtype=torch.bool)

        for regime in range(self.n_experts):
            mask = regime_id == regime
            if not mask.any():
                continue
            q = conformal_quantiles.get(regime, conformal_quantiles.get(str(regime)))
            if q is None:
                continue
            q_tensor = torch.as_tensor(q, device=final_price.device, dtype=final_price.dtype)
            lower[mask] = final_price[mask] - q_tensor
            upper[mask] = final_price[mask] + q_tensor
            filled[mask] = True

        if not filled.all():
            fallback = conformal_quantiles.get("global", conformal_quantiles.get(0, 0.0))
            q_tensor = torch.as_tensor(fallback, device=final_price.device, dtype=final_price.dtype)
            lower[~filled] = final_price[~filled] - q_tensor
            upper[~filled] = final_price[~filled] + q_tensor

        return {
            "final_price": final_price,
            "lower_90": lower,
            "upper_90": upper,
            "regime_id": regime_id,
            "bsm_error_pred": out["bsm_error_pred"],
            "gate_weights": gate_weights,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
