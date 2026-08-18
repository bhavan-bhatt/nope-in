"""
PyTorch Lightning Training Module for NOPE-IN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from nope_in.features.bsm import implied_vol_from_price
from nope_in.models.conformal import calibrate_mondrian, evaluate_coverage
from nope_in.models.nope import NOPEIndia
from nope_in.training.losses import atm_weighted_huber_loss


class NOPELightningModule(pl.LightningModule):
    """LightningModule wrapping NOPE-IN for structured training and evaluation."""

    def __init__(self, model: NOPEIndia, config: dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

        self.lambda_aux = float(config.get("lambda_aux", 0.1))
        self.huber_delta = float(config.get("huber_delta", 0.5))
        self.atm_sigma = float(config.get("atm_sigma", 0.05))
        self.atm_weight = float(config.get("atm_weight", 5.0))
        self.val_iv_every_n_epochs = int(config.get("val_iv_every_n_epochs", 5))

        self._val_sse = torch.tensor(0.0)
        self._val_count = 0
        self._val_atm_sse = torch.tensor(0.0)
        self._val_atm_count = 0
        self._val_regime_stats: dict[int, tuple[torch.Tensor, int]] = {}
        self._val_iv_sse = torch.tensor(0.0)
        self._val_iv_count = 0
        self._compute_iv_this_epoch = False

        self._test_preds: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self._test_price_preds: list[torch.Tensor] = []
        self._test_price_true: list[torch.Tensor] = []
        self._test_regime_ids: list[torch.Tensor] = []
        self._test_moneyness_buckets: list[str] = []

        self.conformal_quantiles: dict[int, float] | None = None

    def _compute_losses(
        self,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_true = batch["bsm_error"]
        y_pred = outputs["bsm_error_pred"]
        primary = atm_weighted_huber_loss(
            y_pred,
            y_true,
            batch["moneyness"],
            delta=self.huber_delta,
            atm_sigma=self.atm_sigma,
            atm_weight=self.atm_weight,
        )

        expert_outputs = outputs["expert_outputs"]
        aux_losses = [
            atm_weighted_huber_loss(
                expert_outputs[:, k],
                y_true,
                batch["moneyness"],
                delta=self.huber_delta,
                atm_sigma=self.atm_sigma,
                atm_weight=self.atm_weight,
            )
            for k in range(expert_outputs.shape[1])
        ]
        auxiliary = torch.stack(aux_losses).mean()
        total = primary + self.lambda_aux * auxiliary
        return total, primary, auxiliary

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["features"], batch["bsm_price"])
        loss, primary, auxiliary = self._compute_losses(batch, outputs)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_primary_loss", primary, on_step=False, on_epoch=True)
        self.log("train_aux_loss", auxiliary, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        device = self.device
        self._val_sse = torch.tensor(0.0, device=device)
        self._val_count = 0
        self._val_atm_sse = torch.tensor(0.0, device=device)
        self._val_atm_count = 0
        self._val_regime_stats = {}
        self._val_iv_sse = torch.tensor(0.0, device=device)
        self._val_iv_count = 0
        self._compute_iv_this_epoch = (
            self.val_iv_every_n_epochs > 0
            and self.current_epoch % self.val_iv_every_n_epochs == 0
        )

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(batch["features"], batch["bsm_price"])
        y_true = batch["bsm_error"]
        y_pred = outputs["bsm_error_pred"]
        sq_err = (y_pred - y_true) ** 2

        self._val_sse += sq_err.sum()
        self._val_count += y_true.numel()

        atm_mask = batch["is_atm"] > 0.5
        if atm_mask.any():
            self._val_atm_sse += sq_err[atm_mask].sum()
            self._val_atm_count += int(atm_mask.sum().item())

        for regime in batch["regime_id"].unique():
            regime_id = int(regime.item())
            mask = batch["regime_id"] == regime
            if not mask.any():
                continue
            regime_sse, regime_count = self._val_regime_stats.get(
                regime_id, (torch.tensor(0.0, device=y_pred.device), 0)
            )
            self._val_regime_stats[regime_id] = (
                regime_sse + sq_err[mask].sum(),
                regime_count + int(mask.sum().item()),
            )

        if self._compute_iv_this_epoch and "spot" in batch:
            iv_pred = self._price_to_iv(batch, outputs["final_price"])
            iv_true = batch["implied_vol"]
            valid = torch.isfinite(iv_pred) & torch.isfinite(iv_true)
            if valid.any():
                iv_sq_err = (iv_pred[valid] - iv_true[valid]) ** 2
                self._val_iv_sse += iv_sq_err.sum()
                self._val_iv_count += int(valid.sum().item())

    def on_validation_epoch_end(self) -> None:
        if self._val_count == 0:
            return

        rmse = torch.sqrt(self._val_sse / self._val_count)
        if self._val_atm_count > 0:
            atm_rmse = torch.sqrt(self._val_atm_sse / self._val_atm_count)
        else:
            atm_rmse = rmse

        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_rmse_atm", atm_rmse, prog_bar=True)

        for regime_id, (regime_sse, regime_count) in self._val_regime_stats.items():
            if regime_count == 0:
                continue
            regime_rmse = torch.sqrt(regime_sse / regime_count)
            self.log(f"val_rmse_regime_{regime_id}", regime_rmse)

        if self._val_iv_count > 0:
            iv_rmse = torch.sqrt(self._val_iv_sse / self._val_iv_count)
            self.log("val_iv_rmse", iv_rmse)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(batch["features"], batch["bsm_price"])
        self._test_preds.append(outputs["bsm_error_pred"].detach().cpu())
        self._test_targets.append(batch["bsm_error"].detach().cpu())
        self._test_price_preds.append(outputs["final_price"].detach().cpu())
        self._test_price_true.append(batch["settle_price"].detach().cpu())
        self._test_regime_ids.append(batch["regime_id"].detach().cpu())
        self._test_moneyness_buckets.extend(batch["moneyness_bucket"])

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        error_pred = torch.cat(self._test_preds).numpy()
        error_true = torch.cat(self._test_targets).numpy()
        price_pred = torch.cat(self._test_price_preds).numpy()
        price_true = torch.cat(self._test_price_true).numpy()
        regime_ids = torch.cat(self._test_regime_ids).numpy()

        test_rmse = float(np.sqrt(np.mean((error_pred - error_true) ** 2)))
        bsm_rmse = float(np.sqrt(np.mean(error_true**2)))
        price_rmse = float(np.sqrt(np.mean((price_pred - price_true) ** 2)))

        self.log("test_rmse", test_rmse)
        self.log("test_bsm_baseline_rmse", bsm_rmse)
        self.log("test_price_rmse", price_rmse)
        self.log("test_improvement_vs_bsm", bsm_rmse - test_rmse)

        if self.conformal_quantiles is not None:
            coverage = evaluate_coverage(
                price_true,
                price_pred,
                self.conformal_quantiles,
                regime_labels=regime_ids,
                moneyness_labels=np.array(self._test_moneyness_buckets),
            )
            self.log("test_conformal_coverage", float(coverage["coverage"]))
            self.log("test_conformal_interval_width", float(coverage["mean_interval_width"]))

        self._test_preds.clear()
        self._test_targets.clear()
        self._test_price_preds.clear()
        self._test_price_true.clear()
        self._test_regime_ids.clear()
        self._test_moneyness_buckets.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        trunk_params = list(self.model.trunk.parameters())
        expert_params = list(self.model.experts.parameters())

        lr_trunk = float(self.config.get("learning_rate_trunk", 1e-4))
        lr_experts = float(self.config.get("learning_rate_experts", 3e-4))
        weight_decay = float(self.config.get("weight_decay", 1e-4))

        optimizer = torch.optim.AdamW(
            [
                {"params": trunk_params, "lr": lr_trunk},
                {"params": expert_params, "lr": lr_experts},
            ],
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.config.get("lr_factor", 0.5)),
            patience=int(self.config.get("lr_patience", 5)),
            min_lr=float(self.config.get("min_lr", 1e-6)),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_rmse",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _price_to_iv(self, batch: dict[str, torch.Tensor], prices: torch.Tensor) -> torch.Tensor:
        spot = batch["spot"].detach().cpu().numpy()
        strike = batch["strike"].detach().cpu().numpy()
        t = batch["T"].detach().cpu().numpy()
        rate = batch["rate"].detach().cpu().numpy()
        option_types = np.asarray(batch["option_type"])
        price_np = prices.detach().cpu().numpy()

        iv = np.asarray(
            implied_vol_from_price(price_np, spot, strike, t, rate, option_types),
            dtype=np.float32,
        )
        return torch.as_tensor(iv, device=prices.device, dtype=prices.dtype)

    @torch.no_grad()
    def calibrate_conformal(self, cal_loader) -> dict[int, float]:
        """Run Mondrian conformal calibration on the held-out calibration split."""
        self.eval()
        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []
        regime_list: list[np.ndarray] = []

        for batch in cal_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(batch["features"], batch["bsm_price"])
            y_true_list.append(batch["settle_price"].cpu().numpy())
            y_pred_list.append(outputs["final_price"].cpu().numpy())
            regime_list.append(batch["regime_id"].cpu().numpy())

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        regimes = np.concatenate(regime_list)

        alpha = float(self.config.get("conformal_alpha", 0.10))
        min_samples = int(self.config.get("conformal_min_samples", 50))
        self.conformal_quantiles = calibrate_mondrian(
            y_true,
            y_pred,
            regimes,
            n_regimes=self.model.n_experts,
            alpha=alpha,
            min_samples=min_samples,
        )
        return self.conformal_quantiles
