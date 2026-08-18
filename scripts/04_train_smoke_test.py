#!/usr/bin/env python3
"""Phase 4 smoke test — synthetic mini-dataset, 3-epoch train."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nope_in.models.nope import NOPEIndia
from nope_in.training.dataset import NOPEDataModule, get_regime_feature_indices
from nope_in.training.pipeline import build_model
from nope_in.training.trainer import NOPELightningModule

log = logging.getLogger(__name__)


def _synthetic_frame(n: int, feature_cols: list[str], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in feature_cols}

    for col in ("regime_p0", "regime_p1", "regime_p2", "regime_p3"):
        if col in data:
            data[col] = rng.uniform(0, 1, size=n)
    if all(c in data for c in ("regime_p0", "regime_p1", "regime_p2", "regime_p3")):
        probs = np.stack([data[f"regime_p{i}"] for i in range(4)], axis=1)
        probs = probs / probs.sum(axis=1, keepdims=True)
        for i in range(4):
            data[f"regime_p{i}"] = probs[:, i]

    data["moneyness"] = rng.uniform(0.95, 1.05, size=n)
    data["vega"] = rng.uniform(1.0, 20.0, size=n)
    data["implied_vol"] = rng.uniform(0.12, 0.25, size=n)
    data["bsm_price"] = rng.uniform(50.0, 300.0, size=n)
    data["bsm_error"] = rng.normal(0, 2.0, size=n)
    data["settle_price"] = data["bsm_price"] + data["bsm_error"]
    data["spot"] = rng.uniform(18000, 22000, size=n)
    data["strike"] = data["spot"] * data["moneyness"]
    data["T"] = rng.uniform(1 / 365, 30 / 365, size=n)
    data["rate_interpolated_dte"] = 0.065
    data["option_type"] = rng.choice(["CE", "PE"], size=n)
    data["is_ATM"] = (data["moneyness"] >= 0.98) & (data["moneyness"] <= 1.02)
    for col, flag in (
        ("is_DITM", data["moneyness"] < 0.90),
        ("is_ITM", (data["moneyness"] >= 0.90) & (data["moneyness"] < 0.98)),
        ("is_OTM", (data["moneyness"] > 1.02) & (data["moneyness"] <= 1.10)),
        ("is_DOTM", data["moneyness"] > 1.10),
    ):
        data[col] = flag

    return pd.DataFrame(data)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    meta_path = PROJECT_ROOT / "artifacts" / "feature_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run `make features-smoke` first.")

    meta = json.loads(meta_path.read_text())
    feature_cols = meta["feature_columns"]

    features_dir = PROJECT_ROOT / "data" / "features_smoke"
    features_dir.mkdir(parents=True, exist_ok=True)

    train_df = _synthetic_frame(512, feature_cols, seed=1)
    val_df = _synthetic_frame(256, feature_cols, seed=2)
    test_df = _synthetic_frame(256, feature_cols, seed=3)
    train_df.to_parquet(features_dir / "train.parquet", index=False)
    val_df.to_parquet(features_dir / "val.parquet", index=False)
    test_df.to_parquet(features_dir / "test.parquet", index=False)

    from hydra import compose, initialize_config_dir

    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
    cfg.training.features_dir = "data/features_smoke"
    cfg.training.batch_size = 128
    cfg.training.max_epochs = 3
    cfg.training.early_stopping_patience = 10
    cfg.training.cal_fraction = 0.5
    cfg.model.hidden_dim = 64

    datamodule = NOPEDataModule(
        features_dir=PROJECT_ROOT / cfg.training.features_dir,
        metadata_path=meta_path,
        batch_size=int(cfg.training.batch_size),
        cal_fraction=float(cfg.training.cal_fraction),
    )
    datamodule.setup()

    model = build_model(cfg, feature_cols)
    regime_indices = get_regime_feature_indices(feature_cols)
    assert regime_indices, "Expected regime feature indices"

    x = torch.randn(4, len(feature_cols))
    bsm = torch.randn(4)
    out = model(x, bsm)
    assert out["bsm_error_pred"].shape == (4,)
    assert out["expert_outputs"].shape == (4, model.n_experts)

    lightning_module = NOPELightningModule(model, OmegaConf.to_container(cfg.training, resolve=True))
    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        gradient_clip_val=1.0,
    )
    trainer.fit(lightning_module, datamodule=datamodule)

    cal_loader = datamodule.cal_dataloader()
    if cal_loader is not None:
        quantiles = lightning_module.calibrate_conformal(cal_loader)
        assert "global" in quantiles

    trainer.test(lightning_module, datamodule=datamodule)
    log.info("Phase 4 smoke test passed.")


if __name__ == "__main__":
    main()
