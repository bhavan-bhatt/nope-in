#!/usr/bin/env python3
"""Phase 5 smoke test — synthetic data, mini model, evaluation + SHAP."""

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

from nope_in.evaluation.metrics import build_eval_metadata, compute_iv_metrics, compute_pricing_metrics
from nope_in.evaluation.pipeline import load_checkpoint, run_evaluation_pipeline
from nope_in.training.dataset import NOPEDataModule
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

    data["moneyness"] = rng.uniform(0.85, 1.15, size=n)
    data["vega"] = rng.uniform(1.0, 20.0, size=n)
    data["implied_vol"] = rng.uniform(0.12, 0.25, size=n)
    data["bsm_price"] = rng.uniform(50.0, 300.0, size=n)
    data["bsm_error"] = rng.normal(0, 2.0, size=n)
    data["settle_price"] = data["bsm_price"] + data["bsm_error"]
    data["spot"] = rng.uniform(18000, 22000, size=n)
    data["strike"] = data["spot"] / data["moneyness"]
    data["T"] = rng.uniform(1 / 365, 30 / 365, size=n)
    data["dte"] = data["T"] * 365.0
    data["rate_interpolated_dte"] = 0.065
    data["option_type"] = rng.choice(["CE", "PE"], size=n)
    data["is_weekly_expiry"] = rng.choice([0.0, 1.0], size=n)
    data["is_monthly_expiry"] = 1.0 - data["is_weekly_expiry"]
    data["days_to_next_rbi_mpc"] = rng.integers(1, 60, size=n)
    data["is_rbi_week"] = (data["days_to_next_rbi_mpc"] <= 5).astype(float)
    data["is_ATM"] = (data["moneyness"] >= 0.98) & (data["moneyness"] <= 1.02)
    for col, flag in (
        ("is_DITM", data["moneyness"] < 0.90),
        ("is_ITM", (data["moneyness"] >= 0.90) & (data["moneyness"] < 0.98)),
        ("is_OTM", (data["moneyness"] > 1.02) & (data["moneyness"] <= 1.10)),
        ("is_DOTM", data["moneyness"] > 1.10),
    ):
        data[col] = flag.astype(float)

    return pd.DataFrame(data)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    meta_path = PROJECT_ROOT / "artifacts" / "feature_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run `make features-smoke` first.")

    meta = json.loads(meta_path.read_text())
    feature_cols = meta["feature_columns"]

    features_dir = PROJECT_ROOT / "data" / "features_eval_smoke"
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

    cfg.training.features_dir = "data/features_eval_smoke"
    cfg.training.batch_size = 128
    cfg.training.max_epochs = 3
    cfg.training.cal_fraction = 0.0
    cfg.model.hidden_dim = 64

    datamodule = NOPEDataModule(
        features_dir=PROJECT_ROOT / cfg.training.features_dir,
        metadata_path=meta_path,
        batch_size=int(cfg.training.batch_size),
        cal_fraction=0.0,
    )
    datamodule.setup()

    model = build_model(cfg, feature_cols)
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

    ckpt_path = PROJECT_ROOT / "artifacts" / "nope_eval_smoke.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "feature_columns": feature_cols,
            "conformal_quantiles": None,
        },
        ckpt_path,
    )

    eval_metadata = build_eval_metadata(test_df)
    metrics = compute_pricing_metrics(
        y_true=test_df["settle_price"].to_numpy(),
        y_pred=test_df["settle_price"].to_numpy(),
        bsm_price=test_df["bsm_price"].to_numpy(),
        settle_price=test_df["settle_price"].to_numpy(),
        metadata=eval_metadata,
    )
    assert metrics["global"]["rmse"] == 0.0

    iv_metrics = compute_iv_metrics(
        predicted_prices=test_df["settle_price"].to_numpy(),
        market_ivs=test_df["implied_vol"].to_numpy(),
        S=test_df["spot"].to_numpy(),
        K=test_df["strike"].to_numpy(),
        T=test_df["T"].to_numpy(),
        r=test_df["rate_interpolated_dte"].to_numpy(),
        option_type=test_df["option_type"].to_numpy(),
        metadata=eval_metadata,
    )
    assert "iv_rmse" in iv_metrics

    loaded_model, loaded_features, _ = load_checkpoint(ckpt_path)
    assert loaded_features == feature_cols
    assert loaded_model.count_parameters() > 0

    cfg.evaluation.checkpoint_path = "artifacts/nope_eval_smoke.pt"
    cfg.evaluation.features_dir = "data/features_eval_smoke"
    cfg.evaluation.output_dir = "artifacts/eval_smoke"
    cfg.evaluation.batch_size = 128
    cfg.evaluation.shap.enabled = True
    cfg.evaluation.shap.background_samples = 32
    cfg.evaluation.shap.explain_samples = 64

    result = run_evaluation_pipeline(cfg)
    assert Path(result["results_path"]).exists()
    plots_dir = PROJECT_ROOT / "artifacts" / "eval_smoke" / "plots"
    assert (plots_dir / "pricing_rmse_comparison.png").exists()
    assert (plots_dir / "global_feature_importance_bar.png").exists()

    log.info("Phase 5 smoke test passed.")


if __name__ == "__main__":
    main()
