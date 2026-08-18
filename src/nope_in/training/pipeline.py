"""
NOPE-IN model training pipeline orchestrator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from nope_in.models.nope import NOPEIndia
from nope_in.training.dataset import NOPEDataModule, load_feature_metadata
from nope_in.training.trainer import NOPELightningModule
from nope_in.utils.device import device_label, resolve_lightning_accelerator

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(cfg: DictConfig, *parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def build_model(cfg: DictConfig, feature_columns: list[str]) -> NOPEIndia:
    from nope_in.training.dataset import get_regime_feature_indices

    model_cfg = cfg.model
    input_dim = len(feature_columns)
    configured = int(model_cfg.get("input_dim", input_dim))
    if configured != input_dim:
        log.warning(
            "cfg.model.input_dim=%d but feature metadata has %d columns; using %d",
            configured,
            input_dim,
            input_dim,
        )
    regime_indices = get_regime_feature_indices(feature_columns)

    return NOPEIndia(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        n_experts=int(model_cfg.get("n_experts", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        regime_feature_indices=regime_indices or None,
    )


def run_training_pipeline(cfg: DictConfig) -> dict:
    """Execute Phase 4 model training end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    features_dir = _resolve_path(cfg, cfg.training.features_dir)
    metadata_path = _resolve_path(cfg, cfg.training.metadata_path)
    artifacts_dir = _resolve_path(cfg, "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for path in (features_dir / "train.parquet", features_dir / "val.parquet", features_dir / "test.parquet", metadata_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required feature artifact missing: {path}. Run `make features` first."
            )

    meta = load_feature_metadata(metadata_path)
    feature_columns = meta["feature_columns"]

    datamodule = NOPEDataModule(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.get("num_workers", 0)),
        cal_fraction=float(cfg.training.get("cal_fraction", 0.5)),
    )
    datamodule.setup()

    model = build_model(cfg, feature_columns)
    log.info("NOPE-IN parameter count: %d", model.count_parameters())

    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_config, dict)
    lightning_module = NOPELightningModule(model, training_config)

    checkpoint_path = artifacts_dir / cfg.training.get("checkpoint_name", "nope_best.pt")
    callbacks = [
        EarlyStopping(
            monitor="val_rmse",
            patience=int(cfg.training.get("early_stopping_patience", 15)),
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=artifacts_dir,
            filename="nope_best",
            monitor="val_rmse",
            mode="min",
            save_top_k=1,
        ),
    ]

    accelerator = resolve_lightning_accelerator(str(cfg.training.get("device", "auto")))
    log.info("Training on %s (accelerator=%s)", device_label(torch.device(accelerator)), accelerator)

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.get("max_epochs", 200)),
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=float(cfg.training.get("gradient_clip", 1.0)),
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=int(cfg.training.get("log_every_n_steps", 50)),
    )

    mlflow_cfg = cfg.get("mlflow", {})
    experiment_name = mlflow_cfg.get("experiment_name", "nope-in")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=mlflow_cfg.get("run_name", "nope_train")):
        mlflow.log_params(
            {
                **{f"model.{k}": v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()},
                **{f"train.{k}": v for k, v in training_config.items()},
                "n_features": len(feature_columns),
                "n_parameters": model.count_parameters(),
            }
        )

        trainer.fit(lightning_module, datamodule=datamodule)

        cal_loader = datamodule.cal_dataloader()
        if cal_loader is not None:
            log.info("Calibrating Mondrian conformal intervals on validation holdout…")
            quantiles = lightning_module.calibrate_conformal(cal_loader)
            conformal_path = artifacts_dir / "conformal_quantiles.json"
            conformal_path.write_text(json.dumps({str(k): v for k, v in quantiles.items()}, indent=2))
            mlflow.log_artifact(str(conformal_path))
            for key, value in quantiles.items():
                mlflow.log_metric(f"conformal_q_{key}", float(value))

        log.info("Running sealed test evaluation…")
        trainer.test(lightning_module, datamodule=datamodule)

        best_ckpt = callbacks[1].best_model_path
        if best_ckpt:
            torch.save(
                {
                    "state_dict": lightning_module.model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "feature_columns": feature_columns,
                    "conformal_quantiles": lightning_module.conformal_quantiles,
                },
                checkpoint_path,
            )
            mlflow.log_artifact(str(checkpoint_path))

    log.info("Phase 4 complete. Best checkpoint → %s", checkpoint_path)
    return {"checkpoint": str(checkpoint_path)}


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training_pipeline(cfg)


if __name__ == "__main__":
    main()
