"""
Feature engineering pipeline orchestrator.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from nope_in.features.bsm import compute_bsm_residual
from nope_in.features.feature_store import (
    build_feature_matrix,
    create_train_val_test_splits,
    fit_and_apply_scaler,
    get_all_feature_columns,
    save_feature_metadata,
)
from nope_in.features.surface_cnn import encode_surfaces_to_parquet, pretrain_surface_encoder
from nope_in.features.vol_surface import build_vol_surface_parquet

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(cfg: DictConfig, key: str) -> Path:
    return PROJECT_ROOT / cfg.data.paths[key]


def run_features_pipeline(cfg: DictConfig) -> None:
    """Execute Phase 2 feature engineering end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    processed_dir = _resolve_path(cfg, "processed_dir")
    features_dir = PROJECT_ROOT / "data" / "features"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    features_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    options_path = processed_dir / "options_chain.parquet"
    rates_path = processed_dir / "rates.parquet"
    underlying_path = processed_dir / "underlying_ohlcv.parquet"
    vix_path = processed_dir / "india_vix.parquet"
    events_path = processed_dir / "events_calendar.parquet"

    for p in [options_path, rates_path, underlying_path, vix_path, events_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    options_df = pd.read_parquet(processed_dir / "options_chain.parquet")
    rates_df = pd.read_parquet(rates_path)
    underlying_df = pd.read_parquet(underlying_path)
    vix_df = pd.read_parquet(vix_path)
    events_df = pd.read_parquet(events_path)

    regime_path = processed_dir / "regime_labels.parquet"
    regime_df = pd.read_parquet(regime_path) if regime_path.exists() else None

    # 1. BSM enrichment
    log.info("Computing BSM residuals and Greeks…")
    bsm_df = compute_bsm_residual(options_df, rates_df, underlying_df=underlying_df)
    bsm_out = processed_dir / "options_chain_bsm.parquet"
    bsm_df.to_parquet(bsm_out, index=False)

    # 2. Vol surface
    log.info("Building IV surfaces…")
    surface_df = build_vol_surface_parquet(
        bsm_df,
        processed_dir / "vol_surface.parquet",
        underlying_df=underlying_df,
    )

    # 3. CNN encoder pre-training + embedding
    from nope_in.features.surface_cnn import _extract_surface_matrix

    if len(surface_df) >= 10:
        matrices = np.stack([_extract_surface_matrix(row) for _, row in surface_df.iterrows()])
        valid_rows = np.isfinite(matrices).any(axis=(1, 2))
        if valid_rows.sum() >= 10:
            log.info("Pre-training IV surface CNN encoder…")
            encoder = pretrain_surface_encoder(
                matrices[valid_rows],
                artifacts_dir / "surface_encoder.pt",
                epochs=cfg.features.get("cnn_epochs", 200),
            )
            surface_df = encode_surfaces_to_parquet(
                surface_df,
                encoder,
                processed_dir / "vol_surface.parquet",
            )

    # 4. Feature matrix + splits
    log.info("Assembling feature matrix…")
    feature_groups = cfg.features.get("feature_groups", None)
    groups = list(feature_groups) if feature_groups else None

    matrix = build_feature_matrix(
        bsm_df,
        underlying_df,
        vix_df,
        rates_df,
        events_df,
        surface_df,
        regime_df,
        feature_groups=groups,
    )

    train_df, val_df, test_df = create_train_val_test_splits(
        matrix,
        train_end=cfg.features.get("train_end", "2021-12-31"),
        val_end=cfg.features.get("val_end", "2022-12-31"),
    )

    feature_cols = get_all_feature_columns(groups)
    feature_cols = [c for c in feature_cols if c in matrix.columns]

    train_scaled, val_scaled, test_scaled = fit_and_apply_scaler(
        train_df,
        val_df,
        test_df,
        feature_cols,
        artifacts_dir / "feature_scaler.pkl",
    )

    train_scaled.to_parquet(features_dir / "train.parquet", index=False)
    val_scaled.to_parquet(features_dir / "val.parquet", index=False)
    test_scaled.to_parquet(features_dir / "test.parquet", index=False)
    save_feature_metadata(feature_cols, artifacts_dir / "feature_metadata.json", groups)

    log.info("Phase 2 complete. Features in %s", features_dir)


def run_bsm_only(cfg: DictConfig | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = _load_default_config()
    processed_dir = _resolve_path(cfg, "processed_dir")
    options_df = pd.read_parquet(processed_dir / "options_chain.parquet")
    rates_df = pd.read_parquet(processed_dir / "rates.parquet")
    underlying_df = pd.read_parquet(processed_dir / "underlying_ohlcv.parquet")
    bsm_df = compute_bsm_residual(options_df, rates_df, underlying_df=underlying_df)
    out = processed_dir / "options_chain_bsm.parquet"
    bsm_df.to_parquet(out, index=False)
    return bsm_df


def _load_default_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_features_pipeline(cfg)


if __name__ == "__main__":
    main()
