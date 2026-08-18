"""
Phase 3 regime detection pipeline — fit HMM and write regime_labels.parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from nope_in.regime.hmm import (
    fit_hmm,
    label_regimes,
    plot_regime_timeline,
    prepare_hmm_observations,
    save_hmm_artifact,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(cfg: DictConfig, key: str) -> Path:
    return PROJECT_ROOT / cfg.data.paths[key]


def run_regime_pipeline(cfg: DictConfig) -> pd.DataFrame:
    """
    Fit HMM on pre-training window, label all dates, save artifacts.

    The HMM is fit once on data up to fit_end (default 2021-12-31) and applied
    to the full history without retraining — no look-ahead in walk-forward eval.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    processed_dir = _resolve_path(cfg, "processed_dir")
    artifacts_dir = PROJECT_ROOT / "artifacts"
    figures_dir = artifacts_dir / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    underlying_path = processed_dir / "underlying_ohlcv.parquet"
    vix_path = processed_dir / "india_vix.parquet"
    for p in (underlying_path, vix_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    underlying_df = pd.read_parquet(underlying_path)
    vix_df = pd.read_parquet(vix_path)

    regime_cfg = cfg.get("regime", {})
    symbols = list(regime_cfg.get("symbols", cfg.data.get("symbols", ["NIFTY"])))
    fit_start = pd.Timestamp(regime_cfg.get("fit_start", "2018-01-01"))
    fit_end = pd.Timestamp(regime_cfg.get("fit_end", cfg.features.get("train_end", "2021-12-31")))
    n_states = int(regime_cfg.get("n_states", 4))
    n_iter = int(regime_cfg.get("n_iter", 200))
    n_restarts = int(regime_cfg.get("n_restarts", 10))
    random_seed = int(regime_cfg.get("random_seed", 42))

    all_labels: list[pd.DataFrame] = []

    for symbol in symbols:
        log.info("=== Regime HMM for %s (fit %s → %s) ===", symbol, fit_start.date(), fit_end.date())

        _, dates, _, _ = prepare_hmm_observations(underlying_df, vix_df, symbol=symbol)
        fit_mask = (pd.to_datetime(dates) >= fit_start) & (pd.to_datetime(dates) <= fit_end)
        fit_mask_arr = np.asarray(fit_mask, dtype=bool)
        if fit_mask_arr.sum() < n_states * 10:
            raise ValueError(
                f"Insufficient training days for {symbol}: {fit_mask_arr.sum()} "
                f"(need ≥ {n_states * 10})"
            )

        X, dates, scaler_params, aligned = prepare_hmm_observations(
            underlying_df,
            vix_df,
            symbol=symbol,
            scaler_params=None,
            fit_mask=fit_mask_arr,
        )
        X_train = X[fit_mask_arr]

        model = fit_hmm(
            X_train,
            n_states=n_states,
            n_iter=n_iter,
            n_restarts=n_restarts,
            random_seed=random_seed,
            artifacts_dir=artifacts_dir / symbol.lower(),
        )

        artifact_path = artifacts_dir / f"hmm_{symbol.lower()}.pkl"
        save_hmm_artifact(
            model,
            scaler_params,
            artifact_path,
            metadata={
                "symbol": symbol,
                "fit_start": str(fit_start.date()),
                "fit_end": str(fit_end.date()),
                "n_states": n_states,
            },
        )

        labels = label_regimes(
            model,
            X,
            dates,
            symbol=symbol,
            scaler_params=scaler_params,
            aligned_frame=aligned,
        )
        all_labels.append(labels)

        plot_regime_timeline(labels, underlying_df, vix_df, figures_dir, symbol=symbol)

        log.info(
            "%s regime distribution:\n%s",
            symbol,
            labels["regime_name"].value_counts().to_string(),
        )

    regime_df = pd.concat(all_labels, ignore_index=True)
    out_path = processed_dir / "regime_labels.parquet"
    regime_df.to_parquet(out_path, index=False)
    log.info("Phase 3 complete. Regime labels → %s (%d rows)", out_path, len(regime_df))
    return regime_df


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_regime_pipeline(cfg)


if __name__ == "__main__":
    main()
