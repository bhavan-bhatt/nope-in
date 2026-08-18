"""Smoke test for Phase 3 regime HMM (fast fit on subset)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nope_in.regime.hmm import fit_hmm, label_regimes, prepare_hmm_observations
from nope_in.regime.pipeline import run_regime_pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    processed = ROOT / "data" / "processed"
    underlying = pd.read_parquet(processed / "underlying_ohlcv.parquet")
    vix = pd.read_parquet(processed / "india_vix.parquet")

    X, dates, scaler, aligned = prepare_hmm_observations(underlying, vix, symbol="NIFTY")
    fit_mask = pd.to_datetime(dates) <= "2021-12-31"
    X_train = X[fit_mask]

    model = fit_hmm(X_train, n_states=4, n_restarts=3, n_iter=50, random_seed=42)
    labels = label_regimes(model, X, dates, "NIFTY", scaler, aligned)

    assert len(labels) == len(dates)
    assert set(labels.columns) >= {
        "date",
        "symbol",
        "regime_id",
        "regime_p0",
        "regime_p1",
        "regime_p2",
        "regime_p3",
        "regime_name",
    }
    prob_sum = labels[["regime_p0", "regime_p1", "regime_p2", "regime_p3"]].sum(axis=1)
    assert ((prob_sum - 1.0).abs() < 1e-5).all(), "Posterior probabilities must sum to 1"

    log.info("Unit smoke passed — %d days, regimes:\n%s", len(labels), labels["regime_name"].value_counts())

    config_dir = str(ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.regime.n_restarts = 3
    cfg.regime.n_iter = 50
    run_regime_pipeline(cfg)
    log.info("Pipeline smoke passed.")


if __name__ == "__main__":
    main()
