"""Smoke test for Phase 2 feature pipeline (single month)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nope_in.features.bsm import compute_bsm_residual
from nope_in.features.feature_store import (
    build_feature_matrix,
    create_train_val_test_splits,
    get_all_feature_columns,
)
from nope_in.features.vol_surface import build_vol_surface_parquet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    options = pd.read_parquet(PROCESSED / "options_chain.parquet")
    options = options[options["date"] >= "2023-06-01"]
    options = options[options["date"] < "2023-07-01"]
    log.info("Smoke subset: %d option rows", len(options))

    rates = pd.read_parquet(PROCESSED / "rates.parquet")
    underlying = pd.read_parquet(PROCESSED / "underlying_ohlcv.parquet")
    vix = pd.read_parquet(PROCESSED / "india_vix.parquet")
    events = pd.read_parquet(PROCESSED / "events_calendar.parquet")

    bsm = compute_bsm_residual(options, rates, underlying_df=underlying)
    log.info("BSM enriched: %d rows, mean |error|=%.4f", len(bsm), bsm["bsm_error"].abs().mean())

    surface = build_vol_surface_parquet(bsm, PROCESSED / "vol_surface_smoke.parquet", underlying_df=underlying)
    log.info("Surfaces built: %d days", len(surface))

    matrix = build_feature_matrix(bsm, underlying, vix, rates, events, surface)
    log.info("Feature matrix: %d rows × %d cols", len(matrix), len(get_all_feature_columns()))

    train, val, test = create_train_val_test_splits(matrix)
    log.info("Splits — train=%d val=%d test=%d", len(train), len(val), len(test))
    log.info("Smoke test passed.")


if __name__ == "__main__":
    main()
