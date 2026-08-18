"""Feature engineering: BSM, IV surface, CNN encoder, feature store."""

from nope_in.features.bsm import (
    bsm_d1_d2,
    bsm_greeks,
    bsm_price,
    compute_bsm_residual,
    implied_vol_from_price,
    interpolate_rate_to_dte,
    validate_put_call_parity,
)
from nope_in.features.feature_store import (
    FEATURE_GROUPS,
    TARGET_COLUMNS,
    build_feature_matrix,
    create_train_val_test_splits,
    fit_and_apply_scaler,
    get_all_feature_columns,
)
from nope_in.features.surface_cnn import IVSurfaceCNNEncoder, IVSurfaceAutoencoder, pretrain_surface_encoder
from nope_in.features.vol_surface import assign_moneyness_bucket, build_vol_surface_parquet, construct_daily_surface

__all__ = [
    "FEATURE_GROUPS",
    "TARGET_COLUMNS",
    "IVSurfaceAutoencoder",
    "IVSurfaceCNNEncoder",
    "assign_moneyness_bucket",
    "bsm_d1_d2",
    "bsm_greeks",
    "bsm_price",
    "build_feature_matrix",
    "build_vol_surface_parquet",
    "compute_bsm_residual",
    "construct_daily_surface",
    "create_train_val_test_splits",
    "fit_and_apply_scaler",
    "get_all_feature_columns",
    "implied_vol_from_price",
    "interpolate_rate_to_dte",
    "pretrain_surface_encoder",
    "validate_put_call_parity",
]
