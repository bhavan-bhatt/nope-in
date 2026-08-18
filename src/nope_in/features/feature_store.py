"""
Feature Store — Master Feature Assembly.

Joins BSM, underlying, VIX, rates, events, vol surface, and regime features
into model-ready train/val/test parquet splits with fitted StandardScaler.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

FEATURE_GROUPS = {
    "contract_features": [
        "moneyness",
        "moneyness_squared",
        "log_dte",
        "sqrt_dte",
        "inv_dte",
        "is_weekly_expiry",
        "is_monthly_expiry",
        "expiry_week_of_month",
        "days_to_current_weekly_expiry",
    ],
    "bsm_greeks": [
        "delta",
        "gamma",
        "vega",
        "theta",
        "vanna",
        "volga",
    ],
    "vol_features": [
        "implied_vol",
        "realised_vol_5d",
        "realised_vol_21d",
        "realised_vol_63d",
        "garch_vol_1d",
        "iv_rv_spread",
    ],
    "surface_stats": [
        "atm_iv",
        "iv_skew_25d",
        "iv_term_slope",
        "iv_smile_curvature",
    ],
    "surface_cnn_embedding": [f"surface_emb_{i}" for i in range(64)],
    "vix_features": [
        "vix_close",
        "vix_pct_change",
        "vix_percentile_252d",
    ],
    "macro_features": [
        "tbill_91d",
        "yield_spread_2y10y",
        "rate_interpolated_dte",
    ],
    "event_features": [
        "days_to_next_rbi_mpc",
        "days_to_next_nse_monthly_expiry",
        "is_rbi_week",
        "is_budget_week",
        "is_expiry_week",
    ],
    "regime_features": [
        "regime_p0",
        "regime_p1",
        "regime_p2",
        "regime_p3",
    ],
}

TARGET_COLUMNS = {
    "primary": "bsm_error",
    "normalised": "bsm_error_norm",
    "price": "settle_price",
}

NECESSARY_FEATURES = {
    "moneyness",
    "log_dte",
    "implied_vol",
    "delta",
    "vega",
    "vix_close",
    "rate_interpolated_dte",
    "bsm_error",
}

BOOLEAN_FEATURES = {
    "is_weekly_expiry",
    "is_monthly_expiry",
    "is_rbi_week",
    "is_budget_week",
    "is_expiry_week",
    "is_DITM",
    "is_ITM",
    "is_ATM",
    "is_OTM",
    "is_DOTM",
}

INTERACTION_FEATURES = [
    "moneyness_x_log_dte",
    "delta_x_vix",
    "vol_x_vix",
]


def _annualised_realised_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=max(2, window // 2)).std() * math.sqrt(252)


def _ewma_garch_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * 252)


def enrich_underlying_features(underlying_df: pd.DataFrame) -> pd.DataFrame:
    """Add realised vol and GARCH-style EWMA vol per symbol."""
    frames = []
    for symbol, grp in underlying_df.groupby("symbol"):
        g = grp.sort_values("date").copy()
        if "returns_1d" not in g.columns:
            g["returns_1d"] = np.log(g["close"] / g["close"].shift(1))
        g["realised_vol_5d"] = _annualised_realised_vol(g["returns_1d"], 5)
        g["realised_vol_21d"] = _annualised_realised_vol(g["returns_1d"], 21)
        g["realised_vol_63d"] = _annualised_realised_vol(g["returns_1d"], 63)
        g["garch_vol_1d"] = _ewma_garch_vol(g["returns_1d"])
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _add_moneyness_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m = out["moneyness"]
    out["is_DITM"] = m < 0.90
    out["is_ITM"] = (m >= 0.90) & (m < 0.98)
    out["is_ATM"] = (m >= 0.98) & (m <= 1.02)
    out["is_OTM"] = (m > 1.02) & (m <= 1.10)
    out["is_DOTM"] = m > 1.10
    return out


def _add_contract_calendar_features(df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expiry_week_of_month"] = pd.to_datetime(out["expiry_date"]).dt.isocalendar().week.astype(int) % 5
    if "days_to_current_weekly_expiry" not in out.columns:
        ev = events_df.copy()
        ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
        out = out.merge(
            ev[["date", "days_to_current_weekly_expiry"]],
            on="date",
            how="left",
        )
    return out


def build_feature_matrix(
    options_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    events_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    regime_df: pd.DataFrame | None = None,
    feature_groups: list[str] | None = None,
) -> pd.DataFrame:
    """
    Master join: options chain spine + all auxiliary feature tables.
    """
    groups = feature_groups or list(FEATURE_GROUPS.keys())
    selected_cols = []
    for g in groups:
        selected_cols.extend(FEATURE_GROUPS.get(g, []))

    df = options_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    n0 = len(df)

    und = enrich_underlying_features(underlying_df)
    und["date"] = pd.to_datetime(und["date"]).dt.normalize()
    df = df.merge(
        und[
            [
                c
                for c in ["date", "symbol", "realised_vol_5d", "realised_vol_21d", "realised_vol_63d", "garch_vol_1d"]
                if c in und.columns
            ]
        ],
        on=["date", "symbol"],
        how="left",
    )
    log.info("After underlying join: %d rows", len(df))

    vix = vix_df.copy()
    vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
    df = df.merge(
        vix[[c for c in ["date", "vix_close", "vix_pct_change", "vix_percentile_252d"] if c in vix.columns]],
        on="date",
        how="left",
    )

    rates = rates_df.copy()
    rates["date"] = pd.to_datetime(rates["date"]).dt.normalize()
    df = df.merge(
        rates[[c for c in ["date", "tbill_91d", "yield_spread_2y10y"] if c in rates.columns]],
        on="date",
        how="left",
    )

    events = events_df.copy()
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    event_cols = [
        c
        for c in [
            "date",
            "days_to_next_rbi_mpc",
            "days_to_next_nse_monthly_expiry",
            "days_to_current_weekly_expiry",
            "is_rbi_week",
            "is_budget_week",
            "is_expiry_week",
        ]
        if c in events.columns
    ]
    df = df.merge(events[event_cols], on="date", how="left")

    surf = surface_df.copy()
    surf["date"] = pd.to_datetime(surf["date"]).dt.normalize()
    surf_cols = ["date", "symbol"] + [c for c in surf.columns if c.startswith(("atm_iv", "iv_", "surface_emb_"))]
    df = df.merge(surf[surf_cols], on=["date", "symbol"], how="left")

    if regime_df is not None and not regime_df.empty:
        reg = regime_df.copy()
        reg["date"] = pd.to_datetime(reg["date"]).dt.normalize()
        reg_cols = ["date", "symbol"] + [c for c in reg.columns if c.startswith("regime_p")]
        df = df.merge(reg[reg_cols], on=["date", "symbol"], how="left")
    else:
        for col in FEATURE_GROUPS["regime_features"]:
            df[col] = 0.25

    df = _add_contract_calendar_features(df, events)
    df = _add_moneyness_dummies(df)

    if "iv_rv_spread" in selected_cols:
        if "iv_rv_spread" not in df.columns or df["iv_rv_spread"].isna().all():
            df["iv_rv_spread"] = df.get("atm_iv", df.get("implied_vol")) - df.get("realised_vol_21d")

    df["moneyness_x_log_dte"] = df["moneyness"] * df["log_dte"]
    df["delta_x_vix"] = df["delta"] * df["vix_close"]
    df["vol_x_vix"] = df["implied_vol"] * df["vix_close"]

    necessary_present = NECESSARY_FEATURES.intersection(df.columns)
    missing_necessary = NECESSARY_FEATURES - necessary_present
    if missing_necessary:
        raise ValueError(f"Missing necessary features after join: {sorted(missing_necessary)}")

    before = len(df)
    df = df.dropna(subset=list(NECESSARY_FEATURES.intersection(df.columns)))
    log.info("Dropped %d rows with missing necessary features (%d → %d)", before - len(df), before, len(df))

    optional_cols = set(selected_cols) - NECESSARY_FEATURES - BOOLEAN_FEATURES
    for col in optional_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    log.info("Feature matrix built: %d rows (from %d)", len(df), n0)
    return df.reset_index(drop=True)


def create_train_val_test_splits(
    feature_matrix: pd.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2022-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based train/val/test split (no random shuffling)."""
    df = feature_matrix.copy()
    df["date"] = pd.to_datetime(df["date"])
    train_cut = pd.Timestamp(train_end)
    val_cut = pd.Timestamp(val_end)

    train_df = df[df["date"] <= train_cut].copy()
    val_df = df[(df["date"] > train_cut) & (df["date"] <= val_cut)].copy()
    test_df = df[df["date"] > val_cut].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    log.info("Splits — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def fit_and_apply_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    scaler_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on train only; apply to all splits."""
    scale_cols = [
        c
        for c in feature_columns
        if c in train_df.columns
        and c not in BOOLEAN_FEATURES
        and not c.startswith("surface_emb_")
    ]

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    if scale_cols:
        scaler.fit(train_df[scale_cols])
        train_scaled[scale_cols] = scaler.transform(train_df[scale_cols])
        val_scaled[scale_cols] = scaler.transform(val_df[scale_cols])
        test_scaled[scale_cols] = scaler.transform(test_df[scale_cols])

    scaler_output_path = Path(scaler_output_path)
    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(scaler, scaler_output_path)
    log.info("Saved scaler (%d columns) → %s", len(scale_cols), scaler_output_path)
    return train_scaled, val_scaled, test_scaled


def save_feature_metadata(
    feature_columns: list[str],
    output_path: Path,
    feature_groups: list[str] | None = None,
) -> None:
    """Write feature metadata JSON for inference reproducibility."""
    meta = {
        "feature_columns": feature_columns,
        "feature_groups": feature_groups or list(FEATURE_GROUPS.keys()),
        "target_columns": TARGET_COLUMNS,
        "boolean_features": sorted(BOOLEAN_FEATURES),
        "n_features": len(feature_columns),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(meta, indent=2))
    log.info("Wrote feature metadata → %s", output_path)


def get_all_feature_columns(feature_groups: list[str] | None = None) -> list[str]:
    """Return flat list of model feature column names."""
    groups = feature_groups or list(FEATURE_GROUPS.keys())
    cols: list[str] = []
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])
    cols.extend(INTERACTION_FEATURES)
    cols.extend(["is_DITM", "is_ITM", "is_ATM", "is_OTM", "is_DOTM"])
    return cols
