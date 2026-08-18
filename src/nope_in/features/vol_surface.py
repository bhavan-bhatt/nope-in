"""
Implied Volatility Surface Constructor.

Builds 4×5 IV surface grids (tenor × moneyness) from options chain data,
with SABR smile interpolation for missing cells and surface-level statistics.
"""

from __future__ import annotations

import logging
from datetime import date
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares

log = logging.getLogger(__name__)

ArrayLike = Union[float, np.ndarray]
MIN_SIGMA = 0.001

TENORS = ["W1", "W2", "M1", "M2"]
MONEYNESS_BUCKETS = [0.90, 0.95, 1.00, 1.05, 1.10]
DEFAULT_BUCKET_LEVELS = MONEYNESS_BUCKETS


def assign_moneyness_bucket(
    moneyness: ArrayLike,
    bucket_levels: list[float] | None = None,
    tolerance: float = 0.02,
) -> ArrayLike:
    """Assign each option to the nearest standard moneyness bucket (S/K ratio)."""
    levels = bucket_levels or DEFAULT_BUCKET_LEVELS
    m_arr = np.asarray(moneyness, dtype=float)
    level_arr = np.array(levels, dtype=float)

    dist = np.abs(m_arr[..., None] - level_arr[None, ...])
    nearest_idx = dist.argmin(axis=-1)
    nearest_dist = dist.min(axis=-1)
    buckets = level_arr[nearest_idx].astype(float)
    buckets = np.where(nearest_dist <= tolerance, buckets, np.nan)
    if np.ndim(buckets) == 0:
        return float(buckets)
    return buckets


def _assign_expiry_tenor(group: pd.DataFrame) -> pd.Series:
    """Map each expiry to W1/W2/M1/M2 based on sorted unique expiries on a date."""
    expiries = sorted(group["expiry_date"].unique())
    tenor_map = {}
    for i, exp in enumerate(expiries[:4]):
        tenor_map[exp] = TENORS[i]
    return group["expiry_date"].map(tenor_map)


def _sabr_vol_hagan(
    strike: np.ndarray,
    forward: float,
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Hagan et al. (2002) SABR implied vol approximation."""
    k = np.asarray(strike, dtype=float)
    f = max(forward, 1e-8)
    t = max(expiry, 1e-8)

    if abs(f - k).max() < 1e-8:
        fk_beta = f ** (beta - 1.0)
        return np.full_like(k, alpha / fk_beta, dtype=float)

    fk_beta = (f * k) ** ((1.0 - beta) / 2.0)
    log_fk = np.log(f / k)
    z = (nu / alpha) * fk_beta * log_fk
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho + 1e-12))
    sigma = (
        alpha
        / (fk_beta * (1.0 + ((1.0 - beta) ** 2 / 24.0) * log_fk**2 + ((1.0 - beta) ** 4 / 1920.0) * log_fk**4))
        * (z / (xz + 1e-12))
    )
    corr = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * alpha**2 / (fk_beta**2)
        + (rho * beta * nu * alpha) / (4.0 * fk_beta)
        + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
    ) * t
    return np.maximum(sigma * corr, MIN_SIGMA)


def sabr_interpolate(
    known_strikes: np.ndarray,
    known_ivs: np.ndarray,
    target_strikes: np.ndarray,
    F: float,
    T: float,
    alpha_init: float = 0.3,
    beta: float = 0.7,
    rho_init: float = -0.3,
    nu_init: float = 0.4,
) -> np.ndarray:
    """
    Fit SABR to known (strike, IV) pairs and interpolate to target strikes.
    Falls back to flat extrapolation when fewer than 3 points are available.
    """
    ks = np.asarray(known_strikes, dtype=float)
    ivs = np.asarray(known_ivs, dtype=float)
    targets = np.asarray(target_strikes, dtype=float)

    valid = np.isfinite(ks) & np.isfinite(ivs) & (ivs > 0)
    ks, ivs = ks[valid], ivs[valid]

    if len(ks) == 0:
        return np.full_like(targets, np.nan, dtype=float)
    if len(ks) < 3:
        nearest = ks[np.argmin(np.abs(ks - targets.mean()))]
        fill = ivs[np.argmin(np.abs(ks - nearest))]
        return np.full_like(targets, fill, dtype=float)

    x0 = np.array([alpha_init, rho_init, nu_init], dtype=float)
    bounds = ([MIN_SIGMA, -0.99, 1e-4], [5.0, 0.99, 5.0])

    def residuals(params: np.ndarray) -> np.ndarray:
        alpha, rho, nu = params
        model = _sabr_vol_hagan(ks, F, T, alpha, beta, rho, nu)
        return model - ivs

    try:
        result = least_squares(residuals, x0, bounds=bounds, max_nfev=500)
        alpha, rho, nu = result.x
        return _sabr_vol_hagan(targets, F, T, alpha, beta, rho, nu)
    except Exception:
        fill = float(np.median(ivs))
        return np.full_like(targets, fill, dtype=float)


def construct_daily_surface(
    options_df: pd.DataFrame,
    date: date,
    symbol: str,
    bucket_levels: list[float] | None = None,
) -> pd.DataFrame:
    """
    Construct a 4×5 IV surface grid for one (date, symbol).
    Rows = tenors (W1–M2), columns = moneyness buckets.
    """
    levels = bucket_levels or DEFAULT_BUCKET_LEVELS
    day = pd.Timestamp(date).normalize()
    sym = symbol.upper()

    df = options_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    subset = df[(df["date"] == day) & (df["symbol"] == sym)].copy()
    if subset.empty or "implied_vol" not in subset.columns:
        return pd.DataFrame(index=TENORS, columns=levels, dtype=float)

    if "moneyness" not in subset.columns:
        subset["moneyness"] = subset["spot"] / subset["strike"]
    subset["moneyness_bucket"] = assign_moneyness_bucket(subset["moneyness"].values, levels)
    subset["expiry_tenor"] = _assign_expiry_tenor(subset)
    subset = subset.dropna(subset=["moneyness_bucket", "expiry_tenor", "implied_vol"])

    grid = pd.DataFrame(index=TENORS, columns=levels, dtype=float)
    for tenor in TENORS:
        for bucket in levels:
            cell = subset[(subset["expiry_tenor"] == tenor) & (subset["moneyness_bucket"] == bucket)]
            if not cell.empty:
                grid.loc[tenor, bucket] = cell["implied_vol"].median()

    # SABR fill per tenor
    for tenor in TENORS:
        row = grid.loc[tenor]
        known_buckets = row.dropna().index.tolist()
        if not known_buckets:
            continue
        missing = [b for b in levels if b not in known_buckets]
        if not missing:
            continue

        tenor_df = subset[subset["expiry_tenor"] == tenor]
        if tenor_df.empty:
            continue
        F = float(tenor_df["spot"].median())
        T = float(tenor_df["T"].median()) if "T" in tenor_df.columns else float(tenor_df["dte"].median()) / 365.0
        known_strikes = (F / np.array(known_buckets)).astype(float)
        known_ivs = row[known_buckets].values.astype(float)
        target_strikes = (F / np.array(missing)).astype(float)
        filled = sabr_interpolate(known_strikes, known_ivs, target_strikes, F, T)
        for b, iv in zip(missing, filled):
            grid.loc[tenor, b] = iv

    return grid


def _surface_stats(
    grid: pd.DataFrame,
    realised_vol_21d: float | None = None,
) -> dict[str, float]:
    """Extract skew, term slope, curvature, and IV-RV spread from a surface grid."""
    stats: dict[str, float] = {
        "atm_iv": float("nan"),
        "iv_skew_25d": float("nan"),
        "iv_term_slope": float("nan"),
        "iv_smile_curvature": float("nan"),
        "iv_rv_spread": float("nan"),
    }
    if grid.empty:
        return stats

    atm_col = 1.00 if 1.00 in grid.columns else grid.columns[len(grid.columns) // 2]
    if "W1" in grid.index and pd.notna(grid.loc["W1", atm_col]):
        stats["atm_iv"] = float(grid.loc["W1", atm_col])

    if 0.95 in grid.columns and 1.05 in grid.columns and "W1" in grid.index:
        put_iv = grid.loc["W1", 0.95]
        call_iv = grid.loc["W1", 1.05]
        if pd.notna(put_iv) and pd.notna(call_iv):
            stats["iv_skew_25d"] = float(put_iv - call_iv)

    if "W1" in grid.index and "M1" in grid.index and pd.notna(grid.loc["W1", atm_col]) and pd.notna(
        grid.loc["M1", atm_col]
    ):
        stats["iv_term_slope"] = float(grid.loc["M1", atm_col] - grid.loc["W1", atm_col])

    if 0.90 in grid.columns and 1.10 in grid.columns and "W1" in grid.index:
        wing_avg = np.nanmean([grid.loc["W1", 0.90], grid.loc["W1", 1.10]])
        if pd.notna(wing_avg) and pd.notna(stats["atm_iv"]):
            stats["iv_smile_curvature"] = float(wing_avg - stats["atm_iv"])

    if pd.notna(stats["atm_iv"]) and realised_vol_21d is not None and pd.notna(realised_vol_21d):
        stats["iv_rv_spread"] = float(stats["atm_iv"] - realised_vol_21d)

    return stats


def _build_one_surface_row(
    options_df: pd.DataFrame,
    day: date,
    symbol: str,
    realised_vol_lookup: dict[tuple[pd.Timestamp, str], float] | None,
) -> dict:
    grid = construct_daily_surface(options_df, day, symbol)
    rv = None
    if realised_vol_lookup is not None:
        rv = realised_vol_lookup.get((pd.Timestamp(day).normalize(), symbol.upper()))
    stats = _surface_stats(grid, rv)

    row: dict = {"date": pd.Timestamp(day).normalize(), "symbol": symbol.upper()}
    row.update(stats)
    for ti, tenor in enumerate(TENORS):
        for bi, bucket in enumerate(MONEYNESS_BUCKETS):
            val = grid.loc[tenor, bucket] if tenor in grid.index and bucket in grid.columns else np.nan
            row[f"surface_{tenor}_{bucket:.2f}"] = val
    return row


def build_vol_surface_parquet(
    options_df: pd.DataFrame,
    output_path: Path,
    underlying_df: pd.DataFrame | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Build vol_surface.parquet for every (date, symbol) in parallel.
    """
    df = options_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "implied_vol" not in df.columns:
        raise ValueError("options_df must contain implied_vol — run compute_bsm_residual first")

    rv_lookup: dict[tuple[pd.Timestamp, str], float] | None = None
    if underlying_df is not None and "realised_vol_21d" in underlying_df.columns:
        und = underlying_df.copy()
        und["date"] = pd.to_datetime(und["date"]).dt.normalize()
        rv_lookup = {
            (row["date"], row["symbol"].upper()): float(row["realised_vol_21d"])
            for _, row in und.dropna(subset=["realised_vol_21d"]).iterrows()
        }

    keys = df.groupby(["date", "symbol"]).size().reset_index()[["date", "symbol"]]
    worker = partial(_build_one_surface_row, df, realised_vol_lookup=rv_lookup)
    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(worker)(row["date"].date(), row["symbol"]) for _, row in keys.iterrows()
    )

    surface_df = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    surface_df.to_parquet(output_path, index=False)
    log.info("Wrote vol surface: %d rows → %s", len(surface_df), output_path)
    return surface_df
