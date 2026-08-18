"""
Evaluation Metrics for NOPE-IN.

PURPOSE:
    Computes the complete set of evaluation metrics used to compare:
      BSM Baseline vs Heston vs XGBoost vs Plain MLP vs NOPE-IN (full)

    All metrics are computed on the held-out test set (2023 data — never seen during training).
    Results are saved to artifacts/evaluation_results.json and plotted to artifacts/plots/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nope_in.features.bsm import implied_vol_from_price
from nope_in.training.dataset import _moneyness_bucket

MONEYNESS_BUCKETS = ("DITM", "ITM", "ATM", "OTM", "DOTM")
REGIME_IDS = (0, 1, 2, 3)
OPTION_TYPES = ("CE", "PE")
DTE_SEGMENTS = ("dte_lt5", "dte_5to20", "dte_gt20")
EXPIRY_TYPES = ("weekly", "monthly")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err**2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, min_price: float = 1.0) -> float:
    mask = np.abs(y_true) >= min_price
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100.0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return _rmse(y_true[mask], y_pred[mask])


def _masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return _mae(y_true[mask], y_pred[mask])


def build_eval_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Build evaluation metadata frame from a feature parquet."""
    meta = pd.DataFrame(index=df.index)
    meta["moneyness_bucket"] = [_moneyness_bucket(row) for _, row in df.iterrows()]

    regime_cols = [c for c in df.columns if c.startswith("regime_p")]
    if regime_cols:
        meta["regime_id"] = df[regime_cols].to_numpy(dtype=np.float32).argmax(axis=1)
    else:
        meta["regime_id"] = 0

    meta["option_type"] = df["option_type"].astype(str).str.upper()
    if "dte" in df.columns:
        meta["dte"] = df["dte"].to_numpy(dtype=np.float32)
    elif "T" in df.columns:
        meta["dte"] = (df["T"].to_numpy(dtype=np.float32) * 365.0).astype(np.float32)
    else:
        meta["dte"] = np.nan

    for col in ("is_weekly_expiry", "is_monthly_expiry", "is_rbi_week", "days_to_next_rbi_mpc"):
        if col in df.columns:
            meta[col] = df[col].to_numpy()

    if "date" in df.columns:
        meta["date"] = df["date"].to_numpy()

    return meta


def compute_pricing_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bsm_price: np.ndarray,
    settle_price: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute pricing accuracy metrics globally and by segment.

    Global metrics include RMSE, MAE, MAPE, and R². Segmented metrics cover
    moneyness buckets, regimes, expiry types, DTE bands, and option types.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bsm_price = np.asarray(bsm_price, dtype=float)
    settle_price = np.asarray(settle_price, dtype=float)

    bsm_rmse = _rmse(settle_price, bsm_price)
    model_rmse = _rmse(y_true, y_pred)
    pct_improvement_rmse = (bsm_rmse - model_rmse) / bsm_rmse * 100.0 if bsm_rmse > 0 else float("nan")

    results: dict[str, Any] = {
        "global": {
            "rmse": model_rmse,
            "mae": _mae(y_true, y_pred),
            "mape_pct": _mape(y_true, y_pred),
            "r2": _r2(y_true, y_pred),
            "bsm_rmse": bsm_rmse,
            "bsm_mae": _mae(settle_price, bsm_price),
            "bsm_r2": _r2(settle_price, bsm_price),
            "pct_improvement_rmse": pct_improvement_rmse,
            "n_samples": int(len(y_true)),
        },
        "by_moneyness": {},
        "by_regime": {},
        "by_expiry_type": {},
        "by_dte": {},
        "by_option_type": {},
    }

    buckets = metadata["moneyness_bucket"].astype(str).to_numpy()
    for bucket in MONEYNESS_BUCKETS:
        mask = buckets == bucket
        rmse_val = _masked_rmse(y_true, y_pred, mask)
        mae_val = _masked_mae(y_true, y_pred, mask)
        if rmse_val is not None:
            results["by_moneyness"][bucket] = {
                "rmse": rmse_val,
                "mae": mae_val,
                "n_samples": int(mask.sum()),
            }

    regimes = metadata["regime_id"].to_numpy(dtype=int)
    for regime_id in REGIME_IDS:
        mask = regimes == regime_id
        rmse_val = _masked_rmse(y_true, y_pred, mask)
        if rmse_val is not None:
            results["by_regime"][f"regime_{regime_id}"] = {
                "rmse": rmse_val,
                "n_samples": int(mask.sum()),
            }

    if "is_weekly_expiry" in metadata.columns:
        weekly_mask = metadata["is_weekly_expiry"].to_numpy(dtype=bool)
        monthly_mask = metadata["is_monthly_expiry"].to_numpy(dtype=bool) if "is_monthly_expiry" in metadata.columns else ~weekly_mask
        for label, mask in (("weekly", weekly_mask), ("monthly", monthly_mask)):
            rmse_val = _masked_rmse(y_true, y_pred, mask)
            if rmse_val is not None:
                results["by_expiry_type"][label] = {
                    "rmse": rmse_val,
                    "n_samples": int(mask.sum()),
                }

    dte = metadata["dte"].to_numpy(dtype=float)
    dte_masks = {
        "dte_lt5": dte < 5,
        "dte_5to20": (dte >= 5) & (dte <= 20),
        "dte_gt20": dte > 20,
    }
    for label, mask in dte_masks.items():
        rmse_val = _masked_rmse(y_true, y_pred, mask)
        if rmse_val is not None:
            results["by_dte"][label] = {
                "rmse": rmse_val,
                "n_samples": int(mask.sum()),
            }

    option_types = metadata["option_type"].astype(str).str.upper().to_numpy()
    for opt_type in OPTION_TYPES:
        mask = option_types == opt_type
        rmse_val = _masked_rmse(y_true, y_pred, mask)
        if rmse_val is not None:
            results["by_option_type"][opt_type] = {
                "rmse": rmse_val,
                "n_samples": int(mask.sum()),
            }

    return results


def compute_iv_metrics(
    predicted_prices: np.ndarray,
    market_ivs: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    option_type: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Convert predicted prices to implied volatilities and compute IV-space metrics.

    IV RMSE is reported in percentage points (vol × 100).
    """
    predicted_prices = np.asarray(predicted_prices, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    option_type = np.asarray(option_type)

    predicted_iv = np.asarray(
        implied_vol_from_price(predicted_prices, S, K, T, r, option_type),
        dtype=float,
    )

    valid = np.isfinite(predicted_iv) & np.isfinite(market_ivs)
    if not valid.any():
        return {
            "iv_rmse": float("nan"),
            "iv_rmse_atm": float("nan"),
            "iv_rmse_by_moneyness": {},
            "n_valid": 0,
        }

    iv_err_pp = (market_ivs[valid] - predicted_iv[valid]) * 100.0
    iv_rmse = float(np.sqrt(np.mean(iv_err_pp**2)))

    results: dict[str, Any] = {
        "iv_rmse": iv_rmse,
        "n_valid": int(valid.sum()),
        "iv_rmse_by_moneyness": {},
    }

    if metadata is not None and "moneyness_bucket" in metadata.columns:
        buckets = metadata["moneyness_bucket"].astype(str).to_numpy()
        atm_mask = valid & (buckets == "ATM")
        if atm_mask.any():
            atm_err = (market_ivs[atm_mask] - predicted_iv[atm_mask]) * 100.0
            results["iv_rmse_atm"] = float(np.sqrt(np.mean(atm_err**2)))
        else:
            results["iv_rmse_atm"] = float("nan")

        for bucket in MONEYNESS_BUCKETS:
            mask = valid & (buckets == bucket)
            if not mask.any():
                continue
            bucket_err = (market_ivs[mask] - predicted_iv[mask]) * 100.0
            results["iv_rmse_by_moneyness"][bucket] = {
                "iv_rmse": float(np.sqrt(np.mean(bucket_err**2))),
                "n_samples": int(mask.sum()),
            }
    else:
        results["iv_rmse_atm"] = float("nan")

    return results


def plot_pricing_comparison(
    model_metrics: dict[str, dict[str, Any]],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Save article-ready bar charts comparing model RMSE by segment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = ("global", "by_moneyness", "by_regime")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, segment in zip(axes, segments):
        if segment == "global":
            labels = list(model_metrics.keys())
            values = [model_metrics[m]["global"]["rmse"] for m in labels]
            ax.bar(labels, values)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title("Global Price RMSE (INR)")
            continue

        keys = sorted(
            {
                key
                for metrics in model_metrics.values()
                for key in metrics.get(segment, {})
            }
        )
        x = np.arange(len(keys))
        width = 0.8 / max(len(model_metrics), 1)
        for i, (model_name, metrics) in enumerate(model_metrics.items()):
            values = [metrics.get(segment, {}).get(k, {}).get("rmse", np.nan) for k in keys]
            ax.bar(x + i * width, values, width=width, label=model_name)
        ax.set_xticks(x + width * (len(model_metrics) - 1) / 2)
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_title(segment.replace("_", " ").title())
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "pricing_rmse_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
