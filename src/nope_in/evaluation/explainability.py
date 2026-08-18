"""
SHAP-Based Explainability for NOPE-IN.

PURPOSE:
    Uses SHAP (SHapley Additive exPlanations) to produce:
      1. Global feature importance: Which features most explain BSM errors?
      2. Regime-conditional importance: How does feature importance change by regime?
      3. Example-level attribution: For a specific option, what drove the BSM error correction?
      4. Event-driven attribution: How do SHAP values change around RBI MPC meetings?

    These SHAP analyses are the RESEARCH OUTPUTS of the project.
    They answer the question: "What does BSM systematically miss in Indian markets?"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nope_in.utils.device import device_label

if TYPE_CHECKING:
    from nope_in.models.nope import NOPEIndia


def _shap_progress(message: str) -> None:
    print(f"  SHAP | {message}", flush=True)


class _SHAPForwardModel(nn.Module):
    """Wrap NOPEIndia for SHAP: features + bsm_price -> scalar BSM error prediction."""

    def __init__(self, model: NOPEIndia):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, bsm_price: torch.Tensor) -> torch.Tensor:
        out = self.model(features, bsm_price.squeeze(-1))
        return out["bsm_error_pred"].unsqueeze(-1)


def compute_shap_values(
    model: NOPEIndia,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    bsm_background: np.ndarray,
    bsm_explain: np.ndarray,
    feature_names: list[str],
    output_dir: Path | None = None,
    max_evals: int = 2000,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Run SHAP DeepExplainer on the NOPE-IN model.

    Uses DeepExplainer (appropriate for PyTorch neural networks) with both
    feature vectors and BSM prices as model inputs. Only feature attributions
    are returned.

    Positive SHAP = feature pushed the BSM error correction UP.
    Negative SHAP = feature pushed the BSM error correction DOWN.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explainability. Install with: pip install shap"
        ) from exc

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    wrapper = _SHAPForwardModel(model).to(device)

    bg_x = torch.as_tensor(X_background, dtype=torch.float32, device=device)
    bg_bsm = torch.as_tensor(bsm_background, dtype=torch.float32, device=device).reshape(-1, 1)
    ex_x = torch.as_tensor(X_explain, dtype=torch.float32, device=device)
    ex_bsm = torch.as_tensor(bsm_explain, dtype=torch.float32, device=device).reshape(-1, 1)

    _shap_progress(
        f"Initialising DeepExplainer on {device_label(device)} ({device}) "
        f"({len(X_background)} background × {len(X_explain)} explain samples) …"
    )
    explainer = shap.DeepExplainer(wrapper, [bg_x, bg_bsm])
    try:
        _shap_progress("Computing SHAP values (DeepExplainer) — please wait …")
        shap_output = explainer.shap_values([ex_x, ex_bsm], check_additivity=False)
        _shap_progress("DeepExplainer finished.")
    except (AssertionError, RuntimeError):
        # LayerNorm/GELU in SharedTrunk are not fully supported by DeepExplainer.
        _shap_progress("DeepExplainer unsupported; falling back to GradientExplainer …")
        explainer = shap.GradientExplainer(wrapper, [bg_x, bg_bsm])
        shap_output = explainer.shap_values([ex_x, ex_bsm])
        _shap_progress("GradientExplainer finished.")

    if isinstance(shap_output, list):
        feature_shap = shap_output[0]
    else:
        feature_shap = shap_output

    feature_shap = np.asarray(feature_shap, dtype=np.float32)
    if feature_shap.ndim == 3:
        feature_shap = feature_shap[..., 0]

    if output_dir is not None:
        np.save(output_dir / "shap_values.npy", feature_shap)
        pd.DataFrame(feature_shap, columns=feature_names).to_parquet(
            output_dir / "shap_values.parquet",
            index=False,
        )
        _shap_progress(f"Saved raw SHAP arrays → {output_dir}")

    _ = max_evals  # reserved for KernelExplainer fallback if needed later
    return feature_shap


def plot_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
    dpi: int = 300,
) -> None:
    """
    Produce SHAP summary plots: mean |SHAP| bar chart and beeswarm-style scatter.

    Saves publication-quality PNGs (300 DPI, A4-friendly dimensions).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explainability plots. Install with: pip install shap"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] for i in order]
    top_values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8.27, 6), dpi=dpi)
    ax.barh(top_names[::-1], top_values[::-1], color="#2563eb")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance — BSM Error Correction")
    fig.tight_layout()
    bar_path = output_path.with_name(output_path.stem + "_bar.png")
    fig.savefig(bar_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    beeswarm_path = output_path.with_name(output_path.stem + "_beeswarm.png")
    plt.figure(figsize=(8.27, 6), dpi=dpi)
    shap.summary_plot(
        shap_values[:, order],
        features=None,
        feature_names=top_names,
        show=False,
        plot_type="dot",
        max_display=top_n,
    )
    plt.title("SHAP Beeswarm — Top Features")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_regime_conditional_importance(
    shap_values: np.ndarray,
    regime_labels: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 15,
    dpi: int = 300,
) -> None:
    """
    Plot feature importance separately for each of the 4 HMM regimes.

    Layout: 2×2 grid of bar charts (one per regime).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regime_labels = np.asarray(regime_labels, dtype=int)
    regimes = sorted(np.unique(regime_labels))
    n_regimes = len(regimes)
    n_cols = 2
    n_rows = int(np.ceil(n_regimes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), dpi=dpi)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, regime_id in zip(axes_flat, regimes):
        mask = regime_labels == regime_id
        if not mask.any():
            ax.set_visible(False)
            continue
        mean_abs = np.mean(np.abs(shap_values[mask]), axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        names = [feature_names[i] for i in order]
        values = mean_abs[order]
        ax.barh(names[::-1], values[::-1], color="#059669")
        ax.set_title(f"Regime {regime_id} (n={int(mask.sum())})")
        ax.set_xlabel("Mean |SHAP|")

    for ax in axes_flat[len(regimes) :]:
        ax.set_visible(False)

    fig.suptitle("Regime-Conditional Feature Importance", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "regime_conditional_importance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_event_attribution(
    shap_values: np.ndarray,
    feature_names: list[str],
    metadata_df: pd.DataFrame,
    output_path: Path,
    feature: str = "days_to_next_rbi_mpc",
    dpi: int = 300,
) -> None:
    """
    Plot mean SHAP value of an event feature vs days until RBI MPC.

    Expected finding: SHAP values spike in the 5 days before RBI meeting.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if feature not in feature_names:
        raise ValueError(f"Feature '{feature}' not found in feature_names")
    if feature not in metadata_df.columns:
        raise ValueError(f"Column '{feature}' missing from metadata_df")

    feat_idx = feature_names.index(feature)
    days = metadata_df[feature].to_numpy(dtype=float)
    shap_feat = shap_values[:, feat_idx]

    valid = np.isfinite(days) & np.isfinite(shap_feat)
    days = days[valid]
    shap_feat = shap_feat[valid]

    plot_df = pd.DataFrame({"days": days, "shap": shap_feat})
    plot_df = plot_df[(plot_df["days"] >= 1) & (plot_df["days"] <= 60)]
    if plot_df.empty:
        return

    grouped = plot_df.groupby("days", as_index=False)["shap"].mean()

    fig, ax = plt.subplots(figsize=(8.27, 4.5), dpi=dpi)
    ax.plot(grouped["days"], grouped["shap"], marker="o", color="#dc2626", linewidth=1.5)
    ax.axvline(5, color="gray", linestyle="--", linewidth=1, label="5 days before MPC")
    ax.set_xlabel("Days to next RBI MPC")
    ax.set_ylabel(f"Mean SHAP — {feature}")
    ax.set_title("Event-Driven Attribution: RBI MPC Proximity")
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
