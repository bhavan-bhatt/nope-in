"""
NOPE-IN evaluation pipeline — metrics, IV analysis, and SHAP explainability.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from nope_in.evaluation.explainability import (
    compute_shap_values,
    plot_event_attribution,
    plot_global_feature_importance,
    plot_regime_conditional_importance,
)
from nope_in.evaluation.metrics import (
    build_eval_metadata,
    compute_iv_metrics,
    compute_pricing_metrics,
    plot_pricing_comparison,
)
from nope_in.evaluation.shrinkage import apply_bucket_shrinkage, bucket_labels, fit_bucket_shrinkage
from nope_in.models.nope import NOPEIndia
from nope_in.training.dataset import (
    NOPEDataModule,
    get_regime_feature_indices,
    load_feature_metadata,
)
from nope_in.training.pipeline import build_model
from nope_in.utils.device import device_label, resolve_device, resolve_shap_device

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_STEPS = 6


def _progress(message: str, step: int | None = None) -> None:
    """Print a user-visible pipeline status line."""
    prefix = f"[Step {step}/{EVAL_STEPS}] " if step is not None else ""
    print(f"{prefix}{message}", flush=True)


def _resolve_path(cfg: DictConfig, *parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_checkpoint(checkpoint_path: Path) -> tuple[NOPEIndia, list[str], dict[str, Any]]:
    """Load a saved NOPE-IN checkpoint for evaluation."""
    _progress(f"Loading checkpoint from {checkpoint_path} …", step=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_columns: list[str] = ckpt["feature_columns"]
    config = ckpt["config"]

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(config)
    model = build_model(cfg, feature_columns)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _progress(
        f"Checkpoint loaded ({model.count_parameters():,} parameters, {len(feature_columns)} features)",
        step=1,
    )
    return model, feature_columns, ckpt


@torch.no_grad()
def run_inference(
    model: NOPEIndia,
    dataloader: DataLoader,
    device: torch.device,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Run batched inference and collect predictions plus contract fields."""
    model = model.to(device)
    model.eval()

    preds: list[np.ndarray] = []
    bsm_prices: list[np.ndarray] = []
    settle_prices: list[np.ndarray] = []
    spots: list[np.ndarray] = []
    strikes: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    rates: list[np.ndarray] = []
    option_types: list[str] = []
    implied_vols: list[np.ndarray] = []
    regime_ids: list[np.ndarray] = []
    moneyness_buckets: list[str] = []

    batch_iter = dataloader
    if show_progress:
        batch_iter = tqdm(
            dataloader,
            desc="Test-set inference",
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
        )

    for batch in batch_iter:
        batch_tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k not in {"moneyness_bucket", "option_type"}
        }
        outputs = model(batch_tensors["features"], batch_tensors["bsm_price"])

        preds.append(outputs["final_price"].cpu().numpy())
        bsm_prices.append(batch_tensors["bsm_price"].cpu().numpy())
        settle_prices.append(batch_tensors["settle_price"].cpu().numpy())
        implied_vols.append(batch_tensors["implied_vol"].cpu().numpy())
        regime_ids.append(batch_tensors["regime_id"].cpu().numpy())
        moneyness_buckets.extend(batch["moneyness_bucket"])

        if "spot" in batch_tensors:
            spots.append(batch_tensors["spot"].cpu().numpy())
            strikes.append(batch_tensors["strike"].cpu().numpy())
            ts.append(batch_tensors["T"].cpu().numpy())
            rates.append(batch_tensors["rate"].cpu().numpy())
            option_types.extend(batch["option_type"])

    result: dict[str, np.ndarray | list[str]] = {
        "y_pred": np.concatenate(preds),
        "bsm_price": np.concatenate(bsm_prices),
        "settle_price": np.concatenate(settle_prices),
        "implied_vol": np.concatenate(implied_vols),
        "regime_id": np.concatenate(regime_ids),
        "moneyness_bucket": np.array(moneyness_buckets),
    }

    if spots:
        result.update(
            {
                "spot": np.concatenate(spots),
                "strike": np.concatenate(strikes),
                "T": np.concatenate(ts),
                "rate": np.concatenate(rates),
                "option_type": np.array(option_types),
            }
        )

    return result  # type: ignore[return-value]


def run_evaluation_pipeline(cfg: DictConfig) -> dict[str, Any]:
    """Execute Phase 5 evaluation end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    checkpoint_path = _resolve_path(cfg, cfg.evaluation.checkpoint_path)
    features_dir = _resolve_path(cfg, cfg.evaluation.features_dir)
    metadata_path = _resolve_path(cfg, cfg.evaluation.metadata_path)
    output_dir = _resolve_path(cfg, cfg.evaluation.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("", flush=True)
    _progress("NOPE-IN Phase 5 — Evaluation & Explainability")
    _progress(f"Output directory: {output_dir}")

    device_pref = str(cfg.evaluation.get("device", "auto"))
    eval_device = resolve_device(device_pref)
    _progress(f"Compute device: {device_label(eval_device)} ({eval_device})")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run `make train` first."
        )

    meta = load_feature_metadata(metadata_path)
    metadata_features = meta["feature_columns"]

    model, ckpt_features, _ckpt = load_checkpoint(checkpoint_path)
    feature_columns = ckpt_features
    if ckpt_features != metadata_features:
        log.warning("Checkpoint feature columns differ from metadata; using checkpoint columns.")

    _progress(f"Loading feature parquets from {features_dir} …", step=2)
    datamodule = NOPEDataModule(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=int(cfg.evaluation.batch_size),
        num_workers=int(cfg.evaluation.get("num_workers", 0)),
        cal_fraction=0.0,
    )
    datamodule.setup()
    assert datamodule.test_dataset is not None
    n_test = len(datamodule.test_dataset)
    _progress(f"Test set ready: {n_test:,} rows", step=2)

    device = eval_device
    _progress(f"Running inference on {device_label(device)} ({device}) …", step=3)
    test_loader = datamodule.test_dataloader()
    inference = run_inference(model, test_loader, device)

    _progress("Calibrating bucket shrinkage on validation split …", step=3)
    val_loader = datamodule.val_dataloader()
    val_inference = run_inference(model, val_loader, device, show_progress=False)
    val_df = pd.read_parquet(features_dir / "val.parquet")
    val_metadata = build_eval_metadata(val_df)
    val_correction_true = val_inference["settle_price"] - val_inference["bsm_price"]
    val_correction_pred = val_inference["y_pred"] - val_inference["bsm_price"]
    val_buckets = bucket_labels(val_metadata)
    shrink_map = fit_bucket_shrinkage(val_correction_true, val_correction_pred, val_buckets)
    _progress(
        f"Shrinkage calibrated on {len(val_correction_true):,} val rows — "
        f"{len(shrink_map.factors)} buckets fitted, global fallback λ={shrink_map.global_factor:.2f}",
        step=3,
    )

    _progress("Building evaluation metadata (moneyness, regime, DTE) …", step=4)
    test_df = pd.read_parquet(features_dir / "test.parquet")
    eval_metadata = build_eval_metadata(test_df)

    y_true = inference["settle_price"]
    y_pred_raw = inference["y_pred"]
    bsm_price = inference["bsm_price"]

    test_correction_pred = y_pred_raw - bsm_price
    test_buckets = bucket_labels(eval_metadata)
    y_pred = apply_bucket_shrinkage(bsm_price, test_correction_pred, test_buckets, shrink_map)

    all_metrics: dict[str, Any] = {}

    _progress("Computing pricing metrics (BSM baseline + NOPE-IN) …", step=4)
    all_metrics["BSM"] = compute_pricing_metrics(
        y_true=y_true,
        y_pred=bsm_price,
        bsm_price=bsm_price,
        settle_price=y_true,
        metadata=eval_metadata,
    )
    all_metrics["NOPE-IN-Raw"] = compute_pricing_metrics(
        y_true=y_true,
        y_pred=y_pred_raw,
        bsm_price=bsm_price,
        settle_price=y_true,
        metadata=eval_metadata,
    )
    all_metrics["NOPE-IN"] = compute_pricing_metrics(
        y_true=y_true,
        y_pred=y_pred,
        bsm_price=bsm_price,
        settle_price=y_true,
        metadata=eval_metadata,
    )

    bsm_rmse = all_metrics["BSM"]["global"]["rmse"]
    nope_raw_rmse = all_metrics["NOPE-IN-Raw"]["global"]["rmse"]
    nope_rmse = all_metrics["NOPE-IN"]["global"]["rmse"]
    improvement = all_metrics["NOPE-IN"]["global"]["pct_improvement_rmse"]
    _progress(
        f"Price RMSE — BSM: ₹{bsm_rmse:.3f} | NOPE-IN-Raw: ₹{nope_raw_rmse:.3f} | "
        f"NOPE-IN (shrunk): ₹{nope_rmse:.3f} ({improvement:+.1f}% vs BSM)",
        step=4,
    )

    if "spot" in inference:
        _progress(f"Inverting {n_test:,} predicted prices to implied vol …", step=4)
        iv_metrics = compute_iv_metrics(
            predicted_prices=y_pred,
            market_ivs=inference["implied_vol"],
            S=inference["spot"],
            K=inference["strike"],
            T=inference["T"],
            r=inference["rate"],
            option_type=inference["option_type"],
            metadata=eval_metadata,
        )
        all_metrics["NOPE-IN"]["iv_metrics"] = iv_metrics

        bsm_iv_metrics = compute_iv_metrics(
            predicted_prices=bsm_price,
            market_ivs=inference["implied_vol"],
            S=inference["spot"],
            K=inference["strike"],
            T=inference["T"],
            r=inference["rate"],
            option_type=inference["option_type"],
            metadata=eval_metadata,
        )
        all_metrics["BSM"]["iv_metrics"] = bsm_iv_metrics
        _progress(
            f"IV RMSE — BSM: {bsm_iv_metrics['iv_rmse']:.3f} pp | "
            f"NOPE-IN: {iv_metrics['iv_rmse']:.3f} pp | "
            f"ATM: {iv_metrics.get('iv_rmse_atm', float('nan')):.3f} pp",
            step=4,
        )

    _progress("Saving metrics JSON and pricing comparison plot …", step=5)
    results_path = output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(all_metrics, indent=2))
    _progress(f"Saved evaluation metrics → {results_path}", step=5)

    shrinkage_path = output_dir / "shrinkage_map.json"
    shrinkage_path.write_text(json.dumps(shrink_map.to_dict(), indent=2))
    _progress(f"Saved shrinkage calibration → {shrinkage_path}", step=5)

    plot_pricing_comparison(
        {"BSM": all_metrics["BSM"], "NOPE-IN-Raw": all_metrics["NOPE-IN-Raw"], "NOPE-IN": all_metrics["NOPE-IN"]},
        plots_dir,
    )
    _progress(f"Saved pricing plot → {plots_dir / 'pricing_rmse_comparison.png'}", step=5)

    shap_cfg = cfg.evaluation.get("shap", {})
    if bool(shap_cfg.get("enabled", True)):
        bg_n = int(shap_cfg.get("background_samples", 500))
        ex_n = int(shap_cfg.get("explain_samples", 1000))
        shap_device = resolve_shap_device(device, str(shap_cfg.get("device", "auto")))
        _progress(
            f"Starting SHAP explainability (background={bg_n}, explain={ex_n}) on "
            f"{device_label(shap_device)} — this is the slowest step …",
            step=6,
        )
        if shap_device.type != device.type:
            _progress(
                "Note: SHAP runs on CPU because DeepExplainer does not support Apple MPS; "
                "inference still used the GPU above.",
            )
        _run_shap_analysis(
            model=model,
            train_df=pd.read_parquet(features_dir / "train.parquet"),
            test_df=test_df,
            feature_columns=feature_columns,
            eval_metadata=eval_metadata,
            shap_cfg=shap_cfg,
            plots_dir=plots_dir,
            device=shap_device,
        )
    else:
        _progress("SHAP explainability skipped (evaluation.shap.enabled=false)", step=6)

    print("", flush=True)
    _progress(f"Phase 5 complete. All outputs → {output_dir}")
    _progress(f"  • Metrics: {results_path}")
    _progress(f"  • Plots:   {plots_dir}/")
    return {"results_path": str(results_path), "metrics": all_metrics}


def _run_shap_analysis(
    model: NOPEIndia,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    eval_metadata: pd.DataFrame,
    shap_cfg: dict[str, Any],
    plots_dir: Path,
    device: torch.device,
) -> None:
    """Sample background/explain sets and produce SHAP plots."""
    bg_n = int(shap_cfg.get("background_samples", 500))
    ex_n = int(shap_cfg.get("explain_samples", 1000))
    max_evals = int(shap_cfg.get("max_evals", 2000))
    seed = int(shap_cfg.get("seed", 42))

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(train_df), size=min(bg_n, len(train_df)), replace=False)
    ex_idx = rng.choice(len(test_df), size=min(ex_n, len(test_df)), replace=False)

    X_background = train_df.iloc[bg_idx][feature_columns].to_numpy(dtype=np.float32)
    bsm_background = train_df.iloc[bg_idx]["bsm_price"].to_numpy(dtype=np.float32)
    X_explain = test_df.iloc[ex_idx][feature_columns].to_numpy(dtype=np.float32)
    bsm_explain = test_df.iloc[ex_idx]["bsm_price"].to_numpy(dtype=np.float32)

    _progress(
        f"SHAP: sampled {len(bg_idx)} background + {len(ex_idx)} explain rows",
    )

    shap_values = compute_shap_values(
        model=model,
        X_background=X_background,
        X_explain=X_explain,
        bsm_background=bsm_background,
        bsm_explain=bsm_explain,
        feature_names=feature_columns,
        output_dir=plots_dir,
        max_evals=max_evals,
        device=device,
    )

    regime_labels = eval_metadata.iloc[ex_idx]["regime_id"].to_numpy(dtype=int)
    explain_metadata = eval_metadata.iloc[ex_idx].reset_index(drop=True)

    _progress("SHAP: generating global feature importance plots …")
    plot_global_feature_importance(
        shap_values,
        feature_columns,
        plots_dir / "global_feature_importance.png",
    )
    _progress("SHAP: generating regime-conditional importance plots …")
    plot_regime_conditional_importance(
        shap_values,
        regime_labels,
        feature_columns,
        plots_dir,
    )

    if "days_to_next_rbi_mpc" in explain_metadata.columns:
        _progress("SHAP: generating RBI MPC event attribution plot …")
        plot_event_attribution(
            shap_values,
            feature_columns,
            explain_metadata,
            plots_dir / "rbi_mpc_event_attribution.png",
        )

    _progress(f"SHAP complete → {plots_dir}")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_evaluation_pipeline(cfg)


if __name__ == "__main__":
    main()
