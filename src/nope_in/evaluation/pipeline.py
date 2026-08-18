"""
NOPE-IN evaluation pipeline — metrics, IV analysis, and SHAP explainability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

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
from nope_in.models.nope import NOPEIndia
from nope_in.training.dataset import (
    NOPEDataModule,
    get_regime_feature_indices,
    load_feature_metadata,
)
from nope_in.training.pipeline import build_model

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(cfg: DictConfig, *parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_checkpoint(checkpoint_path: Path) -> tuple[NOPEIndia, list[str], dict[str, Any]]:
    """Load a saved NOPE-IN checkpoint for evaluation."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_columns: list[str] = ckpt["feature_columns"]
    config = ckpt["config"]

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(config)
    model = build_model(cfg, feature_columns)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, feature_columns, ckpt


@torch.no_grad()
def run_inference(
    model: NOPEIndia,
    dataloader: DataLoader,
    device: torch.device,
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

    for batch in dataloader:
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    checkpoint_path = _resolve_path(cfg, cfg.evaluation.checkpoint_path)
    features_dir = _resolve_path(cfg, cfg.evaluation.features_dir)
    metadata_path = _resolve_path(cfg, cfg.evaluation.metadata_path)
    output_dir = _resolve_path(cfg, cfg.evaluation.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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

    datamodule = NOPEDataModule(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=int(cfg.evaluation.batch_size),
        num_workers=int(cfg.evaluation.get("num_workers", 0)),
        cal_fraction=0.0,
    )
    datamodule.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = datamodule.test_dataloader()
    inference = run_inference(model, test_loader, device)

    test_df = pd.read_parquet(features_dir / "test.parquet")
    eval_metadata = build_eval_metadata(test_df)

    y_true = inference["settle_price"]
    y_pred = inference["y_pred"]
    bsm_price = inference["bsm_price"]

    all_metrics: dict[str, Any] = {}

    all_metrics["BSM"] = compute_pricing_metrics(
        y_true=y_true,
        y_pred=bsm_price,
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

    if "spot" in inference:
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

    results_path = output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(all_metrics, indent=2))
    log.info("Saved evaluation metrics → %s", results_path)

    plot_pricing_comparison(
        {"BSM": all_metrics["BSM"], "NOPE-IN": all_metrics["NOPE-IN"]},
        plots_dir,
    )

    shap_cfg = cfg.evaluation.get("shap", {})
    if bool(shap_cfg.get("enabled", True)):
        _run_shap_analysis(
            model=model,
            train_df=pd.read_parquet(features_dir / "train.parquet"),
            test_df=test_df,
            feature_columns=feature_columns,
            eval_metadata=eval_metadata,
            shap_cfg=shap_cfg,
            plots_dir=plots_dir,
            device=device,
        )

    log.info("Phase 5 complete. Results → %s", output_dir)
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

    log.info(
        "Computing SHAP values (background=%d, explain=%d)…",
        len(bg_idx),
        len(ex_idx),
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

    plot_global_feature_importance(
        shap_values,
        feature_columns,
        plots_dir / "global_feature_importance.png",
    )
    plot_regime_conditional_importance(
        shap_values,
        regime_labels,
        feature_columns,
        plots_dir,
    )

    if "days_to_next_rbi_mpc" in explain_metadata.columns:
        plot_event_attribution(
            shap_values,
            feature_columns,
            explain_metadata,
            plots_dir / "rbi_mpc_event_attribution.png",
        )


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_evaluation_pipeline(cfg)


if __name__ == "__main__":
    main()
