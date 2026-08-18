"""Unit tests for Phase 5 evaluation metrics and explainability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from nope_in.training.dataset import get_regime_feature_indices


def _synthetic_metadata(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    moneyness = rng.uniform(0.85, 1.15, size=n)
    df = pd.DataFrame(
        {
            "moneyness": moneyness,
            "is_DITM": (moneyness < 0.90).astype(float),
            "is_ITM": ((moneyness >= 0.90) & (moneyness < 0.98)).astype(float),
            "is_ATM": ((moneyness >= 0.98) & (moneyness <= 1.02)).astype(float),
            "is_OTM": ((moneyness > 1.02) & (moneyness <= 1.10)).astype(float),
            "is_DOTM": (moneyness > 1.10).astype(float),
            "regime_p0": rng.uniform(0, 1, size=n),
            "regime_p1": rng.uniform(0, 1, size=n),
            "regime_p2": rng.uniform(0, 1, size=n),
            "regime_p3": rng.uniform(0, 1, size=n),
            "option_type": rng.choice(["CE", "PE"], size=n),
            "dte": rng.integers(1, 40, size=n),
            "is_weekly_expiry": rng.choice([0.0, 1.0], size=n),
            "is_monthly_expiry": rng.choice([0.0, 1.0], size=n),
            "days_to_next_rbi_mpc": rng.integers(1, 60, size=n),
            "is_rbi_week": rng.choice([0.0, 1.0], size=n),
        }
    )
    return build_eval_metadata(df)


class TestPricingMetrics:
    def test_perfect_predictions(self):
        n = 200
        y = np.linspace(10, 100, n)
        meta = _synthetic_metadata(n)
        metrics = compute_pricing_metrics(y, y, y, y, meta)
        assert metrics["global"]["rmse"] == pytest.approx(0.0)
        assert metrics["global"]["mae"] == pytest.approx(0.0)
        assert metrics["global"]["r2"] == pytest.approx(1.0)

    def test_bsm_improvement(self):
        n = 100
        rng = np.random.default_rng(1)
        settle = rng.uniform(50, 150, size=n)
        bsm = settle + rng.normal(0, 5, size=n)
        model = settle + rng.normal(0, 1, size=n)
        meta = _synthetic_metadata(n, seed=1)
        metrics = compute_pricing_metrics(settle, model, bsm, settle, meta)
        assert metrics["global"]["pct_improvement_rmse"] > 0
        assert "ATM" in metrics["by_moneyness"]
        assert "regime_0" in metrics["by_regime"]
        assert "CE" in metrics["by_option_type"]

    def test_segmented_keys(self):
        n = 50
        y = np.ones(n) * 100
        meta = _synthetic_metadata(n, seed=2)
        metrics = compute_pricing_metrics(y, y * 1.05, y, y, meta)
        assert metrics["by_dte"]["dte_lt5"]["rmse"] > 0
        assert metrics["global"]["n_samples"] == n


class TestIVMetrics:
    def test_iv_metrics_runs(self):
        n = 32
        rng = np.random.default_rng(3)
        spot = np.full(n, 20000.0)
        strike = spot * rng.uniform(0.95, 1.05, size=n)
        t = np.full(n, 7 / 365)
        r = np.full(n, 0.065)
        iv = np.full(n, 0.18)
        from nope_in.features.bsm import bsm_price

        prices = np.asarray(bsm_price(spot, strike, t, r, iv, "CE"), dtype=float)
        meta = _synthetic_metadata(n, seed=3)
        out = compute_iv_metrics(prices, iv, spot, strike, t, r, np.array(["CE"] * n), meta)
        assert out["iv_rmse"] < 1.0
        assert out["n_valid"] == n


class TestPlots:
    def test_pricing_comparison_plot(self, tmp_path: Path):
        metrics = {
            "BSM": {
                "global": {"rmse": 5.0},
                "by_moneyness": {"ATM": {"rmse": 4.0}},
                "by_regime": {"regime_0": {"rmse": 5.5}},
            },
            "NOPE-IN": {
                "global": {"rmse": 3.0},
                "by_moneyness": {"ATM": {"rmse": 2.0}},
                "by_regime": {"regime_0": {"rmse": 3.5}},
            },
        }
        plot_pricing_comparison(metrics, tmp_path)
        assert (tmp_path / "pricing_rmse_comparison.png").exists()


class TestExplainability:
    @pytest.fixture()
    def tiny_model(self) -> tuple[NOPEIndia, list[str]]:
        feature_names = [f"f{i}" for i in range(8)] + [
            "regime_p0",
            "regime_p1",
            "regime_p2",
            "regime_p3",
            "days_to_next_rbi_mpc",
        ]
        regime_idx = get_regime_feature_indices(feature_names)
        model = NOPEIndia(
            input_dim=len(feature_names),
            hidden_dim=16,
            n_experts=4,
            regime_feature_indices=regime_idx,
        )
        return model, feature_names

    def test_shap_values_shape(self, tiny_model, tmp_path: Path):
        pytest.importorskip("shap")
        model, feature_names = tiny_model
        n_bg, n_ex = 16, 24
        rng = np.random.default_rng(0)
        d = len(feature_names)
        X_bg = rng.normal(size=(n_bg, d)).astype(np.float32)
        X_ex = rng.normal(size=(n_ex, d)).astype(np.float32)
        bsm_bg = rng.uniform(50, 200, size=n_bg).astype(np.float32)
        bsm_ex = rng.uniform(50, 200, size=n_ex).astype(np.float32)

        shap_values = compute_shap_values(
            model,
            X_bg,
            X_ex,
            bsm_bg,
            bsm_ex,
            feature_names,
            output_dir=tmp_path,
        )
        assert shap_values.shape == (n_ex, d)
        assert (tmp_path / "shap_values.npy").exists()

    def test_shap_plots(self, tiny_model, tmp_path: Path):
        pytest.importorskip("shap")
        _, feature_names = tiny_model
        rng = np.random.default_rng(1)
        n = 40
        shap_values = rng.normal(size=(n, len(feature_names))).astype(np.float32)
        regime_labels = rng.integers(0, 4, size=n)

        plot_global_feature_importance(
            shap_values,
            feature_names,
            tmp_path / "global_feature_importance.png",
            top_n=5,
        )
        plot_regime_conditional_importance(
            shap_values,
            regime_labels,
            feature_names,
            tmp_path,
            top_n=5,
        )

        metadata = pd.DataFrame(
            {
                "days_to_next_rbi_mpc": rng.integers(1, 60, size=n),
                "is_rbi_week": rng.choice([0.0, 1.0], size=n),
            }
        )
        plot_event_attribution(
            shap_values,
            feature_names,
            metadata,
            tmp_path / "rbi_event.png",
        )

        assert (tmp_path / "global_feature_importance_bar.png").exists()
        assert (tmp_path / "regime_conditional_importance.png").exists()
        assert (tmp_path / "rbi_event.png").exists()
