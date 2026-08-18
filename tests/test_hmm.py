"""Tests for HMM regime detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nope_in.regime.hmm import (
    fit_hmm,
    label_regimes,
    load_hmm_artifact,
    prepare_hmm_observations,
    save_hmm_artifact,
)


def _synthetic_market(n: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    vix = 14 + rng.normal(0, 3, n).cumsum() * 0.05
    vix = np.clip(vix, 10, 40)
    returns = rng.normal(0.0003, 0.01, n)
    close = 10000 * np.exp(np.cumsum(returns))

    underlying = pd.DataFrame(
        {
            "date": dates,
            "symbol": "NIFTY",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
            "returns_1d": returns,
        }
    )
    vix_df = pd.DataFrame(
        {
            "date": dates,
            "vix_close": vix,
            "vix_change": np.r_[0, np.diff(vix)],
            "vix_percentile_252d": 0.5,
        }
    )
    return underlying, vix_df


def test_prepare_hmm_observations_shape_and_standardisation():
    underlying, vix = _synthetic_market(200)
    X, dates, scaler, _ = prepare_hmm_observations(underlying, vix, symbol="NIFTY")

    assert X.shape == (200, 3)
    assert len(dates) == 200
    assert "mean" in scaler and "std" in scaler
    assert np.allclose(X.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X.std(axis=0), 1, atol=1e-10)


def test_fit_hmm_returns_four_state_model():
    underlying, vix = _synthetic_market(500)
    X, _, _, _ = prepare_hmm_observations(underlying, vix, symbol="NIFTY")
    model = fit_hmm(X, n_states=4, n_restarts=3, n_iter=50, random_seed=1)
    assert model.n_components == 4
    assert model.means_.shape == (4, 3)


def test_label_regimes_probabilities_sum_to_one():
    underlying, vix = _synthetic_market(500)
    X, dates, scaler, aligned = prepare_hmm_observations(underlying, vix, symbol="NIFTY")
    model = fit_hmm(X, n_states=4, n_restarts=3, n_iter=50, random_seed=2)
    labels = label_regimes(model, X, dates, "NIFTY", scaler, aligned)

    prob_cols = ["regime_p0", "regime_p1", "regime_p2", "regime_p3"]
    sums = labels[prob_cols].sum(axis=1)
    assert ((sums - 1.0).abs() < 1e-5).all()
    assert labels["regime_id"].between(0, 3).all()
    assert set(labels["regime_name"].unique()).issubset(
        {"LV-Trending", "LV-Mean-Reverting", "HV-Trending", "HV-Stress"}
    )


def test_scaler_fit_on_train_mask_only():
    underlying, vix = _synthetic_market(300)
    fit_mask = np.array([True] * 150 + [False] * 150)
    X, _, scaler, _ = prepare_hmm_observations(
        underlying, vix, symbol="NIFTY", fit_mask=fit_mask
    )
    raw = underlying.merge(vix, on="date").dropna()
    train_raw = raw.iloc[:150][["returns_1d", "vix_close", "vix_change"]].to_numpy()
    expected_mean = train_raw.mean(axis=0)
    assert np.allclose(scaler["mean"], expected_mean, rtol=0.01)


def test_save_and_load_artifact_roundtrip(tmp_path: Path):
    underlying, vix = _synthetic_market(300)
    X, dates, scaler, aligned = prepare_hmm_observations(underlying, vix, symbol="NIFTY")
    model = fit_hmm(X, n_states=4, n_restarts=2, n_iter=30, random_seed=3)

    path = tmp_path / "hmm_nifty.pkl"
    save_hmm_artifact(model, scaler, path, metadata={"symbol": "NIFTY"})
    loaded_model, loaded_scaler, meta = load_hmm_artifact(path)

    assert meta["symbol"] == "NIFTY"
    assert np.allclose(loaded_model.means_, model.means_)
    assert np.allclose(loaded_scaler["mean"], scaler["mean"])

    labels_orig = label_regimes(model, X, dates, "NIFTY", scaler, aligned)
    labels_loaded = label_regimes(loaded_model, X, dates, "NIFTY", loaded_scaler, aligned)
    pd.testing.assert_frame_equal(labels_orig, labels_loaded)


@pytest.mark.skipif(
    not Path("data/processed/underlying_ohlcv.parquet").exists(),
    reason="Processed data not available",
)
def test_real_data_nifty_hmm_smoke():
    """Integration smoke on real Tier 1 data if present."""
    underlying = pd.read_parquet("data/processed/underlying_ohlcv.parquet")
    vix = pd.read_parquet("data/processed/india_vix.parquet")
    X, dates, scaler, aligned = prepare_hmm_observations(underlying, vix, symbol="NIFTY")
    fit_mask = pd.to_datetime(dates) <= "2021-12-31"
    model = fit_hmm(X[fit_mask], n_states=4, n_restarts=3, n_iter=50)
    labels = label_regimes(model, X, dates, "NIFTY", scaler, aligned)
    assert len(labels) >= 900
    covid = labels[labels["date"] == "2020-03-23"]
    if not covid.empty:
        assert covid.iloc[0]["regime_name"] in ("HV-Stress", "HV-Trending")
