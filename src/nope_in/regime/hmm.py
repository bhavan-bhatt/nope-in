"""
Hidden Markov Model for Indian Market Regime Detection.

PURPOSE:
    Fits a 4-state Gaussian HMM on Indian market data to classify each trading day
    into one of four regimes. The HMM is fit on (daily returns, India VIX, VIX 1-day change)
    — a 3-dimensional Gaussian observation model.

    This module uses the `hmmlearn` library (GaussianHMM) as the primary implementation.

    THE FOUR REGIMES (Indian market interpretation):

      Regime 0 — Low-Vol Trending:
        Characteristics: India VIX < 14, moderate positive drift, low vol clustering.
        Indian context: Bull market (2016-2018, 2020 recovery, 2021 bull run).
        BSM errors here: Small and symmetric. BSM approximately correct.
        Implication for model: Expert 0 should predict near-zero adjustments.

      Regime 1 — Low-Vol Mean-Reverting:
        Characteristics: India VIX 14-18, low drift, oscillating around support/resistance.
        Indian context: Sideways market phases, consolidation periods.
        BSM errors here: Theta slightly overestimated (BSM assumes GBM, not mean-reversion).
        Implication for model: Expert 1 should show slight theta correction.

      Regime 2 — High-Vol Trending:
        Characteristics: India VIX 18-25, significant directional move.
        Indian context: Budget/election rallies, FII-driven directional moves.
        BSM errors here: OTM put underpricing (fat tails not captured), skew premium.
        Implication for model: Expert 2 should show large skew-related correction.

      Regime 3 — High-Vol Stress:
        Characteristics: India VIX > 25, panic or euphoria, extreme moves.
        Indian context: COVID crash (Mar 2020), IL&FS crisis (Oct 2018), global GFC spillover.
        BSM errors here: Systematic underpricing of tail protection (puts very expensive).
        Implication for model: Expert 3 should show largest adjustments, especially for OTM puts.

IMPORTANT: The HMM is fit on DAILY data (one observation per trading day)
using data from the pre-training window (default 2018-2021 with Tier 1 data).
The model is then applied to all dates to produce regime probabilities. The regime
model is NOT retrained in the walk-forward evaluation — this would constitute look-ahead bias.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm

log = logging.getLogger(__name__)

OBSERVATION_COLUMNS = ["daily_log_return", "vix_close", "vix_1d_change"]
REGIME_NAMES_BY_VIX_RANK = [
    "LV-Trending",
    "LV-Mean-Reverting",
    "HV-Trending",
    "HV-Stress",
]
REGIME_COLORS = {
    "LV-Trending": "#2ecc71",
    "LV-Mean-Reverting": "#3498db",
    "HV-Trending": "#f39c12",
    "HV-Stress": "#e74c3c",
}
MARKET_EVENTS = [
    ("2018-10-01", "IL&FS crisis"),
    ("2020-03-23", "COVID crash"),
    ("2022-05-04", "RBI rate hikes"),
    ("2023-02-01", "Budget 2023"),
]


def _normalize_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    return out


def _build_aligned_frame(
    underlying_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Align underlying and VIX on the NSE trading calendar for one symbol."""
    und = _normalize_dates(underlying_df)
    und = und[und["symbol"] == symbol].sort_values("date").copy()

    vix = _normalize_dates(vix_df).sort_values("date").copy()
    if "vix_close" not in vix.columns:
        raise ValueError("vix_df must contain vix_close column")

    merged = und.merge(
        vix[
            [
                c
                for c in [
                    "date",
                    "vix_close",
                    "vix_change",
                    "vix_percentile_252d",
                ]
                if c in vix.columns
            ]
        ],
        on="date",
        how="inner",
    )

    if "returns_1d" in merged.columns:
        merged["daily_log_return"] = merged["returns_1d"]
    else:
        merged["daily_log_return"] = np.log(merged["close"] / merged["close"].shift(1))

    if "vix_change" in merged.columns:
        merged["vix_1d_change"] = merged["vix_change"]
    else:
        merged["vix_1d_change"] = merged["vix_close"].diff()

    merged = merged.dropna(subset=OBSERVATION_COLUMNS).reset_index(drop=True)
    if merged.empty:
        raise ValueError(f"No aligned observations for symbol={symbol}")

    return merged


def _standardize(
    raw: np.ndarray,
    scaler_params: dict[str, np.ndarray] | None = None,
    fit_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Standardise observation matrix; fit scaler on fit_rows when params absent."""
    if scaler_params is None:
        if fit_rows is None:
            fit_rows = np.ones(len(raw), dtype=bool)
        fit_data = raw[fit_rows]
        mean = fit_data.mean(axis=0)
        std = fit_data.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        scaler_params = {"mean": mean, "std": std}

    mean = scaler_params["mean"]
    std = scaler_params["std"]
    scaled = (raw - mean) / std
    return scaled.astype(np.float64), scaler_params


def prepare_hmm_observations(
    underlying_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    symbol: str = "NIFTY",
    scaler_params: dict[str, np.ndarray] | None = None,
    fit_mask: pd.Series | np.ndarray | None = None,
) -> tuple[np.ndarray, list[pd.Timestamp], dict[str, np.ndarray], pd.DataFrame]:
    """
    Constructs the 3-dimensional observation sequence for HMM fitting.

    Observation vector per day: [daily_log_return, vix_close, vix_1d_change]

    Pre-processing steps:
      1. Align underlying and VIX DataFrames on date (NSE trading calendar)
      2. Compute daily log returns from close prices
      3. Standardise each dimension to zero mean, unit variance BEFORE fitting
         (HMM Gaussian emissions work better with standardised features)
      4. Return the standardised observation matrix and the date list

    NOTE: Standardisation parameters are saved as HMM metadata so that
    at inference time, new observations are standardised the same way.
    When fit_mask is provided and scaler_params is None, the scaler is fit
    only on rows where fit_mask is True (avoids look-ahead in walk-forward).

    RETURNS:
        (X, dates, scaler_params, aligned_frame)
        X has shape (N_days, 3); dates is a list of N_days timestamps.
    """
    aligned = _build_aligned_frame(underlying_df, vix_df, symbol)
    raw = aligned[OBSERVATION_COLUMNS].to_numpy(dtype=np.float64)
    dates = aligned["date"].tolist()

    fit_rows: np.ndarray | None = None
    if fit_mask is not None:
        mask = np.asarray(fit_mask, dtype=bool)
        if len(mask) != len(aligned):
            raise ValueError("fit_mask length must match number of aligned trading days")
        fit_rows = mask

    X, scaler_params = _standardize(raw, scaler_params=scaler_params, fit_rows=fit_rows)
    return X, dates, scaler_params, aligned


def _plot_transition_heatmap(model: hmm.GaussianHMM, output_path: Path) -> None:
    """Save transition-matrix persistence heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trans = model.transmat_

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(trans, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("HMM Transition Matrix (Persistence)")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            ax.text(j, i, f"{trans[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved transition heatmap → %s", output_path)


def fit_hmm(
    X: np.ndarray,
    n_states: int = 4,
    n_iter: int = 200,
    n_restarts: int = 10,
    covariance_type: str = "full",
    random_seed: int = 42,
    artifacts_dir: Path | None = None,
) -> hmm.GaussianHMM:
    """
    Fits a Gaussian HMM with n_states states using the Baum-Welch EM algorithm.

    Multiple restarts (n_restarts=10) with different random initialisations address
    the non-convexity of the EM objective (HMM can get stuck in bad local optima).
    Returns the best model by log-likelihood across all restarts.

    Post-fitting validation:
      - Checks that all states are actually visited (no dead states)
      - Checks that the transition matrix has reasonable persistence (diagonal > 0.8)
      - Prints regime statistics (mean and variance of each state)
      - Saves a persistence heatmap as a PNG to artifacts/
    """
    if len(X) < n_states * 10:
        raise ValueError(f"Need at least {n_states * 10} observations for HMM fit, got {len(X)}")

    best_model: hmm.GaussianHMM | None = None
    best_ll = -np.inf
    rng = np.random.default_rng(random_seed)

    for restart in range(n_restarts):
        seed = int(rng.integers(0, 1_000_000))
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
            init_params="stmc",
            params="stmc",
        )
        try:
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception as exc:
            log.warning("HMM restart %d failed: %s", restart, exc)

    if best_model is None:
        raise RuntimeError("HMM fit failed on all initialisations")

    states = best_model.predict(X)
    visited = np.unique(states)
    if len(visited) < n_states:
        log.warning(
            "Dead states detected: only %d/%d states visited in training decode",
            len(visited),
            n_states,
        )

    diag = np.diag(best_model.transmat_)
    if np.any(diag < 0.5):
        log.warning("Low transition persistence on diagonal: %s", np.round(diag, 3))
    elif np.any(diag < 0.8):
        log.info("Moderate transition persistence (diagonal): %s", np.round(diag, 3))
    else:
        log.info("Good transition persistence (diagonal > 0.8): %s", np.round(diag, 3))

    log.info("Best HMM log-likelihood: %.2f", best_ll)
    for state in range(n_states):
        mean = best_model.means_[state]
        if best_model.covariance_type == "full":
            var = np.diag(best_model.covars_[state])
        elif best_model.covariance_type == "diag":
            var = best_model.covars_[state]
        else:
            var = np.full(len(mean), np.nan)
        log.info(
            "State %d — mean=%s var=%s",
            state,
            np.round(mean, 4),
            np.round(var, 4),
        )

    if artifacts_dir is not None:
        _plot_transition_heatmap(best_model, Path(artifacts_dir) / "hmm_transition_matrix.png")

    return best_model


def _state_to_regime_map(
    model: hmm.GaussianHMM,
    scaler_params: dict[str, np.ndarray],
    n_states: int,
) -> dict[int, int]:
    """Map raw HMM state indices to regime_id sorted by ascending mean VIX."""
    mean = scaler_params["mean"]
    std = scaler_params["std"]
    vix_idx = OBSERVATION_COLUMNS.index("vix_close")
    state_vix = {
        s: model.means_[s, vix_idx] * std[vix_idx] + mean[vix_idx] for s in range(n_states)
    }
    sorted_states = sorted(state_vix, key=state_vix.get)
    return {state: rank for rank, state in enumerate(sorted_states)}


def _remap_posteriors(
    posteriors: np.ndarray,
    state_to_regime: dict[int, int],
    n_states: int,
) -> np.ndarray:
    """Reorder posterior columns from HMM state order to VIX-sorted regime order."""
    remapped = np.zeros_like(posteriors)
    for state, regime in state_to_regime.items():
        remapped[:, regime] = posteriors[:, state]
    row_sums = remapped.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return remapped / row_sums


def label_regimes(
    model: hmm.GaussianHMM,
    X: np.ndarray,
    dates: list,
    symbol: str,
    scaler_params: dict[str, np.ndarray],
    aligned_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Applies the fitted HMM to produce hard labels (Viterbi) and soft probabilities.

    Soft posterior probabilities are remapped so regime_p0..p3 correspond to
    ascending VIX levels (LV-Trending → HV-Stress).

    RETURNS: DataFrame with columns
        [date, symbol, regime_id, regime_p0, regime_p1, regime_p2, regime_p3,
         vix_percentile, regime_name]
    """
    n_states = model.n_components
    state_to_regime = _state_to_regime_map(model, scaler_params, n_states)
    regime_names = {regime: REGIME_NAMES_BY_VIX_RANK[regime] for regime in range(n_states)}

    raw_states = model.predict(X)
    _, posteriors = model.score_samples(X)
    remapped_states = np.array([state_to_regime[s] for s in raw_states])
    remapped_probs = _remap_posteriors(posteriors, state_to_regime, n_states)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "symbol": symbol,
            "regime_id": remapped_states,
            "regime_p0": remapped_probs[:, 0],
            "regime_p1": remapped_probs[:, 1],
            "regime_p2": remapped_probs[:, 2],
            "regime_p3": remapped_probs[:, 3],
        }
    )

    if aligned_frame is not None and "vix_percentile_252d" in aligned_frame.columns:
        out["vix_percentile"] = aligned_frame["vix_percentile_252d"].values
    else:
        out["vix_percentile"] = np.nan

    out["regime_name"] = out["regime_id"].map(regime_names)
    return out


def save_hmm_artifact(
    model: hmm.GaussianHMM,
    scaler_params: dict[str, np.ndarray],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save fitted HMM and standardisation parameters to a pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler_params,
        "observation_columns": OBSERVATION_COLUMNS,
        "metadata": metadata or {},
    }
    with output_path.open("wb") as fh:
        pickle.dump(payload, fh)
    log.info("Saved HMM artifact → %s", output_path)


def load_hmm_artifact(
    artifact_path: Path,
) -> tuple[hmm.GaussianHMM, dict[str, np.ndarray], dict[str, Any]]:
    """Load HMM model, scaler params, and metadata from pickle."""
    with Path(artifact_path).open("rb") as fh:
        payload = pickle.load(fh)
    return payload["model"], payload["scaler"], payload.get("metadata", {})


def plot_regime_timeline(
    regime_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    output_dir: Path,
    symbol: str = "NIFTY",
) -> None:
    """
    Three-panel timeline: NIFTY price with regime shading, VIX, stacked regime probs.

    Annotates key Indian market events on the timeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reg = regime_df.copy()
    reg["date"] = pd.to_datetime(reg["date"])
    if "symbol" in reg.columns:
        reg = reg[reg["symbol"] == symbol]

    und = _normalize_dates(underlying_df)
    und = und[und["symbol"] == symbol].sort_values("date")
    vix = _normalize_dates(vix_df).sort_values("date")

    merged = reg.merge(und[["date", "close"]], on="date", how="left")
    merged = merged.merge(vix[["date", "vix_close"]], on="date", how="left")

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})

    ax0 = axes[0]
    dates = merged["date"]
    ax0.plot(dates, merged["close"], color="black", linewidth=0.9, label=f"{symbol} close")

    for regime_name in REGIME_NAMES_BY_VIX_RANK:
        mask = merged["regime_name"] == regime_name
        if not mask.any():
            continue
        color = REGIME_COLORS.get(regime_name, "#cccccc")
        starts = np.where(mask & ~mask.shift(1, fill_value=False))[0]
        ends = np.where(mask & ~mask.shift(-1, fill_value=False))[0]
        for start_idx in starts:
            end_idx = ends[ends >= start_idx][0]
            ax0.axvspan(dates.iloc[start_idx], dates.iloc[end_idx], color=color, alpha=0.25)

    ax0.set_ylabel("Index level")
    ax0.set_title(f"{symbol} price coloured by HMM regime")
    ax0.legend(loc="upper left")

    ax1 = axes[1]
    ax1.plot(vix["date"], vix["vix_close"], color="#8e44ad", linewidth=0.9)
    for level, label in [(14, "VIX 14"), (18, "VIX 18"), (25, "VIX 25")]:
        ax1.axhline(level, linestyle="--", linewidth=0.8, alpha=0.7, label=label)
    ax1.set_ylabel("India VIX")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = axes[2]
    prob_cols = ["regime_p0", "regime_p1", "regime_p2", "regime_p3"]
    colors = [REGIME_COLORS[n] for n in REGIME_NAMES_BY_VIX_RANK]
    bottom = np.zeros(len(merged))
    for col, name, color in zip(prob_cols, REGIME_NAMES_BY_VIX_RANK, colors):
        vals = merged[col].fillna(0).values
        ax2.fill_between(dates, bottom, bottom + vals, label=name, color=color, alpha=0.85, linewidth=0)
        bottom = bottom + vals
    ax2.set_ylabel("Regime prob.")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=8, ncol=2)

    for event_date, label in MARKET_EVENTS:
        ts = pd.Timestamp(event_date)
        if ts < dates.min() or ts > dates.max():
            continue
        for ax in axes:
            ax.axvline(ts, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
        ax0.annotate(
            label,
            xy=(ts, ax0.get_ylim()[1]),
            xytext=(4, -8),
            textcoords="offset points",
            fontsize=8,
            rotation=90,
            va="top",
        )

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out_path = output_dir / f"regime_timeline_{symbol.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved regime timeline → %s", out_path)
