"""
Black-Scholes-Merton Analytical Pricer — From Scratch.

Implements BSM option pricing and all Greeks analytically using scipy.stats.norm.
Written from scratch (not py_vollib / QuantLib) for exact control over dividend yield,
rate interpolation, and numerical stability at deep ITM/OTM strikes.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

log = logging.getLogger(__name__)

ArrayLike = Union[float, np.ndarray]

MIN_SIGMA = 0.001
DEFAULT_Q = 0.013
DAYS_PER_YEAR = 365.0


def _as_array(*args: ArrayLike) -> tuple[np.ndarray, ...]:
    """Broadcast scalar/array inputs to a common numpy shape."""
    arrays = [np.asarray(a, dtype=float) for a in args]
    if len(arrays) == 1:
        return (arrays[0],)
    return tuple(np.broadcast_arrays(*arrays))


def _is_call(option_type: Union[str, np.ndarray]) -> np.ndarray:
    ot = np.asarray(option_type)
    if ot.dtype.kind in {"U", "S", "O"}:
        return np.char.upper(ot.astype(str)) == "CE"
    return ot.astype(bool)


def bsm_d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: float = DEFAULT_Q,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute BSM d1 and d2 with edge-case handling for T≈0 and sigma≈0.
    """
    S_arr, K_arr, T_arr, r_arr, sig_arr = _as_array(S, K, T, r, sigma)
    if np.any(S_arr <= 0) or np.any(K_arr <= 0):
        raise ValueError("Spot and strike must be strictly positive")

    sig_safe = np.maximum(sig_arr, MIN_SIGMA)
    sqrt_t = np.sqrt(np.maximum(T_arr, 0.0))
    log_m = np.log(S_arr / K_arr)

    d1 = (log_m + (r_arr - q + 0.5 * sig_safe**2) * T_arr) / (sig_safe * sqrt_t + 1e-15)
    d2 = d1 - sig_safe * sqrt_t

    # Expiry-day limit: intrinsic moneyness determines ±inf
    expiry_mask = T_arr <= 1e-10
    if np.any(expiry_mask):
        itm_call = S_arr > K_arr
        d1 = np.where(expiry_mask & itm_call, np.inf, d1)
        d1 = np.where(expiry_mask & ~itm_call, -np.inf, d1)
        d2 = np.where(expiry_mask, d1, d2)

    if np.ndim(d1) == 0:
        return float(d1), float(d2)
    return d1, d2


def bsm_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: Union[str, np.ndarray],
    q: float = DEFAULT_Q,
) -> ArrayLike:
    """Compute BSM theoretical price for European CE/PE options (vectorised)."""
    S_arr, K_arr, T_arr, r_arr, sig_arr = _as_array(S, K, T, r, sigma)
    is_call = _is_call(option_type)

    intrinsic_call = np.maximum(S_arr - K_arr, 0.0)
    intrinsic_put = np.maximum(K_arr - S_arr, 0.0)
    intrinsic = np.where(is_call, intrinsic_call, intrinsic_put)

    near_expiry = T_arr <= 1e-10
    near_zero_vol = sig_arr <= MIN_SIGMA
    fallback = near_expiry | near_zero_vol
    if np.all(fallback):
        return intrinsic if np.ndim(intrinsic) else float(intrinsic)

    d1, d2 = bsm_d1_d2(S_arr, K_arr, T_arr, r_arr, sig_arr, q=q)
    disc_s = np.exp(-q * T_arr)
    disc_k = np.exp(-r_arr * T_arr)

    call_price = disc_s * S_arr * norm.cdf(d1) - disc_k * K_arr * norm.cdf(d2)
    put_price = disc_k * K_arr * norm.cdf(-d2) - disc_s * S_arr * norm.cdf(-d1)
    price = np.where(is_call, call_price, put_price)
    price = np.where(fallback, intrinsic, price)
    price = np.maximum(price, 0.0)

    if np.ndim(price) == 0:
        return float(price)
    return price


def bsm_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: Union[str, np.ndarray],
    q: float = DEFAULT_Q,
) -> dict[str, ArrayLike]:
    """
    Compute first- and second-order BSM Greeks analytically.

    Theta is per calendar day (negative). Vega is for a 1 percentage-point vol move.
    """
    S_arr, K_arr, T_arr, r_arr, sig_arr = _as_array(S, K, T, r, sigma)
    is_call = _is_call(option_type)

    intrinsic_call = np.maximum(S_arr - K_arr, 0.0)
    intrinsic_put = np.maximum(K_arr - S_arr, 0.0)

    d1, d2 = bsm_d1_d2(S_arr, K_arr, T_arr, r_arr, sig_arr, q=q)
    sqrt_t = np.sqrt(np.maximum(T_arr, 1e-10))
    disc_s = np.exp(-q * T_arr)
    disc_k = np.exp(-r_arr * T_arr)
    pdf_d1 = norm.pdf(d1)

    delta = disc_s * norm.cdf(d1)
    delta = np.where(is_call, delta, delta - disc_s)

    gamma = disc_s * pdf_d1 / (S_arr * sig_arr * sqrt_t + 1e-15)

    vega_raw = S_arr * disc_s * pdf_d1 * sqrt_t
    vega = vega_raw * 0.01  # 1 percentage-point convention (NSE)

    theta_call = (
        -S_arr * disc_s * pdf_d1 * sig_arr / (2.0 * sqrt_t)
        - r_arr * K_arr * disc_k * norm.cdf(d2)
        + q * S_arr * disc_s * norm.cdf(d1)
    ) / DAYS_PER_YEAR
    theta_put = (
        -S_arr * disc_s * pdf_d1 * sig_arr / (2.0 * sqrt_t)
        + r_arr * K_arr * disc_k * norm.cdf(-d2)
        - q * S_arr * disc_s * norm.cdf(-d1)
    ) / DAYS_PER_YEAR
    theta = np.where(is_call, theta_call, theta_put)

    rho_call = K_arr * T_arr * disc_k * norm.cdf(d2) * 0.01
    rho_put = -K_arr * T_arr * disc_k * norm.cdf(-d2) * 0.01
    rho = np.where(is_call, rho_call, rho_put)

    vanna = -disc_s * pdf_d1 * d2 / (sig_arr + 1e-15)
    volga = vega_raw * d1 * d2 / (sig_arr + 1e-15) * 0.01

    zero_t = T_arr <= 1e-10
    zero_sig = sig_arr <= MIN_SIGMA
    zero_mask = zero_t | zero_sig

    delta = np.where(zero_mask & is_call, np.where(S_arr > K_arr, disc_s, 0.0), delta)
    delta = np.where(zero_mask & ~is_call, np.where(S_arr < K_arr, -disc_s, 0.0), delta)
    gamma = np.where(zero_mask, 0.0, gamma)
    vega = np.where(zero_mask, 0.0, vega)
    theta = np.where(zero_mask, 0.0, theta)
    rho = np.where(zero_mask, 0.0, rho)
    vanna = np.where(zero_mask, 0.0, vanna)
    volga = np.where(zero_mask, 0.0, volga)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "vanna": vanna,
        "volga": volga,
    }


def _iv_objective(
    sigma: float,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float,
) -> float:
    return float(
        bsm_price(S, K, T, r, sigma, option_type, q=q) - market_price
    )


def _solve_iv_scalar(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float,
    initial_guess: float,
    max_iterations: int,
    tolerance: float,
) -> float:
    if market_price <= 0 or T <= 1e-10:
        return float("nan")

    is_call = str(option_type).upper() == "CE"
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if market_price < intrinsic - tolerance:
        return float("nan")
    if market_price <= intrinsic + tolerance:
        return 0.0

    lo, hi = MIN_SIGMA, 20.0
    try:
        f_lo = _iv_objective(lo, market_price, S, K, T, r, option_type, q)
        f_hi = _iv_objective(hi, market_price, S, K, T, r, option_type, q)
        if f_lo * f_hi > 0:
            # Newton-Raphson fallback
            sigma = initial_guess
            for _ in range(max_iterations):
                price = float(bsm_price(S, K, T, r, sigma, option_type, q=q))
                diff = price - market_price
                if abs(diff) < tolerance:
                    return max(sigma, 0.0)
                greeks = bsm_greeks(S, K, T, r, sigma, option_type, q=q)
                vega_raw = float(greeks["vega"]) / 0.01
                if abs(vega_raw) < 1e-12:
                    break
                sigma -= diff / vega_raw
                sigma = min(max(sigma, MIN_SIGMA), hi)
            return float("nan")
        return float(
            brentq(
                lambda s: _iv_objective(s, market_price, S, K, T, r, option_type, q),
                lo,
                hi,
                xtol=tolerance,
                maxiter=max_iterations,
            )
        )
    except (ValueError, RuntimeError):
        return float("nan")


def implied_vol_from_price(
    market_price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    option_type: Union[str, np.ndarray],
    q: float = DEFAULT_Q,
    initial_guess: float = 0.20,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    chunk_size: int = 250_000,
) -> ArrayLike:
    """Invert BSM to find implied volatility (vectorised Newton-Raphson with Brent fallback)."""
    mp = np.atleast_1d(np.asarray(market_price, dtype=float))
    s, k, t, rate, mp = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        mp,
    )
    scalar_input = np.ndim(market_price) == 0

    if scalar_input:
        return _solve_iv_scalar(
            float(mp[0]), float(s[0]), float(k[0]), float(t[0]), float(rate[0]),
            str(option_type), q, initial_guess, max_iterations, tolerance,
        )

    if mp.size <= chunk_size:
        return _implied_vol_vectorized(
            mp, s, k, t, rate, option_type, q, initial_guess, max_iterations, tolerance
        )

    out = np.empty(mp.shape, dtype=float)
    ot_full = None if isinstance(option_type, str) else np.asarray(option_type).ravel()
    for start in range(0, mp.size, chunk_size):
        end = min(start + chunk_size, mp.size)
        sl = slice(start, end)
        ot_chunk = option_type if isinstance(option_type, str) else ot_full[sl]
        out[sl] = _implied_vol_vectorized(
            mp[sl], s[sl], k[sl], t[sl], rate[sl], ot_chunk, q,
            initial_guess, max_iterations, tolerance,
        )
    return out


def _implied_vol_vectorized(
    mp: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    rate: np.ndarray,
    option_type: Union[str, np.ndarray],
    q: float,
    initial_guess: float,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Vectorised Newton-Raphson IV solver for a single chunk."""
    sigma = np.full(mp.shape, initial_guess, dtype=float)
    result = np.full(mp.shape, np.nan, dtype=float)

    if isinstance(option_type, str):
        is_call = np.full(mp.shape, option_type.upper() == "CE", dtype=bool)
    else:
        is_call = np.asarray(_is_call(option_type), dtype=bool)
        is_call = np.broadcast_to(is_call, mp.shape)

    intrinsic = np.where(is_call, np.maximum(s - k, 0.0), np.maximum(k - s, 0.0))

    valid = (mp > 0) & (t > 1e-10) & (mp >= intrinsic - tolerance)
    invalid = ~valid
    result[invalid] = np.nan
    result[valid & (mp <= intrinsic + tolerance)] = 0.0

    work = valid & (mp > intrinsic + tolerance)
    if not np.any(work):
        return result

    sigma_w = sigma[work].copy()
    mp_w = mp[work]
    s_w, k_w, t_w, r_w = s[work], k[work], t[work], rate[work]
    ot_w = np.where(is_call[work], "CE", "PE")

    bulk_iters = min(max_iterations, 25)
    for _ in range(bulk_iters):
        price = bsm_price(s_w, k_w, t_w, r_w, sigma_w, ot_w, q=q)
        diff = price - mp_w
        if np.max(np.abs(diff)) < tolerance:
            break
        vega_raw = np.asarray(bsm_greeks(s_w, k_w, t_w, r_w, sigma_w, ot_w, q=q)["vega"]) / 0.01
        sigma_w = np.clip(sigma_w - diff / np.maximum(vega_raw, 1e-12), MIN_SIGMA, 20.0)

    result[work] = sigma_w

    bad_idx = np.where(work)[0]
    price_check = bsm_price(s_w, k_w, t_w, r_w, sigma_w, ot_w, q=q)
    still_bad = np.abs(price_check - mp_w) > tolerance * 10
    for i in np.where(still_bad)[0]:
        iv = _solve_iv_scalar(
            float(mp_w[i]), float(s_w[i]), float(k_w[i]), float(t_w[i]), float(r_w[i]),
            str(ot_w[i]), q, float(sigma_w[i]), max_iterations, tolerance,
        )
        result[bad_idx[i]] = iv

    return result


def interpolate_rate_to_dte(
    dte: ArrayLike,
    tbill_91d: ArrayLike,
    tbill_182d: ArrayLike | None = None,
    tbill_364d: ArrayLike | None = None,
) -> ArrayLike:
    """
    Interpolate risk-free rate to contract DTE using T-bill tenors (91/182/364 days).

    Falls back to flat 91-day rate when longer tenors are unavailable.
    """
    dte_arr = np.asarray(dte, dtype=float)
    r91 = np.asarray(tbill_91d, dtype=float)
    r182 = np.asarray(tbill_182d if tbill_182d is not None else tbill_91d, dtype=float)
    r364 = np.asarray(tbill_364d if tbill_364d is not None else tbill_91d, dtype=float)

    r91, r182, r364, dte_arr = np.broadcast_arrays(r91, r182, r364, dte_arr)
    tenors = np.array([91.0, 182.0, 364.0])
    out = np.empty_like(dte_arr, dtype=float)
    for idx in np.ndindex(dte_arr.shape):
        rates = [float(r91[idx]), float(r182[idx]), float(r364[idx])]
        out[idx] = np.interp(float(dte_arr[idx]), tenors, rates)
    if np.ndim(out) == 0:
        return float(out)
    return out


def _prepare_options_frame(
    options_df: pd.DataFrame,
    underlying_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"date", "symbol", "expiry_date", "strike", "option_type", "settle_price"}
    missing = required - set(options_df.columns)
    if missing:
        raise ValueError(f"options_df missing columns: {sorted(missing)}")

    df = options_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.normalize()
    df["dte"] = (df["expiry_date"] - df["date"]).dt.days.astype(int)

    if "underlying_value" in df.columns and df["underlying_value"].notna().any():
        df["spot"] = pd.to_numeric(df["underlying_value"], errors="coerce")
    elif "spot" in df.columns:
        df["spot"] = pd.to_numeric(df["spot"], errors="coerce")
    elif underlying_df is not None:
        und = underlying_df.copy()
        und["date"] = pd.to_datetime(und["date"]).dt.normalize()
        spot_map = und[["date", "symbol", "close"]].rename(columns={"close": "spot"})
        df = df.merge(spot_map, on=["date", "symbol"], how="left")
    else:
        raise ValueError(
            "options_df must contain underlying_value/spot, or pass underlying_df to join close"
        )

    if df["spot"].isna().any():
        n_missing = int(df["spot"].isna().sum())
        log.warning("Dropping %d rows with missing spot after underlying join", n_missing)
        df = df.dropna(subset=["spot"])

    df["moneyness"] = df["spot"] / df["strike"]
    df["T"] = df["dte"] / DAYS_PER_YEAR
    return df


def _join_rates(options_df: pd.DataFrame, rates_df: pd.DataFrame) -> pd.DataFrame:
    rates = rates_df.copy()
    rates["date"] = pd.to_datetime(rates["date"]).dt.normalize()
    merged = options_df.merge(
        rates[
            [
                c
                for c in ["date", "tbill_91d", "tbill_182d", "tbill_364d", "yield_spread_2y10y"]
                if c in rates.columns
            ]
        ],
        on="date",
        how="left",
    )
    merged["rate_interpolated_dte"] = interpolate_rate_to_dte(
        merged["dte"].values,
        merged["tbill_91d"].values,
        merged.get("tbill_182d", merged["tbill_91d"]).values,
        merged.get("tbill_364d", merged["tbill_91d"]).values,
    )
    return merged


def compute_bsm_residual(
    options_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    underlying_df: pd.DataFrame | None = None,
    q: float = DEFAULT_Q,
) -> pd.DataFrame:
    """
    Enrich options chain with IV, BSM price, Greeks, and filtered training rows.
    """
    df = _prepare_options_frame(options_df, underlying_df=underlying_df)
    n0 = len(df)
    df = _join_rates(df, rates_df)

    df["implied_vol"] = implied_vol_from_price(
        df["settle_price"].values,
        df["spot"].values,
        df["strike"].values,
        df["T"].values,
        df["rate_interpolated_dte"].values,
        df["option_type"].values,
        q=q,
    )

    df["bsm_price"] = bsm_price(
        df["spot"].values,
        df["strike"].values,
        df["T"].values,
        df["rate_interpolated_dte"].values,
        df["implied_vol"].fillna(0.2).values,
        df["option_type"].values,
        q=q,
    )

    greeks = bsm_greeks(
        df["spot"].values,
        df["strike"].values,
        df["T"].values,
        df["rate_interpolated_dte"].values,
        df["implied_vol"].fillna(0.2).values,
        df["option_type"].values,
        q=q,
    )
    for name, values in greeks.items():
        df[name] = values

    df["bsm_error"] = df["settle_price"] - df["bsm_price"]
    df["bsm_error_norm"] = df["bsm_error"] / df["vega"].replace(0, np.nan)

    # Contract-level derived features
    df["moneyness_squared"] = df["moneyness"] ** 2
    df["log_dte"] = np.log1p(df["dte"].clip(lower=0))
    df["sqrt_dte"] = np.sqrt(df["dte"].clip(lower=0))
    df["inv_dte"] = 1.0 / df["dte"].clip(lower=1)
    df["is_weekly_expiry"] = df["dte"] <= 7
    df["is_monthly_expiry"] = df["dte"] > 7

    # Filtering pipeline
    def _filter(name: str, mask: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
        dropped = (~mask).sum()
        if dropped:
            log.info("BSM filter '%s': dropped %d / %d rows", name, dropped, len(frame))
        return frame.loc[mask].copy()

    vol_col = "contracts" if "contracts" in df.columns else None
    oi_col = "open_interest" if "open_interest" in df.columns else None
    if vol_col and oi_col:
        df = _filter(
            "zero_volume_and_oi",
            ~((df[vol_col] == 0) & (df[oi_col] == 0)),
            df,
        )

    df = _filter("dte_zero", df["dte"] > 0, df)
    df = _filter("iv_nan", df["implied_vol"].notna(), df)

    atm_iv = (
        df.assign(strike_dist=(df["strike"] - df["spot"]).abs())
        .sort_values(["date", "symbol", "strike_dist"])
        .groupby(["date", "symbol"], as_index=False)
        .first()[["date", "symbol", "implied_vol"]]
        .rename(columns={"implied_vol": "atm_iv"})
    )
    df = df.merge(atm_iv, on=["date", "symbol"], how="left")
    outlier_mask = df["bsm_error"].abs() <= 5.0 * df["atm_iv"].fillna(0.2)
    df = _filter("bsm_error_outlier", outlier_mask, df)
    df = _filter("moneyness_band", (df["moneyness"] >= 0.70) & (df["moneyness"] <= 1.30), df)

    log.info("compute_bsm_residual: %d → %d rows (from %d)", len(df), len(df), n0)
    return df.reset_index(drop=True)


def validate_put_call_parity(
    options_df: pd.DataFrame,
    tolerance_pct: float = 0.01,
    q: float = DEFAULT_Q,
) -> pd.DataFrame:
    """
    Validate put-call parity: C - P ≈ S·e^{-qT} - K·e^{-rT}.
    Returns violations sorted by relative magnitude.
    """
    df = _prepare_options_frame(options_df)
    if "rate_interpolated_dte" not in df.columns:
        raise ValueError("options_df must include rate_interpolated_dte (run compute_bsm_residual first)")

    calls = df[df["option_type"] == "CE"][
        ["date", "symbol", "expiry_date", "strike", "settle_price", "spot", "T", "rate_interpolated_dte"]
    ].rename(columns={"settle_price": "call_price"})
    puts = df[df["option_type"] == "PE"][
        ["date", "symbol", "expiry_date", "strike", "settle_price"]
    ].rename(columns={"settle_price": "put_price"})

    pairs = calls.merge(puts, on=["date", "symbol", "expiry_date", "strike"], how="inner")
    if pairs.empty:
        return pd.DataFrame(columns=["date", "symbol", "expiry_date", "strike", "parity_error", "parity_error_pct"])

    forward = pairs["spot"] * np.exp(-q * pairs["T"]) - pairs["strike"] * np.exp(
        -pairs["rate_interpolated_dte"] * pairs["T"]
    )
    observed = pairs["call_price"] - pairs["put_price"]
    pairs["parity_error"] = observed - forward
    pairs["parity_error_pct"] = pairs["parity_error"].abs() / pairs["spot"].clip(lower=1.0)

    violations = pairs[pairs["parity_error_pct"] > tolerance_pct].sort_values(
        "parity_error_pct", ascending=False
    )
    if len(violations):
        log.warning("Put-call parity: %d violations above %.2f%%", len(violations), tolerance_pct * 100)
    return violations.reset_index(drop=True)
