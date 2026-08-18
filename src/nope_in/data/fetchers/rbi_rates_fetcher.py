"""
RBI Interest Rates Fetcher.

Loads manually curated RBI rate data and builds daily rates.parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RATE_COLUMNS = [
    "date",
    "repo_rate",
    "reverse_repo_rate",
    "tbill_91d",
    "tbill_182d",
    "tbill_364d",
    "gsec_2y",
    "gsec_10y",
]

# Approximate fallback when seed CSV is missing (decimal rates)
_FALLBACK_SCHEDULE = [
    ("2018-01-01", 0.063, 0.060, 0.063, 0.0645, 0.0655, 0.072, 0.074),
    ("2018-06-01", 0.065, 0.062, 0.065, 0.066, 0.067, 0.0735, 0.0755),
    ("2019-02-01", 0.065, 0.062, 0.064, 0.065, 0.066, 0.071, 0.075),
    ("2019-08-01", 0.058, 0.055, 0.055, 0.056, 0.057, 0.062, 0.068),
    ("2020-03-01", 0.045, 0.040, 0.045, 0.046, 0.047, 0.055, 0.065),
    ("2020-06-01", 0.040, 0.035, 0.038, 0.039, 0.040, 0.050, 0.060),
    ("2021-05-01", 0.040, 0.035, 0.035, 0.036, 0.037, 0.052, 0.061),
    ("2022-05-01", 0.048, 0.040, 0.048, 0.050, 0.051, 0.065, 0.073),
    ("2022-10-01", 0.065, 0.035, 0.065, 0.066, 0.067, 0.072, 0.074),
    ("2023-02-01", 0.072, 0.035, 0.072, 0.073, 0.074, 0.071, 0.073),
    ("2024-06-01", 0.069, 0.035, 0.069, 0.070, 0.071, 0.070, 0.072),
]


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def load_rbi_tbill_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Parse RBI rate CSV, convert percentages to decimals, forward-fill to daily.
    """
    df = pd.read_csv(csv_path)
    df["date"] = _parse_mixed_dates(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    pct_cols = [
        c
        for c in df.columns
        if c != "date" and df[c].dtype in ("float64", "int64", "float32", "int32")
    ]
    for col in pct_cols:
        if df[col].max() > 1.0:
            df[col] = df[col] / 100.0

    df = df.set_index("date").sort_index()
    daily = df.resample("B").ffill().reset_index()
    daily["yield_spread_2y10y"] = daily["gsec_10y"] - daily["gsec_2y"]
    return daily


def _build_fallback_rates(start: str, end: str) -> pd.DataFrame:
    """Build approximate daily rates from piecewise schedule."""
    dates = pd.bdate_range(start, end)
    rows = []
    for d in dates:
        row = {"date": d}
        for i, (cutoff, *rates) in enumerate(_FALLBACK_SCHEDULE):
            cutoff_ts = pd.Timestamp(cutoff)
            next_cutoff = (
                pd.Timestamp(_FALLBACK_SCHEDULE[i + 1][0])
                if i + 1 < len(_FALLBACK_SCHEDULE)
                else d + pd.Timedelta(days=1)
            )
            if cutoff_ts <= d < next_cutoff:
                keys = [
                    "repo_rate",
                    "reverse_repo_rate",
                    "tbill_91d",
                    "tbill_182d",
                    "tbill_364d",
                    "gsec_2y",
                    "gsec_10y",
                ]
                row.update(dict(zip(keys, rates)))
                break
        rows.append(row)

    df = pd.DataFrame(rows)
    df["yield_spread_2y10y"] = df["gsec_10y"] - df["gsec_2y"]
    return df


def build_rates_parquet(
    manual_path: Path,
    output_path: Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Build daily rates DataFrame from manual CSV or fallback schedule.
    """
    if manual_path.exists():
        df = load_rbi_tbill_from_csv(manual_path)
        log.info("Loaded RBI rates from %s", manual_path)
    else:
        log.warning("Manual RBI CSV not found at %s; using fallback schedule", manual_path)
        df = _build_fallback_rates(start, end)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()

    if df["tbill_91d"].isna().any():
        raise ValueError("Missing tbill_91d values in training window")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Wrote rates.parquet: %d rows", len(df))
    return df
