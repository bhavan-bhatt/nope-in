"""
Data Quality Validation Framework.

Schema and cross-dataset validation for all NOPE-IN datasets.
Uses pandas checks with optional Great Expectations when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from nope_in.exceptions import DataQualityError

log = logging.getLogger(__name__)

try:
    import great_expectations as gx  # noqa: F401

    HAS_GE = True
except ImportError:
    HAS_GE = False


@dataclass
class ValidationResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", critical: bool = True) -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail, "critical": critical})
        if not passed and critical:
            self.passed = False
        elif not passed:
            self.warnings.append(f"{name}: {detail}")


def validate_options_chain(df: pd.DataFrame) -> ValidationResult:
    """Validate options_chain DataFrame."""
    result = ValidationResult(passed=True)

    required = {
        "date",
        "symbol",
        "expiry_date",
        "strike",
        "option_type",
        "settle_price",
    }
    missing = required - set(df.columns)
    result.add("required_columns", len(missing) == 0, f"missing={missing}")

    if "settle_price" in df.columns:
        bad_settle = (df["settle_price"] <= 0).sum()
        result.add("settle_price_positive", bad_settle == 0, f"{bad_settle} rows <= 0")

    if "strike" in df.columns:
        bad_strike = (df["strike"] <= 0).sum()
        result.add("strike_positive", bad_strike == 0, f"{bad_strike} rows <= 0")

    if "option_type" in df.columns:
        invalid = ~df["option_type"].isin(["CE", "PE"])
        result.add("option_type_valid", invalid.sum() == 0, f"{invalid.sum()} invalid")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"]).sort_values().unique()
        if len(dates) > 1:
            gaps = pd.Series(dates).diff().dt.days
            max_gap = gaps.max()
            result.add(
                "date_gaps",
                max_gap <= 10,
                f"max gap {max_gap} calendar days",
                critical=True,
            )

    return result


def validate_underlying_ohlcv(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult(passed=True)
    result.add("has_close", "close" in df.columns)
    if "close" in df.columns:
        result.add("close_positive", (df["close"] <= 0).sum() == 0)
    if "date" in df.columns and "symbol" in df.columns:
        monotonic = all(
            grp.sort_values("date")["date"].is_monotonic_increasing
            for _, grp in df.groupby("symbol")
        )
        result.add("date_monotonic_per_symbol", monotonic)
    elif "date" in df.columns:
        result.add("date_monotonic", df["date"].is_monotonic_increasing)
    return result


def validate_india_vix(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult(passed=True)
    result.add("has_vix_close", "vix_close" in df.columns)
    if "vix_close" in df.columns:
        result.add("vix_close_positive", (df["vix_close"] <= 0).sum() == 0)
    return result


def validate_rates(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None) -> ValidationResult:
    result = ValidationResult(passed=True)
    result.add("has_tbill_91d", "tbill_91d" in df.columns)

    window = df.copy()
    if start:
        window = window[window["date"] >= pd.Timestamp(start)]
    if end:
        window = window[window["date"] <= pd.Timestamp(end)]

    if "tbill_91d" in window.columns:
        nan_count = window["tbill_91d"].isna().sum()
        result.add("tbill_91d_complete", nan_count == 0, f"{nan_count} NaN in window")
    return result


def validate_cross_dataset_consistency(
    options_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    vix_df: pd.DataFrame,
) -> ValidationResult:
    result = ValidationResult(passed=True)

    opt_dates = set(pd.to_datetime(options_df["date"]).dt.normalize())
    und_dates = set(pd.to_datetime(underlying_df["date"]).dt.normalize())
    vix_dates = set(pd.to_datetime(vix_df["date"]).dt.normalize())

    missing_underlying = opt_dates - und_dates
    result.add(
        "options_subset_underlying",
        len(missing_underlying) == 0,
        f"{len(missing_underlying)} option dates missing from underlying",
    )

    missing_vix = opt_dates - vix_dates
    result.add(
        "vix_covers_options",
        len(missing_vix) == 0,
        f"{len(missing_vix)} option dates missing from VIX",
    )

    return result


def validate_all(
    processed_dir: Path,
    start: str,
    end: str,
    halt_on_critical: bool = True,
) -> ValidationResult:
    """Run all validations on processed parquet files."""
    processed_dir = Path(processed_dir)
    combined = ValidationResult(passed=True)

    options_path = processed_dir / "options_chain.parquet"
    underlying_path = processed_dir / "underlying_ohlcv.parquet"
    vix_path = processed_dir / "india_vix.parquet"
    rates_path = processed_dir / "rates.parquet"

    if options_path.exists():
        r = validate_options_chain(pd.read_parquet(options_path))
        combined.checks.extend(r.checks)
        combined.warnings.extend(r.warnings)
        if not r.passed:
            combined.passed = False

    if underlying_path.exists():
        r = validate_underlying_ohlcv(pd.read_parquet(underlying_path))
        combined.checks.extend(r.checks)
        combined.warnings.extend(r.warnings)
        if not r.passed:
            combined.passed = False

    if vix_path.exists():
        r = validate_india_vix(pd.read_parquet(vix_path))
        combined.checks.extend(r.checks)
        combined.warnings.extend(r.warnings)
        if not r.passed:
            combined.passed = False

    if rates_path.exists():
        r = validate_rates(pd.read_parquet(rates_path), start, end)
        combined.checks.extend(r.checks)
        combined.warnings.extend(r.warnings)
        if not r.passed:
            combined.passed = False

    if options_path.exists() and underlying_path.exists() and vix_path.exists():
        r = validate_cross_dataset_consistency(
            pd.read_parquet(options_path),
            pd.read_parquet(underlying_path),
            pd.read_parquet(vix_path),
        )
        combined.checks.extend(r.checks)
        combined.warnings.extend(r.warnings)
        if not r.passed:
            combined.passed = False

    for check in combined.checks:
        status = "PASS" if check["passed"] else "FAIL"
        log.info("[%s] %s — %s", status, check["name"], check.get("detail", ""))

    for warning in combined.warnings:
        log.warning("Validation warning: %s", warning)

    if halt_on_critical and not combined.passed:
        failed = [c["name"] for c in combined.checks if not c["passed"] and c.get("critical", True)]
        raise DataQualityError(f"Critical validation failures: {failed}")

    return combined


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Validate NOPE-IN processed datasets")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-12-31")
    args = parser.parse_args()
    validate_all(Path(args.processed_dir), args.start, args.end)


if __name__ == "__main__":
    main()
