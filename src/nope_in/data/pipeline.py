"""
NOPE-IN data pipeline orchestrator (Hydra-driven).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from nope_in.data.fetchers.bhav_copy_fetcher import download_bhav_range, merge_bhav_to_parquet
from nope_in.data.fetchers.events_calendar_fetcher import build_events_calendar
from nope_in.data.fetchers.rbi_rates_fetcher import build_rates_parquet
from nope_in.data.fetchers.yfinance_fetcher import download_all_indices
from nope_in.data.validators.schema_validator import validate_all

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(cfg: DictConfig, key: str) -> Path:
    return PROJECT_ROOT / cfg.data.paths[key]


def run_pipeline(cfg: DictConfig) -> None:
    """Execute full data acquisition pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw_dir = _resolve_path(cfg, "raw_dir")
    manual_dir = _resolve_path(cfg, "manual_dir")
    processed_dir = _resolve_path(cfg, "processed_dir")

    start = cfg.data.start_date
    end = cfg.data.end_date
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    symbols = list(cfg.data.symbols)

    log.info("NOPE-IN data pipeline: %s → %s", start, end)

    # 1. yfinance indices + VIX
    download_all_indices(OmegaConf.to_container(cfg.data, resolve=True), raw_dir)

    # 2. RBI rates
    rbi_seed = manual_dir / "rbi_rates_seed.csv"
    build_rates_parquet(rbi_seed, processed_dir / "rates.parquet", start, end)

    # 3. Events calendar (use underlying trading dates if available)
    underlying_path = processed_dir / "underlying_ohlcv.parquet"
    trading_calendar = None
    if underlying_path.exists():
        trading_calendar = pd_read_dates(underlying_path)

    events_seed = manual_dir / "events_calendar_seed.csv"
    build_events_calendar(
        start_date,
        end_date,
        events_seed,
        processed_dir / "events_calendar.parquet",
        trading_calendar=trading_calendar,
    )

    # 4. Bhav copy download
    bhav_dir = raw_dir / "bhav_copy"
    if not cfg.pipeline.get("skip_bhav_download", False):
        download_bhav_range(
            start_date,
            end_date,
            bhav_dir,
            symbols=symbols,
            delay_seconds=cfg.data.bhav.get("delay_seconds", 1.0),
            max_retries=cfg.data.bhav.get("max_retries", 3),
        )

    # 5. Merge options chain
    merge_bhav_to_parquet(
        bhav_dir,
        processed_dir / "options_chain.parquet",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    # 6. Validate
    if cfg.pipeline.get("validate", True):
        validate_all(processed_dir, start, end)

    log.info("Pipeline complete. Outputs in %s", processed_dir)


def pd_read_dates(path: Path):
    import pandas as pd

    df = pd.read_parquet(path)
    return pd.DatetimeIndex(pd.to_datetime(df["date"]).unique())


def run_indices_only(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        cfg = _load_default_config()
    download_all_indices(OmegaConf.to_container(cfg.data, resolve=True), _resolve_path(cfg, "raw_dir"))


def run_rates_only(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        cfg = _load_default_config()
    manual_dir = _resolve_path(cfg, "manual_dir")
    processed_dir = _resolve_path(cfg, "processed_dir")
    build_rates_parquet(
        manual_dir / "rbi_rates_seed.csv",
        processed_dir / "rates.parquet",
        cfg.data.start_date,
        cfg.data.end_date,
    )


def run_events_only(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        cfg = _load_default_config()
    manual_dir = _resolve_path(cfg, "manual_dir")
    processed_dir = _resolve_path(cfg, "processed_dir")
    start = date.fromisoformat(cfg.data.start_date)
    end = date.fromisoformat(cfg.data.end_date)
    build_events_calendar(
        start,
        end,
        manual_dir / "events_calendar_seed.csv",
        processed_dir / "events_calendar.parquet",
    )


def run_merge_only(cfg: DictConfig | None = None) -> None:
    if cfg is None:
        cfg = _load_default_config()
    raw_dir = _resolve_path(cfg, "raw_dir")
    processed_dir = _resolve_path(cfg, "processed_dir")
    merge_bhav_to_parquet(
        raw_dir / "bhav_copy",
        processed_dir / "options_chain.parquet",
        symbols=list(cfg.data.symbols),
        start_date=date.fromisoformat(cfg.data.start_date),
        end_date=date.fromisoformat(cfg.data.end_date),
    )


def _load_default_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
