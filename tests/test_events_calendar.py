"""Tests for events calendar builder."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from nope_in.data.fetchers.events_calendar_fetcher import (
    build_events_calendar,
    generate_nse_expiry_dates,
    load_manual_events,
)

SEED = Path(__file__).resolve().parents[1] / "data" / "manual" / "events_calendar_seed.csv"


def test_generate_nse_expiry_dates():
    df = generate_nse_expiry_dates(2023, 2023)
    monthly = df[df["event_type"] == "NSE_MONTHLY_EXPIRY"]
    assert len(monthly) == 12
    # Jan 2023 last Thursday
    jan_expiry = monthly[monthly["date"].dt.month == 1]["date"].iloc[0]
    assert jan_expiry.dayofweek == 3  # Thursday


def test_load_manual_events():
    df = load_manual_events(SEED)
    assert "RBI_MPC" in df["event_type"].values


def test_build_events_calendar(tmp_path):
    out = tmp_path / "events_calendar.parquet"
    df = build_events_calendar(
        date(2023, 1, 1),
        date(2023, 1, 31),
        SEED,
        out,
    )
    assert "days_to_next_rbi_mpc" in df.columns
    assert "is_rbi_week" in df.columns
    assert out.exists()
