"""
Indian Market Events Calendar Builder.

Auto-generates NSE expiry dates and merges manual RBI/Budget/FOMC events.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# US FOMC 2018-2023 (scheduled meeting start dates)
_FOMC_DATES = [
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29",
    "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-01-31", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
]


def _last_thursday(year: int, month: int) -> date:
    """Return last Thursday of a calendar month."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:  # Thursday
        d -= timedelta(days=1)
    return d


def _thursdays_in_month(year: int, month: int) -> list[date]:
    thursdays = []
    d = date(year, month, 1)
    while d.month == month:
        if d.weekday() == 3:
            thursdays.append(d)
        d += timedelta(days=1)
    return thursdays


def _wednesdays_in_month(year: int, month: int) -> list[date]:
    weds = []
    d = date(year, month, 1)
    while d.month == month:
        if d.weekday() == 2:
            weds.append(d)
        d += timedelta(days=1)
    return weds


def generate_nse_expiry_dates(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Auto-generate NSE F&O expiry dates.
    NIFTY: weekly + monthly on Thursday; BANKNIFTY weekly on Wednesday.
    """
    rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            last_thu = _last_thursday(year, month)
            rows.append(
                {
                    "date": last_thu,
                    "event_type": "NSE_MONTHLY_EXPIRY",
                    "description": f"NSE monthly expiry {last_thu.isoformat()}",
                    "rbi_rate_change_bps": None,
                }
            )
            for thu in _thursdays_in_month(year, month):
                if thu != last_thu:
                    rows.append(
                        {
                            "date": thu,
                            "event_type": "NSE_WEEKLY_EXPIRY",
                            "description": f"NIFTY weekly expiry {thu.isoformat()}",
                            "rbi_rate_change_bps": None,
                        }
                    )
            for wed in _wednesdays_in_month(year, month):
                if wed != last_thu:
                    rows.append(
                        {
                            "date": wed,
                            "event_type": "NSE_WEEKLY_EXPIRY",
                            "description": f"BANKNIFTY weekly expiry {wed.isoformat()}",
                            "rbi_rate_change_bps": None,
                        }
                    )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(subset=["date", "event_type", "description"]).sort_values("date")


def load_manual_events(manual_csv_path: Path) -> pd.DataFrame:
    """Load manually curated events CSV."""
    df = pd.read_csv(manual_csv_path)
    required = {"date", "event_type", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    if "rbi_rate_change_bps" not in df.columns:
        df["rbi_rate_change_bps"] = None
    return df


def _fomc_events() -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp(d),
            "event_type": "US_FOMC",
            "description": f"US FOMC {d}",
            "rbi_rate_change_bps": None,
        }
        for d in _FOMC_DATES
    ]
    return pd.DataFrame(rows)


def _days_to_next_event(
    trading_dates: pd.DatetimeIndex,
    event_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Trading-day distance to next event occurrence."""
    event_dates = event_dates.sort_values()
    result = []
    for d in trading_dates:
        future = event_dates[event_dates >= d]
        if len(future) == 0:
            result.append(None)
        else:
            result.append(len(trading_dates[(trading_dates >= d) & (trading_dates <= future[0])]) - 1)
    return pd.Series(result, index=trading_dates)


def build_events_calendar(
    start_date: date,
    end_date: date,
    manual_csv: Optional[Path],
    output_path: Path,
    trading_calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Build daily-resolution events calendar with proximity features.
    """
    start_year = start_date.year
    end_year = end_date.year

    auto_expiry = generate_nse_expiry_dates(start_year, end_year)
    fomc = _fomc_events()

    frames = [auto_expiry, fomc]
    if manual_csv and manual_csv.exists():
        frames.append(load_manual_events(manual_csv))

    events = pd.concat(frames, ignore_index=True)
    events = events.drop_duplicates(subset=["date", "event_type"]).sort_values("date")

    if trading_calendar is None:
        trading_calendar = pd.bdate_range(start_date, end_date)
    else:
        trading_calendar = pd.DatetimeIndex(trading_calendar)

    daily = pd.DataFrame({"date": trading_calendar})
    daily["date"] = pd.to_datetime(daily["date"])

    event_types = {
        "RBI_MPC": "days_to_next_rbi_mpc",
        "UNION_BUDGET": "days_to_next_budget",
        "US_FOMC": "days_to_next_us_fomc",
        "NSE_MONTHLY_EXPIRY": "days_to_next_nse_monthly_expiry",
    }

    for etype, col in event_types.items():
        ev_dates = pd.DatetimeIndex(events.loc[events["event_type"] == etype, "date"].unique())
        daily[col] = _days_to_next_event(trading_calendar, ev_dates).values

    daily["days_to_current_weekly_expiry"] = _days_to_next_event(
        trading_calendar,
        pd.DatetimeIndex(events.loc[events["event_type"] == "NSE_WEEKLY_EXPIRY", "date"].unique()),
    ).values

    daily["is_rbi_week"] = daily["days_to_next_rbi_mpc"].le(5)
    daily["is_budget_week"] = daily["days_to_next_budget"].le(10)
    daily["is_expiry_week"] = daily["days_to_current_weekly_expiry"].le(5)

    daily["has_rbi_mpc"] = daily["date"].isin(
        events.loc[events["event_type"] == "RBI_MPC", "date"]
    )
    daily["has_budget"] = daily["date"].isin(
        events.loc[events["event_type"] == "UNION_BUDGET", "date"]
    )
    daily["has_nse_monthly_expiry"] = daily["date"].isin(
        events.loc[events["event_type"] == "NSE_MONTHLY_EXPIRY", "date"]
    )
    daily["has_nse_weekly_expiry"] = daily["date"].isin(
        events.loc[events["event_type"] == "NSE_WEEKLY_EXPIRY", "date"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path, index=False)
    log.info("Wrote events_calendar.parquet: %d rows", len(daily))
    return daily
