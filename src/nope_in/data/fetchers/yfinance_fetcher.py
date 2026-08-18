"""
yfinance Data Fetcher — NIFTY, BANKNIFTY, India VIX.

Downloads OHLCV for Indian market indices and writes parquet caches.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from nope_in.exceptions import DataFetchError, DataQualityError

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
VIX_COLUMNS = ["date", "vix_open", "vix_high", "vix_low", "vix_close"]


def _normalize_yfinance_symbol(symbol: str) -> str:
    if symbol.startswith("^"):
        return symbol
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def download_index_ohlcv(
    symbol: str,
    start: str,
    end: str,
    output_path: Path,
    internal_name: Optional[str] = None,
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV for a single index symbol using yfinance.
    Forward-fills holiday gaps and flags interpolated rows.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force_refresh:
        return pd.read_parquet(output_path)

    yf_symbol = _normalize_yfinance_symbol(symbol)
    name = internal_name or symbol.lstrip("^").replace("NSE", "NIFTY").replace("BANK", "BANKNIFTY")
    if symbol == "^NSEI":
        name = "NIFTY"
    elif symbol == "^NSEBANK":
        name = "BANKNIFTY"
    elif symbol == "^INDIAVIX":
        name = "INDIAVIX"

    ticker = yf.Ticker(yf_symbol)
    raw = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
    if raw.empty:
        raise DataFetchError(f"No data returned for {symbol} ({yf_symbol})")

    df = raw.reset_index()
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = name

    if name == "INDIAVIX":
        vix_df = df.rename(
            columns={
                "open": "vix_open",
                "high": "vix_high",
                "low": "vix_low",
                "close": "vix_close",
            }
        )[["date", "vix_open", "vix_high", "vix_low", "vix_close"]]
        vix_df = _forward_fill_gaps(vix_df, "date")
        vix_df.to_parquet(output_path, index=False)
        return vix_df

    df = df[OHLCV_COLUMNS]
    df = _forward_fill_gaps(df, "date")
    df.to_parquet(output_path, index=False)
    return df


def _forward_fill_gaps(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Forward-fill missing trading days; flag interpolated rows."""
    out = df.sort_values(date_col).copy()
    out["is_interpolated"] = False

    if len(out) < 2:
        return out

    full_range = pd.bdate_range(out[date_col].min(), out[date_col].max())
    reindexed = out.set_index(date_col).reindex(full_range)
    was_missing = reindexed.isna().any(axis=1)
    reindexed = reindexed.ffill()
    reindexed["is_interpolated"] = was_missing.values
    reindexed = reindexed.reset_index().rename(columns={"index": date_col})

    consecutive_nan = was_missing.astype(int).groupby((~was_missing).cumsum()).sum().max()
    if consecutive_nan and consecutive_nan > 5:
        raise DataQualityError(
            f"More than 5 consecutive missing rows detected (max={consecutive_nan})"
        )

    return reindexed


def download_all_indices(
    config: dict[str, Any],
    output_dir: Path,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download all configured indices and write raw + processed parquet files.
    """
    start = config["start_date"]
    end = config["end_date"]
    indices = config["yfinance"]["indices"]
    processed_dir = Path(config["paths"]["processed_dir"])

    raw_frames: dict[str, pd.DataFrame] = {}
    underlying_frames: list[pd.DataFrame] = []
    vix_frame: Optional[pd.DataFrame] = None

    for entry in indices:
        yf_sym = entry["symbol"]
        name = entry["name"]

        if name == "INDIAVIX":
            raw_path = output_dir / "vix" / "india_vix.parquet"
        else:
            raw_path = output_dir / "ohlcv" / f"{name.lower()}.parquet"

        df = download_index_ohlcv(
            yf_sym,
            start,
            end,
            raw_path,
            internal_name=name,
            force_refresh=force_refresh,
        )
        raw_frames[name] = df

        if name == "INDIAVIX":
            vix_frame = _build_vix_processed(df)
        else:
            underlying_frames.append(_build_underlying_processed(df, name))

    if underlying_frames:
        underlying = pd.concat(underlying_frames, ignore_index=True)
        underlying_path = processed_dir / "underlying_ohlcv.parquet"
        processed_dir.mkdir(parents=True, exist_ok=True)
        underlying.to_parquet(underlying_path, index=False)
        raw_frames["underlying_ohlcv"] = underlying

    if vix_frame is not None:
        vix_path = processed_dir / "india_vix.parquet"
        processed_dir.mkdir(parents=True, exist_ok=True)
        vix_frame.to_parquet(vix_path, index=False)
        raw_frames["india_vix"] = vix_frame

    return raw_frames


def _build_underlying_processed(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Minimal Phase 1 processing for underlying OHLCV."""
    out = df.copy()
    if "symbol" not in out.columns:
        out["symbol"] = symbol
    out = out.sort_values("date")
    out["returns_1d"] = (out["close"] / out["close"].shift(1)).apply(
        lambda x: float("nan") if pd.isna(x) or x <= 0 else math.log(x)
    )
    out["returns_5d"] = (out["close"] / out["close"].shift(5)).apply(
        lambda x: float("nan") if pd.isna(x) or x <= 0 else math.log(x)
    )
    out["returns_21d"] = (out["close"] / out["close"].shift(21)).apply(
        lambda x: float("nan") if pd.isna(x) or x <= 0 else math.log(x)
    )
    return out


def _build_vix_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal Phase 1 VIX features."""
    out = df.copy().sort_values("date")
    out["vix_change"] = out["vix_close"].diff()
    out["vix_pct_change"] = out["vix_close"].pct_change()
    out["vix_5d_ma"] = out["vix_close"].rolling(5, min_periods=1).mean()
    out["vix_21d_ma"] = out["vix_close"].rolling(21, min_periods=1).mean()
    out["vix_percentile_252d"] = out["vix_close"].rolling(252, min_periods=21).apply(
        lambda s: (s.iloc[-1] >= s).mean() if len(s) > 0 else float("nan"),
        raw=False,
    )
    out["vix_regime_label"] = pd.cut(
        out["vix_percentile_252d"],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=["Low", "Mid", "High", "Stress"],
    ).astype(str)
    out["vix_term_slope"] = out["vix_21d_ma"] - out["vix_close"]
    return out
