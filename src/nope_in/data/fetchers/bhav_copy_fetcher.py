"""
NSE F&O Bhav Copy Fetcher — index options (NIFTY, BANKNIFTY).

Downloads daily F&O Bhav Copy from NSE archives, parses OPTIDX (legacy) and
IDO (UDIFF 2024+) rows, and merges to options_chain.parquet.
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

OPTIDX = "OPTIDX"
IDO = "IDO"

NSE_BHAV_URL = "https://www.nseindia.com/archives/fo/bhav/fo{date_str}bhav.csv.zip"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "expiry_date",
    "strike",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "settle_price",
    "contracts",
    "val_in_lakh",
    "open_interest",
    "chg_in_oi",
    "underlying_value",
]

SCHEMA = {
    "SYMBOL": "string",
    "SERIES": "string",
    "EXPIRY_DT": "string",
    "STRIKE_PR": "float32",
    "OPTION_TYP": "string",
    "OPEN": "float32",
    "HIGH": "float32",
    "LOW": "float32",
    "CLOSE": "float32",
    "SETTLE_PR": "float32",
    "CONTRACTS": "int32",
    "VAL_INLAKH": "float32",
    "OPEN_INT": "int32",
    "CHG_IN_OI": "int32",
    "TIMESTAMP": "string",
}


def get_nse_trading_dates(start: date, end: date) -> list[date]:
    """Return weekdays between start and end (holidays handled via 404 on download)."""
    dates: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def build_bhav_url(trading_date: date) -> str:
    """Construct NSE Bhav Copy URL: fo05JAN2023bhav.csv.zip."""
    date_str = trading_date.strftime("%d%b%Y").upper()
    return NSE_BHAV_URL.format(date_str=date_str)


def _fo_bhavcopy_urls(trade_date: pd.Timestamp) -> list[str]:
    """Candidate NSE URLs (format changed over time)."""
    dt = pd.Timestamp(trade_date)
    yyyymmdd = dt.strftime("%Y%m%d")
    mon = dt.strftime("%b").upper()
    dd = f"{dt.day:02d}"
    yy = f"{dt.year % 100:02d}"

    return [
        f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip",
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{dt.year}/{mon}/fo{dd}{mon}{dt.year}bhav.csv.zip",
        f"https://www.nseindia.com/content/historical/DERIVATIVES/{dt.year}/{mon}/fo{dd}{mon}{dt.year}bhav.csv.zip",
        build_bhav_url(dt.date()),
        f"https://www.nseindia.com/archives/fo/bhav/fo{dd}{dt.month:02d}{yy}.zip",
    ]


def _nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=30)
    return session


def _csv_path(output_dir: Path, trading_date: date) -> Path:
    date_str = trading_date.strftime("%d%b%Y").upper()
    return (
        output_dir
        / str(trading_date.year)
        / f"{trading_date.month:02d}"
        / f"fo{date_str}bhav.csv"
    )


def _normalize_options_frame(df: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    """Parse legacy OPTIDX or UDIFF IDO rows into standard schema."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    upper_cols = {c.upper().strip(): c for c in df.columns}

    if "FININSTRMTP" in upper_cols or "FinInstrmTp" in df.columns:
        out = df.copy()
        out = out[out["FinInstrmTp"].astype(str).str.upper() == IDO]
        underlying_col = None
        for candidate in ("UndrlygPric", "UndrlygVal", "UndrlygPr"):
            if candidate in out.columns:
                underlying_col = candidate
                break
        out = out.assign(
            date=pd.Timestamp(trade_date).normalize(),
            symbol=out["TckrSymb"].astype(str).str.strip().str.upper(),
            expiry_date=pd.to_datetime(out["XpryDt"], errors="coerce"),
            strike=pd.to_numeric(out["StrkPric"], errors="coerce"),
            option_type=out["OptnTp"].astype(str).str.strip().str.upper(),
            open=pd.to_numeric(out["OpnPric"], errors="coerce"),
            high=pd.to_numeric(out["HghPric"], errors="coerce"),
            low=pd.to_numeric(out["LwPric"], errors="coerce"),
            close=pd.to_numeric(out["ClsPric"], errors="coerce"),
            settle_price=pd.to_numeric(out["SttlmPric"], errors="coerce"),
            contracts=pd.to_numeric(out["TtlTradgVol"], errors="coerce").fillna(0).astype("int32"),
            val_in_lakh=pd.to_numeric(out["TtlTrfVal"], errors="coerce") / 100_000.0,
            open_interest=pd.to_numeric(out["OpnIntrst"], errors="coerce").fillna(0).astype("int32"),
            chg_in_oi=pd.to_numeric(out.get("ChngInOpnIntrst", 0), errors="coerce")
            .fillna(0)
            .astype("int32"),
            underlying_value=pd.to_numeric(out[underlying_col], errors="coerce")
            if underlying_col
            else pd.NA,
        )
        return out[OUTPUT_COLUMNS]

    col_map = {c.upper().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    mapping = [
        ("SYMBOL", "symbol"),
        ("INSTRUMENT", "instrument"),
        ("EXPIRY_DT", "expiry_date"),
        ("STRIKE_PR", "strike"),
        ("OPTION_TYP", "option_type"),
        ("OPEN", "open"),
        ("HIGH", "high"),
        ("LOW", "low"),
        ("CLOSE", "close"),
        ("SETTLE_PR", "settle_price"),
        ("CONTRACTS", "contracts"),
        ("VAL_INLAKH", "val_in_lakh"),
        ("OPEN_INT", "open_interest"),
        ("CHG_IN_OI", "chg_in_oi"),
    ]
    for src, dst in mapping:
        if src in col_map:
            rename[col_map[src]] = dst

    out = df.rename(columns=rename)
    if "instrument" in out.columns:
        out = out[out["instrument"].astype(str).str.upper() == OPTIDX]

    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["expiry_date"] = pd.to_datetime(out["expiry_date"], errors="coerce", format="mixed")
    out["option_type"] = out["option_type"].astype(str).str.strip().str.upper()
    out["date"] = pd.Timestamp(trade_date).normalize()

    for col in ["open", "high", "low", "close", "settle_price", "val_in_lakh", "strike"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["contracts"] = pd.to_numeric(out["contracts"], errors="coerce").fillna(0).astype("int32")
    out["open_interest"] = (
        pd.to_numeric(out["open_interest"], errors="coerce").fillna(0).astype("int32")
    )
    out["chg_in_oi"] = pd.to_numeric(out["chg_in_oi"], errors="coerce").fillna(0).astype("int32")

    underlying_col = None
    for candidate in ("UNDERLYING", "UNDERLYING_VALUE", "UNDRLYNG"):
        if candidate in col_map:
            underlying_col = col_map[candidate]
            break
    if underlying_col and underlying_col in out.columns:
        out["underlying_value"] = pd.to_numeric(out[underlying_col], errors="coerce")
    else:
        out["underlying_value"] = pd.NA

    return out[OUTPUT_COLUMNS]


def parse_bhavcopy_bytes(content: bytes, trade_date: pd.Timestamp) -> pd.DataFrame:
    """Parse a single-day F&O bhavcopy CSV (plain or inside a zip)."""
    if not content:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return pd.DataFrame(columns=OUTPUT_COLUMNS)
            content = zf.read(csv_names[0])

    text = content.decode("utf-8", errors="replace")
    raw = pd.read_csv(io.StringIO(text))
    return _normalize_options_frame(raw, trade_date)


def validate_bhav_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a raw Bhav Copy DataFrame.
    Drops rows failing soft constraints; raises on missing critical columns.
    """
    required = {"date", "symbol", "expiry_date", "strike", "option_type", "settle_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Critical columns missing: {sorted(missing)}")

    out = df.copy()
    initial = len(out)

    out = out[out["option_type"].isin(["CE", "PE"])]
    out = out[out["strike"] > 0]
    out = out[out["settle_price"] > 0]
    out = out.dropna(subset=["symbol", "expiry_date", "settle_price"])

    dropped = initial - len(out)
    if dropped:
        log.warning("Dropped %d rows failing bhav schema constraints", dropped)

    return out.reset_index(drop=True)


def download_single_bhav(
    trading_date: date,
    output_dir: Path,
    symbols: Optional[list[str]] = None,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
) -> Optional[Path]:
    """
    Download one day's F&O Bhav Copy, unzip, save CSV.
    Returns path on success, None on holiday (404).
    """
    csv_path = _csv_path(output_dir, trading_date)
    if csv_path.exists():
        return csv_path

    trade_ts = pd.Timestamp(trading_date)
    sess = session or _nse_session()
    content: Optional[bytes] = None

    for url in _fo_bhavcopy_urls(trade_ts):
        for attempt in range(max_retries):
            try:
                resp = sess.get(url, timeout=60)
                if resp.status_code == 200 and len(resp.content) >= 2 and resp.content[:2] == b"PK":
                    content = resp.content
                    break
                if resp.status_code == 404:
                    break
            except requests.RequestException as exc:
                log.debug("Request failed %s attempt %d: %s", url, attempt + 1, exc)
            time.sleep(0.5 * (attempt + 1))
        if content:
            break

    if not content:
        return None

    df = parse_bhavcopy_bytes(content, trade_ts)
    if symbols is not None:
        sym_set = {s.upper() for s in symbols}
        df = df[df["symbol"].isin(sym_set)]

    if df.empty:
        return None

    df = validate_bhav_schema(df)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def _load_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"downloaded": [], "missing": []}


def _save_manifest(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def download_bhav_range(
    start_date: date,
    end_date: date,
    output_dir: Path,
    symbols: Optional[list[str]] = None,
    delay_seconds: float = 1.0,
    max_retries: int = 3,
) -> dict[date, Path]:
    """
    Bulk download Bhav Copy for a date range. Resumable via existing CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trading_dates = get_nse_trading_dates(start_date, end_date)
    manifest = _load_manifest(output_dir)
    downloaded: dict[date, Path] = {}
    session = _nse_session()

    for trading_date in tqdm(trading_dates, desc="Bhav copy"):
        result = download_single_bhav(
            trading_date,
            output_dir,
            symbols=symbols,
            session=session,
            max_retries=max_retries,
        )
        date_str = trading_date.isoformat()
        if result is not None:
            downloaded[trading_date] = result
            if date_str not in manifest["downloaded"]:
                manifest["downloaded"].append(date_str)
            if date_str in manifest["missing"]:
                manifest["missing"].remove(date_str)
        else:
            if date_str not in manifest["missing"]:
                manifest["missing"].append(date_str)
        time.sleep(delay_seconds)

    _save_manifest(output_dir, manifest)
    log.info(
        "Bhav download complete: %d days saved, %d unavailable",
        len(downloaded),
        len(trading_dates) - len(downloaded),
    )
    return downloaded


def _parse_csv_trade_date(csv_path: Path) -> Optional[date]:
    """Extract trade date from bhav CSV path or filename."""
    stem = csv_path.stem
    if stem.startswith("fo") and stem.endswith("bhav"):
        date_part = stem[2:-4]
        try:
            return pd.to_datetime(date_part, format="%d%b%Y").date()
        except ValueError:
            pass
    parts = csv_path.parts
    if len(parts) >= 3:
        try:
            return date(int(parts[-3]), int(parts[-2]), 1)
        except ValueError:
            pass
    return None


def merge_bhav_to_parquet(
    raw_dir: Path,
    output_path: Path,
    symbols: list[str] | None = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Read daily CSVs, filter, validate, concatenate, write parquet."""
    if symbols is None:
        symbols = ["NIFTY", "BANKNIFTY"]
    sym_set = {s.upper() for s in symbols}

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(raw_dir.rglob("fo*bhav.csv")):
        day = _parse_csv_trade_date(csv_path)
        if day is None:
            continue

        if start_date and day < start_date:
            continue
        if end_date and day > end_date:
            continue

        day_df = pd.read_csv(csv_path, parse_dates=["date", "expiry_date"])
        day_df = day_df[day_df["symbol"].isin(sym_set)]
        if not day_df.empty:
            frames.append(day_df)

    if not frames:
        raise FileNotFoundError(f"No bhav CSV files found under {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = validate_bhav_schema(combined)
    combined = combined.sort_values(["date", "symbol", "expiry_date", "strike", "option_type"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False, engine="pyarrow", row_group_size=500_000)
    log.info("Wrote %d rows to %s", len(combined), output_path)
    return combined
