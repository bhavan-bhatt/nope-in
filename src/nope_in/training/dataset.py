"""
PyTorch datasets and DataModule for NOPE-IN feature parquets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from nope_in.utils.device import supports_pin_memory


def load_feature_metadata(metadata_path: Path | str) -> dict:
    path = Path(metadata_path)
    return json.loads(path.read_text())


def get_regime_feature_indices(feature_columns: list[str]) -> list[int]:
    indices = []
    for col in ("regime_p0", "regime_p1", "regime_p2", "regime_p3"):
        if col in feature_columns:
            indices.append(feature_columns.index(col))
    return indices


def _moneyness_bucket(row: pd.Series) -> str:
    for col, label in (
        ("is_DITM", "DITM"),
        ("is_ITM", "ITM"),
        ("is_ATM", "ATM"),
        ("is_OTM", "OTM"),
        ("is_DOTM", "DOTM"),
    ):
        if col in row.index and bool(row[col]):
            return label
    return "OTHER"


class NOPEFeatureDataset(Dataset):
    """Loads scaled feature parquet rows for NOPE-IN training."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_col: str = "bsm_error",
    ):
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in dataset: {missing[:5]}")

        self.feature_columns = feature_columns
        self.X = torch.tensor(df[feature_columns].to_numpy(dtype=np.float32))
        self.bsm_error = torch.tensor(df[target_col].to_numpy(dtype=np.float32))
        self.bsm_price = torch.tensor(df["bsm_price"].to_numpy(dtype=np.float32))
        self.settle_price = torch.tensor(df["settle_price"].to_numpy(dtype=np.float32))
        self.moneyness = torch.tensor(df["moneyness"].to_numpy(dtype=np.float32))
        self.vega = torch.tensor(df["vega"].to_numpy(dtype=np.float32))
        self.implied_vol = torch.tensor(df["implied_vol"].to_numpy(dtype=np.float32))

        if "spot" in df.columns and "strike" in df.columns and "T" in df.columns:
            self.spot = torch.tensor(df["spot"].to_numpy(dtype=np.float32))
            self.strike = torch.tensor(df["strike"].to_numpy(dtype=np.float32))
            self.T = torch.tensor(df["T"].to_numpy(dtype=np.float32))
            self.rate = torch.tensor(df["rate_interpolated_dte"].to_numpy(dtype=np.float32))
            self.option_type = df["option_type"].astype(str).tolist()
        else:
            self.spot = None
            self.strike = None
            self.T = None
            self.rate = None
            self.option_type = None

        regime_cols = [c for c in feature_columns if c.startswith("regime_p")]
        if regime_cols:
            regime_probs = df[regime_cols].to_numpy(dtype=np.float32)
            self.regime_id = torch.tensor(regime_probs.argmax(axis=1), dtype=torch.long)
        else:
            self.regime_id = torch.zeros(len(df), dtype=torch.long)

        self.is_atm = torch.tensor(
            df["is_ATM"].to_numpy(dtype=np.float32) if "is_ATM" in df.columns else np.zeros(len(df)),
        )
        self.moneyness_bucket = [_moneyness_bucket(row) for _, row in df.iterrows()]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item: dict[str, torch.Tensor | str] = {
            "features": self.X[idx],
            "bsm_error": self.bsm_error[idx],
            "bsm_price": self.bsm_price[idx],
            "settle_price": self.settle_price[idx],
            "moneyness": self.moneyness[idx],
            "vega": self.vega[idx],
            "implied_vol": self.implied_vol[idx],
            "regime_id": self.regime_id[idx],
            "is_atm": self.is_atm[idx],
            "moneyness_bucket": self.moneyness_bucket[idx],
        }
        if self.spot is not None:
            item["spot"] = self.spot[idx]
            item["strike"] = self.strike[idx]
            item["T"] = self.T[idx]
            item["rate"] = self.rate[idx]
            item["option_type"] = self.option_type[idx]  # type: ignore[assignment]
        return item


def _collate_batch(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    out: dict[str, torch.Tensor | list[str]] = {}
    tensor_keys = [k for k in batch[0].keys() if k not in {"moneyness_bucket", "option_type"}]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch])  # type: ignore[arg-type]
    out["moneyness_bucket"] = [item["moneyness_bucket"] for item in batch]
    if "option_type" in batch[0]:
        out["option_type"] = [item["option_type"] for item in batch]  # type: ignore[assignment]
    return out


class NOPEDataModule(pl.LightningDataModule):
    """Loads train/val/test parquets from ``data/features/``."""

    def __init__(
        self,
        features_dir: Path | str,
        metadata_path: Path | str,
        batch_size: int = 2048,
        num_workers: int = 0,
        cal_fraction: float = 0.5,
    ):
        super().__init__()
        self.features_dir = Path(features_dir)
        self.metadata_path = Path(metadata_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cal_fraction = cal_fraction

        meta = load_feature_metadata(self.metadata_path)
        self.feature_columns = meta["feature_columns"]
        self.target_col = meta["target_columns"]["primary"]
        self.regime_indices = get_regime_feature_indices(self.feature_columns)

        self.train_dataset: NOPEFeatureDataset | None = None
        self.val_dataset: NOPEFeatureDataset | None = None
        self.cal_dataset: NOPEFeatureDataset | None = None
        self.test_dataset: NOPEFeatureDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        train_df = pd.read_parquet(self.features_dir / "train.parquet")
        val_df = pd.read_parquet(self.features_dir / "val.parquet")
        test_df = pd.read_parquet(self.features_dir / "test.parquet")

        self.train_dataset = NOPEFeatureDataset(train_df, self.feature_columns, self.target_col)

        if self.cal_fraction > 0:
            split_idx = int(len(val_df) * (1.0 - self.cal_fraction))
            val_fit_df = val_df.iloc[:split_idx].reset_index(drop=True)
            cal_df = val_df.iloc[split_idx:].reset_index(drop=True)
            self.val_dataset = NOPEFeatureDataset(val_fit_df, self.feature_columns, self.target_col)
            self.cal_dataset = NOPEFeatureDataset(cal_df, self.feature_columns, self.target_col)
        else:
            self.val_dataset = NOPEFeatureDataset(val_df, self.feature_columns, self.target_col)
            self.cal_dataset = None

        self.test_dataset = NOPEFeatureDataset(test_df, self.feature_columns, self.target_col)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
            pin_memory=supports_pin_memory(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
            pin_memory=supports_pin_memory(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
            pin_memory=supports_pin_memory(),
        )

    def cal_dataloader(self) -> DataLoader | None:
        if self.cal_dataset is None:
            return None
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
            pin_memory=supports_pin_memory(),
        )
