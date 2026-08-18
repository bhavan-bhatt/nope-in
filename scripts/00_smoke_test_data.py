#!/usr/bin/env python3
"""Run 1-month smoke test data pipeline (Tier 0: January 2023)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nope_in.data.pipeline import run_pipeline


def main() -> None:
    config_dir = str(ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["data=tier0"])
    print(OmegaConf.to_yaml(cfg))
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
