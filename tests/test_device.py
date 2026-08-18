"""Tests for device selection utilities."""

from __future__ import annotations

import torch

from nope_in.utils.device import (
    device_label,
    resolve_device,
    resolve_lightning_accelerator,
    resolve_shap_device,
    supports_pin_memory,
)


class TestResolveDevice:
    def test_cpu_override(self):
        assert resolve_device("cpu").type == "cpu"

    def test_auto_returns_valid_device(self):
        device = resolve_device("auto")
        assert device.type in {"cpu", "cuda", "mps"}

    def test_lightning_accelerator(self):
        acc = resolve_lightning_accelerator("auto")
        assert acc in {"cpu", "cuda", "mps"}

    def test_shap_falls_back_from_mps(self):
        shap_dev = resolve_shap_device(torch.device("mps"), "auto")
        assert shap_dev.type == "cpu"

    def test_shap_honours_explicit_mps(self):
        shap_dev = resolve_shap_device(torch.device("mps"), "mps")
        assert shap_dev.type in {"mps", "cpu", "cuda"}

    def test_device_label(self):
        assert "Metal" in device_label(torch.device("mps"))
        assert device_label(torch.device("cpu")) == "CPU"

    def test_pin_memory_false_without_cuda(self):
        if not torch.cuda.is_available():
            assert supports_pin_memory() is False
