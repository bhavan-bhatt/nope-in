"""PyTorch device selection — CUDA, Apple MPS (M1/M2/M3), or CPU."""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """
    Pick the best available PyTorch device.

    On Mac M2/M3, ``auto`` selects ``mps`` (Metal GPU) when available.
    Preference can be ``auto``, ``mps``, ``cuda``, ``gpu``, or ``cpu``.
    """
    pref = preference.lower().strip()

    if pref == "cpu":
        return torch.device("cpu")

    if pref in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "mps":
        if _mps_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_lightning_accelerator(preference: str = "auto") -> str:
    """Map device preference to a PyTorch Lightning accelerator string."""
    device = resolve_device(preference)
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"


def resolve_shap_device(inference_device: torch.device, preference: str = "auto") -> torch.device:
    """
    SHAP DeepExplainer does not reliably support Apple MPS.

    By default, keep SHAP on CPU when inference uses MPS; honour explicit overrides.
    """
    pref = preference.lower().strip()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in {"mps", "cuda", "gpu"}:
        return resolve_device(pref)
    if inference_device.type == "mps":
        return torch.device("cpu")
    return inference_device


def supports_pin_memory(device: torch.device | str | None = None) -> bool:
    """pin_memory only benefits CUDA transfers."""
    if device is None:
        return torch.cuda.is_available()
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return dev.type == "cuda"


def device_label(device: torch.device | str) -> str:
    """Human-readable device name for progress output."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    labels = {
        "cuda": "CUDA GPU",
        "mps": "Apple Metal (MPS)",
        "cpu": "CPU",
    }
    return labels.get(dev.type, dev.type.upper())


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
