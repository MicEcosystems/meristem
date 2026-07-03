"""Centralized compute-device selection.

MiDAP duplicated hardware detection inside each segmentation class. Here it lives once. Real
backends call :func:`select_device` in their ``load()`` so device policy is consistent and
overridable from config. This module deliberately does **not** import torch — it returns a plain
string and only probes torch if it happens to be installed, keeping the core dependency-light.
"""

from __future__ import annotations

from typing import Literal

Device = Literal["cpu", "cuda", "mps", "auto"]


def select_device(preference: Device = "auto") -> str:
    """Resolve a device preference to a concrete device string.

    ``"auto"`` picks CUDA if available, else Apple-Silicon MPS, else CPU. An explicit preference
    is honored but falls back to CPU with a warning if the requested backend isn't usable, so a
    config written for a GPU box still runs (slowly) on a laptop.
    """
    if preference != "auto":
        if preference == "cpu" or _backend_available(preference):
            return preference
        import warnings

        warnings.warn(f"requested device {preference!r} unavailable; falling back to cpu")
        return "cpu"

    for candidate in ("cuda", "mps"):
        if _backend_available(candidate):
            return candidate
    return "cpu"


def _backend_available(device: str) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    if device == "cuda":
        return bool(torch.cuda.is_available())
    if device == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return False
