"""Shared plumbing for the Cellpose-family backends.

The two backends in this package (Cellpose-SAM and Omnipose) differ only in *which* model they
build and *how* they call ``eval``. Everything around that — resolving the device, giving a
friendly error when the heavy library isn't installed, and assembling per-frame 2D label images
into a ``(T, Y, X)`` :class:`~meristem.core.contracts.SegMasks` — lives here so neither backend
repeats it.

Crucially, nothing in this module (or the backend modules) imports cellpose/omnipose at import
time. The registry loads these classes to list them in the UI; the multi-hundred-MB ML stack is
only touched inside ``load()``, when the user actually selects the backend.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from meristem.core.contracts import ImageStack, SegMasks
from meristem.core.device import select_device


def import_or_hint(module: str, *, backend: str, extra: str):
    """Import a heavy dependency by name, or raise a clear install hint.

    Keeping this out of module scope is what lets the backend be *discoverable* without the ML
    stack installed — a laptop can list ``cellpose-sam`` in the napari dropdown even if the weights
    aren't there yet, and only hit this when it tries to run.
    """
    try:
        return __import__(module, fromlist=["_"])
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"the '{backend}' segmentation backend requires the '{module}' package, which is not "
            f"installed. Install it with:  pip install 'meristem-models-cellpose[{extra}]'"
        ) from exc


def resolve_gpu(device_preference: str) -> tuple[str, bool]:
    """Map a Meristem device preference to a concrete device and cellpose's ``gpu`` flag.

    Cellpose/Omnipose take a boolean ``gpu`` (and pick CUDA/MPS internally). We resolve the
    preference once, centrally, so device policy is identical to every other backend.
    """
    device = select_device(device_preference)  # "cpu" | "cuda" | "mps"
    return device, device != "cpu"


def segment_per_frame(
    stack: ImageStack, per_frame: Callable[[np.ndarray], np.ndarray], *, source: str
) -> SegMasks:
    """Apply a 2D ``per_frame`` segmentation callable to each frame and stack the results.

    ``per_frame`` receives a single ``(Y, X)`` image and must return a ``(Y, X)`` integer label
    image (0 = background). This isolates the version-sensitive ``model.eval`` call in the backend
    while guaranteeing the output satisfies the :class:`SegMasks` contract.
    """
    frames = []
    for t in range(stack.n_frames):
        labels = np.asarray(per_frame(stack.data[t]), dtype=np.int32)
        if labels.shape != stack.shape_yx:
            raise ValueError(
                f"{source}: frame {t} produced labels of shape {labels.shape}, "
                f"expected {stack.shape_yx}"
            )
        frames.append(labels)
    return SegMasks(data=np.stack(frames, axis=0), source=source)
