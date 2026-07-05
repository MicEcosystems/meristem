"""napari reader: open a channel TIFF stack as an Image layer."""

from __future__ import annotations

from typing import Callable, Optional


def napari_get_reader(path) -> Optional[Callable]:
    """npe2 reader hook: return a reader for single TIFF stacks, else None."""
    if isinstance(path, list):
        return None
    if not str(path).lower().endswith((".tif", ".tiff")):
        return None
    return _read


def _read(path):
    from meristem.core.io import read_image_stack

    stack = read_image_stack(path, name=_stem(path))
    return [(stack.data, {"name": stack.name}, "image")]


def _stem(path) -> str:
    from pathlib import Path

    return Path(str(path)).stem
