"""Napari-free helpers shared by the widgets.

Everything here is plain Python/NumPy so it can be unit-tested without a Qt/napari stack. The
widgets in ``_widgets`` import napari/magicgui and delegate the real work to these helpers and to
``meristem.core``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from meristem.core import ROI, get_segmenter, get_tracker, list_segmenters, list_trackers


def segmenter_choices(*_args) -> List[str]:
    """Segmenter names for a dropdown (signature tolerant of magicgui's widget arg)."""
    return list_segmenters()


def tracker_choices(*_args) -> List[str]:
    return list_trackers()


def params_for_segmenter(name: str, **kwargs) -> Dict:
    """Keep only kwargs that the chosen segmenter's Params model actually declares."""
    return _filter_params(get_segmenter(name).Params, kwargs)


def params_for_tracker(name: str, **kwargs) -> Dict:
    return _filter_params(get_tracker(name).Params, kwargs)


def _filter_params(params_model, kwargs: Dict) -> Dict:
    fields = getattr(params_model, "model_fields", {})
    return {k: v for k, v in kwargs.items() if k in fields and v is not None}


def to_tyx(array: np.ndarray) -> np.ndarray:
    """Coerce a napari image layer's data to a single-channel (T, Y, X) stack."""
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim == 3:
        return array
    raise ValueError(f"expected 2D or 3D image data, got shape {array.shape}")


def roi_from_rectangle(rectangle: np.ndarray, image_shape_yx) -> ROI:
    """Convert a napari rectangle Shapes item (its corner coords) to an :class:`ROI`.

    A napari rectangle is given as its corner vertices in (row, col) = (y, x) coordinates. We take
    the axis-aligned bounding box and clamp it to the image. This is what turns an interactively
    drawn box into the pipeline's manual crop.
    """
    pts = np.asarray(rectangle, dtype=float)
    ys, xs = pts[:, -2], pts[:, -1]  # last two axes are (y, x) even for t/z-augmented shapes
    y0, x0 = int(np.floor(ys.min())), int(np.floor(xs.min()))
    y1, x1 = int(np.ceil(ys.max())), int(np.ceil(xs.max()))
    max_y, max_x = image_shape_yx
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(max_y, y1), min(max_x, x1)
    return ROI(y=y0, x=x0, height=max(1, y1 - y0), width=max(1, x1 - x0))
