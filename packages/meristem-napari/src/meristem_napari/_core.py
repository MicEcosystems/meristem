"""Napari-free helpers shared by the widgets.

Everything here is plain Python/NumPy so it can be unit-tested without a Qt/napari stack. The
widgets in ``_widgets`` import napari/magicgui and delegate the real work to these helpers and to
``meristem.core``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from meristem.core import (
    ROI,
    BackendConfig,
    ImageStack,
    SegMasks,
    filter_by_size,
    get_segmenter,
    get_tracker,
    list_segmenters,
    list_trackers,
)
from meristem.core import apply_shifts, crop_with_drift, estimate_drift
from meristem.core.pipeline import segment as core_segment
from meristem.core.pipeline import track as core_track

LayerDataTuple = Tuple[np.ndarray, dict, str]


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


# ---------------------------------------------------------------------------
# The actual work behind each widget — napari-free, so it is fully unit-testable
# ---------------------------------------------------------------------------
def segment_to_layer(
    image_data: np.ndarray,
    name: str,
    segmenter: str,
    *,
    device: str = "auto",
    min_size_frac: float = 0.0,
    crop_rect: Optional[np.ndarray] = None,
) -> LayerDataTuple:
    """Segment an image array (optionally cropped) -> a napari Labels LayerDataTuple."""
    stack = ImageStack(data=to_tyx(image_data), name=name)
    if crop_rect is not None:
        stack = stack.crop(roi_from_rectangle(crop_rect, stack.shape_yx))
    params = params_for_segmenter(segmenter, device=device)
    masks = core_segment(stack, BackendConfig(name=segmenter, params=params))
    if min_size_frac > 0:
        masks = filter_by_size(masks, min_size_frac=min_size_frac)
    return (masks.data, {"name": f"{name}_masks ({segmenter})"}, "labels")


def track_to_layer(
    mask_data: np.ndarray,
    image_data: np.ndarray,
    name: str,
    tracker: str,
    *,
    frame_start: int = 0,
    frame_stop: int = 0,
) -> LayerDataTuple:
    """Track a mask array over [start, stop) -> a napari Tracks LayerDataTuple."""
    marr, iarr = to_tyx(mask_data), to_tyx(image_data)
    stop = frame_stop or marr.shape[0]
    marr, iarr = marr[frame_start:stop], iarr[frame_start:stop]
    stack = ImageStack(data=iarr, name=name)
    seg = SegMasks(data=marr.astype("int32"), source=name)
    params = params_for_tracker(tracker, device="cpu")
    tg = core_track(stack, seg, BackendConfig(name=tracker, params=params))
    data, graph = tg.to_napari_tracks()
    meta = {"name": f"{name}_tracks ({tracker})", "graph": graph, "tail_length": 10}
    return (data, meta, "tracks")


def segment_and_track_to_layers(
    image_data: np.ndarray,
    name: str,
    segmenter: str,
    tracker: str,
    *,
    device: str = "auto",
    crop_rect: Optional[np.ndarray] = None,
    register: bool = False,
) -> List[LayerDataTuple]:
    """The whole thing in one call: (optional register) -> (optional crop) -> segment -> track.

    Returns a Labels layer and a Tracks layer. This is what the one-click launcher runs, so a
    biologist never touches YAML — pick a segmenter and tracker, press Run.
    """
    data = to_tyx(image_data)
    shifts = estimate_drift(data) if (register and data.shape[0] > 1) else None

    if crop_rect is not None:
        roi = roi_from_rectangle(crop_rect, data.shape[-2:])
        if shifts is not None:
            data = crop_with_drift(data, roi.y, roi.x, roi.height, roi.width, shifts)
        else:
            data = data[:, roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    elif shifts is not None:
        data = apply_shifts(data, shifts)

    stack = ImageStack(data=data, name=name)
    masks = core_segment(
        stack, BackendConfig(name=segmenter, params=params_for_segmenter(segmenter, device=device))
    )
    tg = core_track(
        stack, masks, BackendConfig(name=tracker, params=params_for_tracker(tracker, device="cpu"))
    )
    tdata, graph = tg.to_napari_tracks()
    return [
        (masks.data, {"name": f"{name}_masks ({segmenter})"}, "labels"),
        (tdata, {"name": f"{name}_tracks ({tracker})", "graph": graph, "tail_length": 10}, "tracks"),
    ]
