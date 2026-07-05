"""magicgui widgets: the interactive segment -> inspect -> track workflow.

Two dock widgets mirror the CLI stages. **Segment** runs a chosen backend on an image layer
(optionally cropped to a rectangle you draw on a Shapes layer) and adds a Labels layer you can
inspect. **Track + measure** links a Labels layer over a chosen frame window and adds a Tracks
layer. Both delegate to ``meristem.core`` — the GUI is a thin front-end, not a reimplementation.
"""

from __future__ import annotations

import napari
from magicgui import magic_factory

from meristem.core import BackendConfig, ImageStack, SegMasks, filter_by_size
from meristem.core.pipeline import segment as core_segment
from meristem.core.pipeline import track as core_track

from ._core import (
    params_for_segmenter,
    params_for_tracker,
    roi_from_rectangle,
    segmenter_choices,
    to_tyx,
    tracker_choices,
)


@magic_factory(
    call_button="Segment",
    segmenter={"choices": segmenter_choices},
    device={"choices": ["auto", "cpu", "cuda", "mps"]},
    min_size_frac={"label": "size filter (frac of mean, 0=off)", "min": 0.0, "step": 0.01},
)
def segment_widget(
    image: "napari.layers.Image",
    segmenter: str = "cellpose-sam",
    device: str = "auto",
    min_size_frac: float = 0.0,
    crop: "napari.layers.Shapes" = None,
) -> "napari.types.LayerDataTuple":
    """Segment ``image`` (optionally cropped to the first rectangle in ``crop``) and add masks."""
    stack = ImageStack(data=to_tyx(image.data), name=image.name)
    if crop is not None and len(crop.data):
        roi = roi_from_rectangle(crop.data[0], stack.shape_yx)
        stack = stack.crop(roi)

    params = params_for_segmenter(segmenter, device=device)
    masks = core_segment(stack, BackendConfig(name=segmenter, params=params))
    if min_size_frac > 0:
        masks = filter_by_size(masks, min_size_frac=min_size_frac)

    return (masks.data, {"name": f"{image.name}_masks ({segmenter})"}, "labels")


@magic_factory(
    call_button="Track",
    tracker={"choices": tracker_choices},
    frame_stop={"label": "frame stop (0 = end)"},
)
def track_widget(
    masks: "napari.layers.Labels",
    image: "napari.layers.Image",
    tracker: str = "strack",
    frame_start: int = 0,
    frame_stop: int = 0,
) -> "napari.types.LayerDataTuple":
    """Track ``masks`` over an optional [start, stop) frame window and add a Tracks layer."""
    mask_arr = to_tyx(masks.data)
    img_arr = to_tyx(image.data)
    stop = frame_stop or mask_arr.shape[0]
    mask_arr = mask_arr[frame_start:stop]
    img_arr = img_arr[frame_start:stop]

    stack = ImageStack(data=img_arr, name=image.name)
    seg = SegMasks(data=mask_arr.astype("int32"), source=masks.name)
    params = params_for_tracker(tracker, device="cpu")
    tg = core_track(stack, seg, BackendConfig(name=tracker, params=params))

    data, graph = tg.to_napari_tracks()
    meta = {"name": f"{masks.name}_tracks ({tracker})", "graph": graph, "tail_length": 10}
    return (data, meta, "tracks")
