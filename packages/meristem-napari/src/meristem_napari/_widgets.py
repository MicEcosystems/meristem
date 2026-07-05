"""magicgui widgets: the interactive segment -> inspect -> track workflow.

Two dock widgets mirror the CLI stages. **Segment** runs a chosen backend on an image layer
(optionally cropped to a rectangle you draw on a Shapes layer) and adds a Labels layer you can
inspect. **Track + measure** links a Labels layer over a chosen frame window and adds a Tracks
layer. The widgets are thin adapters: they pull arrays off the layers and hand them to the
napari-free functions in ``_core`` (which delegate to ``meristem.core``).
"""

from __future__ import annotations

import napari
from magicgui import magic_factory

from ._core import (
    segment_to_layer,
    segmenter_choices,
    track_to_layer,
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
    crop_rect = crop.data[0] if (crop is not None and len(crop.data)) else None
    return segment_to_layer(
        image.data,
        image.name,
        segmenter,
        device=device,
        min_size_frac=min_size_frac,
        crop_rect=crop_rect,
    )


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
    return track_to_layer(
        masks.data,
        image.data,
        masks.name,
        tracker,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
