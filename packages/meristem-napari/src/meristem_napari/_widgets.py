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
from magicgui.widgets import CheckBox, ComboBox, Container, PushButton, create_widget

from ._core import (
    segment_and_track_to_layers,
    segment_to_layer,
    segmenter_choices,
    track_to_layer,
    tracker_choices,
)

CROP_LAYER = "crop"  # the Shapes layer the crop rectangle lives on


def run_widget():
    """The guided, no-YAML panel: pick a segmenter + tracker, optionally add a crop box, press Run.

    Built as a Container (not a simple form) so it can carry an **Add crop box** button that creates
    a rectangle Shapes layer in draw mode for you — no manual layer juggling. The crop is read from
    that ``crop`` layer automatically on Run.
    """
    image = create_widget(annotation="napari.layers.Image", label="Image")
    segmenter = ComboBox(choices=segmenter_choices, label="Segmentation model", value="cellpose-sam")
    tracker = ComboBox(choices=tracker_choices, label="Tracker", value="strack")
    device = ComboBox(choices=["auto", "cpu", "cuda", "mps"], label="Device", value="auto")
    register = CheckBox(value=True, text="correct drift")
    add_crop = PushButton(text="✏  Add crop box")
    run = PushButton(text="Run  ▶")
    panel = Container(widgets=[image, segmenter, tracker, device, register, add_crop, run])

    def _viewer():
        return napari.current_viewer()

    def _on_add_crop():
        v = _viewer()
        if v is None:
            return
        layer = v.layers[CROP_LAYER] if CROP_LAYER in v.layers else v.add_shapes(
            name=CROP_LAYER, edge_color="yellow", edge_width=3, face_color="transparent"
        )
        v.layers.selection.active = layer
        layer.mode = "add_rectangle"  # user just drags a box; no mode-hunting

    def _on_run():
        v = _viewer()
        img = image.value
        if img is None:
            return
        crop_rect = None
        if v is not None and CROP_LAYER in v.layers and len(v.layers[CROP_LAYER].data):
            crop_rect = v.layers[CROP_LAYER].data[0]  # first rectangle drawn
        layers = segment_and_track_to_layers(
            img.data, img.name, segmenter.value, tracker.value,
            device=device.value, crop_rect=crop_rect, register=register.value,
        )
        if v is not None:
            for data, meta, ltype in layers:
                getattr(v, f"add_{ltype}")(data, **meta)

    add_crop.clicked.connect(_on_add_crop)
    run.clicked.connect(_on_run)
    return panel


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
