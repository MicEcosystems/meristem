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
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSpinBox,
    PushButton,
    create_widget,
)

from ._core import (
    segment_and_track_to_layers,
    segment_to_layer,
    segmenter_choices,
    track_to_layer,
    tracker_choices,
)

CROP_LAYER = "crop"  # the Shapes layer the crop rectangle lives on


def _add_crop_box() -> None:
    """Create (or reuse) a rectangle Shapes layer in draw mode — the one-click crop."""
    v = napari.current_viewer()
    if v is None:
        return
    layer = v.layers[CROP_LAYER] if CROP_LAYER in v.layers else v.add_shapes(
        name=CROP_LAYER, edge_color="yellow", edge_width=3, face_color="transparent"
    )
    v.layers.selection.active = layer
    layer.mode = "add_rectangle"  # user just drags a box; no mode-hunting


def _read_crop_rect():
    """Return the first rectangle drawn on the crop layer, or None."""
    v = napari.current_viewer()
    if v is not None and CROP_LAYER in v.layers and len(v.layers[CROP_LAYER].data):
        return v.layers[CROP_LAYER].data[0]
    return None


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

    def _on_run():
        v = napari.current_viewer()
        img = image.value
        if img is None:
            return
        layers = segment_and_track_to_layers(
            img.data, img.name, segmenter.value, tracker.value,
            device=device.value, crop_rect=_read_crop_rect(), register=register.value,
        )
        if v is not None:
            for data, meta, ltype in layers:
                getattr(v, f"add_{ltype}")(data, **meta)

    add_crop.clicked.connect(_add_crop_box)
    run.clicked.connect(_on_run)
    return panel


def segment_widget():
    """Stage 1 (Container): pick a model, add a crop box, Segment — masks appear as a Labels layer.

    Same one-click crop as the Run panel: **Add crop box** creates a rectangle to drag on the PH
    image; no manual Shapes-layer setup.
    """
    image = create_widget(annotation="napari.layers.Image", label="Image")
    segmenter = ComboBox(choices=segmenter_choices, label="Segmentation model", value="cellpose-sam")
    device = ComboBox(choices=["auto", "cpu", "cuda", "mps"], label="Device", value="auto")
    min_size = FloatSpinBox(value=0.0, min=0.0, step=0.01, label="size filter (frac of mean, 0=off)")
    add_crop = PushButton(text="✏  Add crop box")
    segment = PushButton(text="Segment")
    panel = Container(widgets=[image, segmenter, device, min_size, add_crop, segment])

    def _on_segment():
        v = napari.current_viewer()
        img = image.value
        if img is None:
            return
        data, meta, ltype = segment_to_layer(
            img.data, img.name, segmenter.value,
            device=device.value, min_size_frac=min_size.value, crop_rect=_read_crop_rect(),
        )
        if v is not None:
            getattr(v, f"add_{ltype}")(data, **meta)

    add_crop.clicked.connect(_add_crop_box)
    segment.clicked.connect(_on_segment)
    return panel


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
