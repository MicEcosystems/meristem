"""Manifest validity and napari-free helper logic (GUI widget tests skip without napari)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from meristem.core import list_segmenters, list_trackers
from meristem_napari._core import (
    params_for_segmenter,
    params_for_tracker,
    roi_from_rectangle,
    segment_and_track_to_layers,
    segment_to_layer,
    segmenter_choices,
    to_tyx,
    track_to_layer,
    tracker_choices,
)

MANIFEST = Path(__file__).resolve().parents[1] / "src" / "meristem_napari" / "napari.yaml"


def test_manifest_is_valid_npe2():
    m = yaml.safe_load(MANIFEST.read_text())
    assert m["name"] == "meristem-napari"
    command_ids = {c["id"] for c in m["contributions"]["commands"]}
    # Every reader/widget must reference a declared command.
    for r in m["contributions"]["readers"]:
        assert r["command"] in command_ids
    for w in m["contributions"]["widgets"]:
        assert w["command"] in command_ids
    # The commands point at real python callables (module:function) that exist.
    for c in m["contributions"]["commands"]:
        mod, func = c["python_name"].split(":")
        assert mod.startswith("meristem_napari")
        assert func


def test_choices_come_from_registry():
    assert segmenter_choices() == list_segmenters()
    assert tracker_choices() == list_trackers()
    assert "strack" in tracker_choices()


def test_params_filtered_to_backend_fields():
    # strack has no `device` field, so it must be dropped; cellpose-sam keeps it.
    assert "device" not in params_for_tracker("strack", device="cpu")
    assert params_for_segmenter("cellpose-sam", device="mps")["device"] == "mps"
    # unknown/None values are never passed through
    assert params_for_segmenter("mock", nonsense=1, threshold=None) == {}


def test_to_tyx_promotes_2d():
    assert to_tyx(np.zeros((8, 8))).shape == (1, 8, 8)
    assert to_tyx(np.zeros((3, 8, 8))).shape == (3, 8, 8)
    with pytest.raises(ValueError):
        to_tyx(np.zeros((2, 3, 4, 5)))


def test_roi_from_rectangle_bounding_box():
    # napari rectangle corners in (y, x); expect the clamped bounding box.
    rect = np.array([[10, 20], [10, 60], [40, 60], [40, 20]], dtype=float)
    roi = roi_from_rectangle(rect, image_shape_yx=(100, 100))
    assert (roi.y, roi.x, roi.height, roi.width) == (10, 20, 30, 40)
    # Clamps to the image bounds.
    big = np.array([[-5, -5], [-5, 200], [200, 200], [200, -5]], dtype=float)
    roi2 = roi_from_rectangle(big, image_shape_yx=(50, 50))
    assert (roi2.y, roi2.x, roi2.height, roi2.width) == (0, 0, 50, 50)


def test_segment_to_layer_produces_labels():
    # The segment widget's actual work, exercised without napari.
    img = np.zeros((3, 40, 40), dtype=np.float32)
    img[:, 10:20, 10:20] = 1.0  # a bright block
    data, meta, ltype = segment_to_layer(img, "PH", "mock", min_size_frac=0.0)
    assert ltype == "labels"
    assert data.shape == (3, 40, 40)
    assert data.max() >= 1  # something segmented
    assert "mock" in meta["name"]


def test_segment_to_layer_honors_crop_rect():
    img = np.zeros((2, 50, 50), dtype=np.float32)
    rect = np.array([[10, 10], [10, 30], [35, 30], [35, 10]], dtype=float)  # 25x20 box
    data, _, _ = segment_to_layer(img, "PH", "mock", crop_rect=rect)
    assert data.shape == (2, 25, 20)  # cropped before segmentation


def test_track_to_layer_produces_tracks():
    labels = np.zeros((3, 40, 40), dtype=np.int32)
    labels[:, 10:16, 10:16] = 1  # one persistent cell
    img = labels.astype("float32")
    data, meta, ltype = track_to_layer(labels, img, "PH", "strack")
    assert ltype == "tracks"
    assert data.shape[1] == 4  # [track_id, t, y, x]
    assert "graph" in meta


def test_track_to_layer_frame_window():
    labels = np.zeros((5, 40, 40), dtype=np.int32)
    labels[:, 10:16, 10:16] = 1
    data, _, _ = track_to_layer(labels, labels.astype("float32"), "PH", "strack",
                                frame_start=1, frame_stop=4)
    assert set(np.unique(data[:, 1]).astype(int)) <= {0, 1, 2}  # 3-frame window, local indices


def test_one_click_run_produces_masks_and_tracks():
    # The launcher's "Run" does segment + track in one call and returns both layers.
    img = np.zeros((3, 40, 40), dtype=np.float32)
    img[:, 12:20, 12:20] = 1.0
    layers = segment_and_track_to_layers(img, "PH", "mock", "strack", register=False)
    kinds = [ltype for _, _, ltype in layers]
    assert kinds == ["labels", "tracks"]
    assert layers[0][0].shape == (3, 40, 40)  # masks
    assert layers[1][0].shape[1] == 4  # tracks [track_id, t, y, x]


def test_one_click_run_with_crop_and_register():
    img = np.zeros((3, 60, 60), dtype=np.float32)
    img[:, 20:30, 20:30] = 1.0
    rect = np.array([[15, 15], [15, 45], [45, 45], [45, 15]], dtype=float)
    layers = segment_and_track_to_layers(
        img, "PH", "mock", "strack", crop_rect=rect, register=True
    )
    assert layers[0][0].shape == (3, 30, 30)  # cropped masks


@pytest.mark.skipif(
    importlib.util.find_spec("magicgui") is None, reason="magicgui/napari not installed"
)
def test_widgets_construct():
    # Only runs where the GUI stack is present; confirms the factories build a widget.
    from meristem_napari._widgets import run_widget, segment_widget, track_widget

    assert run_widget() is not None
    assert segment_widget() is not None
    assert track_widget() is not None
