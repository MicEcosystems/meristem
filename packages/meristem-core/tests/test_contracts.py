"""Contract validation and TrackGraph export round-trips."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from meristem.core.contracts import ROI, ImageStack, SegMasks, TrackGraph


def test_imagestack_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="3D"):
        ImageStack(data=np.zeros((10, 10), dtype=np.float32))


def test_imagestack_rejects_non_array():
    with pytest.raises(TypeError):
        ImageStack(data=[[1, 2], [3, 4]])  # type: ignore[arg-type]


def test_imagestack_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size_um"):
        ImageStack(data=np.zeros((1, 4, 4), dtype=np.float32), pixel_size_um=0)


def test_normalized_range_and_flat_image():
    stack = ImageStack(data=np.array([[[0, 2], [4, 6]]], dtype=np.float32))
    norm = stack.normalized()
    assert norm.min() == 0.0 and norm.max() == 1.0
    flat = ImageStack(data=np.full((1, 3, 3), 5.0, dtype=np.float32))
    assert np.all(flat.normalized() == 0.0)  # no divide-by-zero on a constant image


def test_segmasks_requires_integer_labels():
    with pytest.raises(ValueError, match="integer"):
        SegMasks(data=np.zeros((2, 4, 4), dtype=np.float32))


def test_roi_crop_and_clamp():
    stack = ImageStack(data=np.arange(3 * 10 * 10).reshape(3, 10, 10).astype(np.int32))
    cropped = stack.crop(ROI(y=2, x=3, height=4, width=5))
    assert cropped.shape_yx == (4, 5)
    # An oversized ROI is clamped to the image rather than raising.
    clamped = stack.crop(ROI(y=8, x=8, height=100, width=100))
    assert clamped.shape_yx == (2, 2)


def test_roi_rejects_bad_geometry():
    with pytest.raises(ValueError):
        ROI(y=0, x=0, height=0, width=5)
    with pytest.raises(ValueError):
        ROI(y=-1, x=0, height=5, width=5)


def _linear_lineage_with_division() -> TrackGraph:
    """parent(0) -> {a(1), b(1)} division; a continues to a2(2)."""
    tg = TrackGraph()
    tg.add_detection(0, frame=0, label=1, centroid=(10.0, 10.0))
    tg.add_detection(1, frame=1, label=1, centroid=(10.0, 6.0))
    tg.add_detection(2, frame=1, label=2, centroid=(10.0, 14.0))
    tg.add_detection(3, frame=2, label=1, centroid=(10.0, 6.0))
    tg.link(0, 1)
    tg.link(0, 2)
    tg.link(1, 3)
    return tg


def test_trackgraph_division_detection():
    tg = _linear_lineage_with_division()
    assert tg.divisions() == [0]
    assert tg.roots() == [0]
    assert tg.n_detections == 4


def test_to_napari_tracks_shape_and_graph():
    tg = _linear_lineage_with_division()
    data, graph = tg.to_napari_tracks()
    assert data.shape == (4, 4)  # [track_id, t, y, x]
    # The parent track spans frame 0; two daughter tracks branch from it.
    parent_track_id = int(data[data[:, 1] == 0][0, 0])
    daughter_tracks = [child for child, parents in graph.items() if parent_track_id in parents]
    assert len(daughter_tracks) == 2


def test_to_ctc_rows_encode_parent():
    tg = _linear_lineage_with_division()
    rows = tg.to_ctc()
    by_track = {r[0]: r for r in rows}
    # Two tracks must name a non-zero parent (the daughters); at least one has parent 0 (the root).
    parents = [r[3] for r in rows]
    assert parents.count(0) >= 1
    assert sum(1 for p in parents if p != 0) == 2
    # res_track rows are (track_id, start, end, parent) with start <= end.
    for _tid, start, end, _parent in by_track.values():
        assert start <= end


def test_empty_trackgraph_exports_cleanly():
    tg = TrackGraph(graph=nx.DiGraph())
    data, graph = tg.to_napari_tracks()
    assert data.shape == (0, 4)
    assert graph == {}
    assert tg.to_ctc() == []
