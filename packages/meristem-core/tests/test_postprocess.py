"""MiDAP-style size filtering of segmentation masks."""

from __future__ import annotations

import numpy as np

from meristem.core import (
    BackendConfig,
    ImageStack,
    PipelineConfig,
    SegMasks,
    filter_by_size,
    run_segmentation,
)


def _masks_with_speck():
    # One frame: a big cell (400 px), a small cell (9 px), and a 1-px speck. With mean ~137, the
    # MiDAP threshold (1% of mean = ~1.4 px) removes only the 1-px speck.
    labels = np.zeros((1, 30, 30), dtype=np.int32)
    labels[0, 2:22, 2:22] = 1  # 400 px
    labels[0, 24:27, 24:27] = 2  # 9 px
    labels[0, 0, 29] = 3  # 1-px speck
    return SegMasks(data=labels)


def test_small_specks_removed_labels_preserved():
    masks = _masks_with_speck()
    out = filter_by_size(masks, min_size_frac=0.01)  # MiDAP default
    remaining = set(np.unique(out.data)) - {0}
    assert remaining == {1, 2}  # 1-px speck (label 3) dropped, real cells keep their ids
    assert out.source.endswith("+sizefilter")


def test_min_size_px_absolute_floor():
    masks = _masks_with_speck()
    # An absolute floor of 50 px removes the 9-px and 1-px labels but keeps the 400-px cell.
    out = filter_by_size(masks, min_size_frac=0.0, min_size_px=50)
    assert set(np.unique(out.data)) - {0} == {1}


def test_max_size_frac_removes_giant_blob():
    # Several ~25-px cells keep the mean low; one giant blob is many times the mean and is removed.
    labels = np.zeros((1, 60, 60), dtype=np.int32)
    for i, (y, x) in enumerate([(0, 0), (0, 10), (10, 0), (10, 10)], start=1):
        labels[0, y : y + 5, x : x + 5] = i  # 25 px each
    labels[0, 30:58, 30:58] = 5  # ~784 px merged blob
    masks = SegMasks(data=labels)
    out = filter_by_size(masks, min_size_frac=0.0, max_size_frac=3.0)
    remaining = set(np.unique(out.data)) - {0}
    assert 5 not in remaining  # oversized blob removed
    assert {1, 2, 3, 4} <= remaining  # normal cells kept


def test_pipeline_applies_postprocess(tmp_path, dividing_stack: ImageStack):
    import tifffile

    p = tmp_path / "PH.tif"
    tifffile.imwrite(str(p), dividing_stack.data)
    cfg = PipelineConfig(
        input={"path": str(p), "name": "PH"},
        segmenter=BackendConfig(name="mock", params={"threshold": 0.5}),
        tracker=BackendConfig(name="strack"),
        postprocess={"min_size_px": 5},
        output={"dir": str(tmp_path / "out")},
    )
    bundle = run_segmentation(cfg, save=False)
    assert bundle.channels[0].masks.source.endswith("+sizefilter")
