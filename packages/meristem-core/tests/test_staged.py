"""Modular workflow: segment standalone, inspect, then track a chosen frame window."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from meristem.core import (
    BackendConfig,
    ImageStack,
    PipelineConfig,
    run_segmentation,
    run_tracking,
)


def _cfg(stack_path, out_dir, **overrides):
    base = dict(
        input={"path": str(stack_path), "name": "PH"},
        segmenter=BackendConfig(name="mock", params={"threshold": 0.5}),
        tracker=BackendConfig(name="strack"),
        output={"dir": str(out_dir)},
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_segment_writes_masks_without_tracks(tmp_path, dividing_stack: ImageStack):
    p = tmp_path / "PH.tif"
    tifffile.imwrite(str(p), dividing_stack.data)
    out = tmp_path / "out"
    bundle = run_segmentation(_cfg(p, out), save=True)

    assert bundle.channels[0].tracks is None  # segmentation stage produces no tracks
    assert (out / "PH_masks.tif").exists()
    assert (out / "PH_seg_bin.tif").exists()
    assert not (out / "PH_tracks.npy").exists()  # nothing tracked yet


def test_track_consumes_saved_masks(tmp_path, dividing_stack: ImageStack):
    p = tmp_path / "PH.tif"
    tifffile.imwrite(str(p), dividing_stack.data)
    out = tmp_path / "out"
    cfg = _cfg(p, out)

    run_segmentation(cfg, save=True)  # stage 1 writes PH_masks.tif
    bundle = run_tracking(cfg, save=True)  # stage 2 reads them and tracks

    ch = bundle.channels[0]
    assert ch.tracks is not None and ch.tracks.n_detections > 0
    assert (out / "PH_tracks.npy").exists()
    assert (out / "PH_res_track.txt").exists()
    # The saved full-length masks are not overwritten by the tracking stage.
    saved = tifffile.imread(out / "PH_masks.tif")
    assert saved.shape[0] == dividing_stack.n_frames


def test_track_on_frame_window(tmp_path, dividing_stack: ImageStack):
    p = tmp_path / "PH.tif"
    tifffile.imwrite(str(p), dividing_stack.data)
    out = tmp_path / "out"
    cfg = _cfg(p, out)
    run_segmentation(cfg, save=True)

    # Track only frames 2..5 (where the daughters exist) — the modular quality-based choice.
    bundle = run_tracking(cfg, frames=(2, 5), save=False)
    ch = bundle.channels[0]
    assert ch.stack.n_frames == 3  # window applied to both image and masks
    assert ch.masks.n_frames == 3
    assert ch.tracks.n_detections > 0


def test_track_window_out_of_range_is_rejected(tmp_path, dividing_stack: ImageStack):
    p = tmp_path / "PH.tif"
    tifffile.imwrite(str(p), dividing_stack.data)
    out = tmp_path / "out"
    cfg = _cfg(p, out)
    run_segmentation(cfg, save=True)
    with pytest.raises(ValueError):
        run_tracking(cfg, frames=(10, 20), save=False)  # beyond the 5 segmented frames


def test_staged_measurement_on_window(tmp_path, dividing_stack: ImageStack):
    ph = tmp_path / "PH.tif"
    gfp = tmp_path / "GFP.tif"
    tifffile.imwrite(str(ph), dividing_stack.data)
    tifffile.imwrite(str(gfp), (dividing_stack.data * 300).astype(np.uint16))
    out = tmp_path / "out"
    cfg = _cfg(
        ph,
        out,
        input={
            "channels": [
                {"name": "PH", "path": str(ph), "segment": True, "track": True},
                {"name": "GFP", "path": str(gfp), "segment": False, "measure": True},
            ],
            "pixel_size_um": 0.065,
        },
        measure_on="PH",
    )
    run_segmentation(cfg, save=True)
    bundle = run_tracking(cfg, frames=(2, 5), save=True)

    assert bundle.measurements is not None
    # Every measured cell-frame is within the tracked window.
    assert all(0 <= r.frame < 3 for r in bundle.measurements.rows)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["measurements"]["channels"] == ["GFP"]
