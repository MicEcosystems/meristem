"""Per-cell intensity measurement through a segmentation, joined to tracks."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import tifffile

from meristem.core import (
    BackendConfig,
    ImageStack,
    PipelineConfig,
    SegMasks,
    measure_intensities,
)
from meristem.core.measure import UNTRACKED


def test_measure_reads_mean_and_total_per_cell():
    # Two cells with known constant fluorescence values; measurement must recover them exactly.
    labels = np.zeros((1, 10, 10), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1  # 4 px
    labels[0, 5:8, 5:8] = 2  # 9 px
    masks = SegMasks(data=labels)
    gfp = np.zeros((1, 10, 10), dtype=np.uint16)
    gfp[0, 1:3, 1:3] = 100  # cell 1 -> mean 100, total 400
    gfp[0, 5:8, 5:8] = 50  # cell 2 -> mean 50, total 450

    table = measure_intensities(masks, {"GFP": gfp}, pixel_size_um=0.1)
    by_label = {r.label: r for r in table.rows}
    assert by_label[1].intensities["GFP"]["mean"] == 100.0
    assert by_label[1].intensities["GFP"]["total"] == 400.0
    assert by_label[1].area_px == 4
    assert by_label[2].intensities["GFP"]["mean"] == 50.0
    assert by_label[2].intensities["GFP"]["total"] == 450.0


def test_measure_rejects_shape_mismatch():
    masks = SegMasks(data=np.zeros((2, 8, 8), dtype=np.int32))
    with pytest.raises(ValueError, match="shape"):
        measure_intensities(masks, {"GFP": np.zeros((2, 4, 4), dtype=np.uint16)})


def test_track_id_joined_by_centroid(dividing_stack: ImageStack):
    # Segment + track the synthetic stack, then measure a channel through those masks and confirm
    # cells receive real track ids (not UNTRACKED) because centroids line up with the tracks.
    from meristem.core.segmentation.mock import MockSegmenter, MockSegmenterParams
    from meristem.core.tracking.mock import MockTracker, MockTrackerParams

    seg = MockSegmenter()
    seg.load(MockSegmenterParams(threshold=0.5))
    masks = seg.segment(dividing_stack)
    trk = MockTracker()
    trk.load(MockTrackerParams(min_iou=0.05))
    tracks = trk.track(dividing_stack, masks)

    fluor = (dividing_stack.data * 1000).astype(np.uint16)
    table = measure_intensities(masks, {"FL": fluor}, tracks=tracks)
    assert len(table.rows) == sum(masks.n_cells_per_frame())
    assert any(r.track_id != UNTRACKED for r in table.rows)  # cells got linked to tracks


def _mock_multichannel_cfg(ph_path, gfp_path, out_dir, **overrides):
    base = dict(
        input={
            "channels": [
                {"name": "PH", "path": str(ph_path), "segment": True, "track": True},
                {"name": "GFP", "path": str(gfp_path), "segment": False, "measure": True},
            ],
            "pixel_size_um": 0.065,
        },
        segmenter=BackendConfig(name="mock", params={"threshold": 0.5}),
        tracker=BackendConfig(name="strack"),
        measure_on="PH",
        output={"dir": str(out_dir)},
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_pipeline_measures_gfp_through_ph_and_writes_csv(tmp_path, dividing_stack: ImageStack):
    ph = tmp_path / "PH.tif"
    gfp = tmp_path / "GFP.tif"
    tifffile.imwrite(str(ph), dividing_stack.data)
    tifffile.imwrite(str(gfp), (dividing_stack.data * 500).astype(np.uint16))
    out_dir = tmp_path / "out"

    from meristem.core import run_pipeline

    bundle = run_pipeline(_mock_multichannel_cfg(ph, gfp, out_dir), save=True)
    assert bundle.measurements is not None
    assert bundle.measurements.channels == ["GFP"]

    csv_path = out_dir / "measurements.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(open(csv_path)))
    assert rows and {"frame", "label", "track_id", "GFP_mean", "GFP_total"} <= set(rows[0])


def test_measure_requires_measure_on(tmp_path):
    # A measure channel without measure_on must be rejected at config time.
    with pytest.raises(Exception, match="measure_on"):
        PipelineConfig(
            input={
                "channels": [
                    {"name": "PH", "path": "ph.tif", "segment": True},
                    {"name": "GFP", "path": "gfp.tif", "segment": False, "measure": True},
                ]
            },
            segmenter=BackendConfig(name="mock"),
            tracker=BackendConfig(name="mock"),
        )


def test_measure_on_must_be_segmented(tmp_path):
    with pytest.raises(Exception, match="segmented"):
        PipelineConfig(
            input={
                "channels": [
                    {"name": "PH", "path": "ph.tif", "segment": True, "track": True},
                    {"name": "GFP", "path": "gfp.tif", "segment": False, "measure": True},
                ]
            },
            segmenter=BackendConfig(name="mock"),
            tracker=BackendConfig(name="mock"),
            measure_on="GFP",  # GFP is a measure channel, not segmented
        )
