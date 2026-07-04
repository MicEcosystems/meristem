"""Compare mode: tabulate several models on the same input (mock backends, zero ML deps)."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from meristem.core import CompareSpec, ImageStack, format_report, run_comparison


def _write_stack(path, stack: ImageStack) -> None:
    tifffile.imwrite(str(path), stack.data)


def test_compare_segmenters_tabulates_each(tmp_path, dividing_stack: ImageStack):
    p = tmp_path / "stack.tif"
    _write_stack(p, dividing_stack)
    spec = CompareSpec(
        input={"path": str(p), "name": "syn", "pixel_size_um": 0.065},
        segmenters=[
            {"name": "mock", "params": {"threshold": 0.4}},
            {"name": "mock", "params": {"threshold": 0.6}},
        ],
    )
    report = run_comparison(spec)
    assert len(report.segmenters) == 2
    assert report.trackers == []
    for s in report.segmenters:
        assert len(s.cells_per_frame) == dividing_stack.n_frames
        assert s.total_cells == sum(s.cells_per_frame)
        assert s.median_area_um2 is not None  # pixel size supplied
        assert s.seconds >= 0.0


def test_compare_trackers_share_one_segmentation(tmp_path, dividing_stack: ImageStack):
    p = tmp_path / "stack.tif"
    _write_stack(p, dividing_stack)
    spec = CompareSpec(
        input={"path": str(p), "name": "syn"},
        segmenters=[{"name": "mock"}],
        trackers=[{"name": "mock"}, {"name": "strack"}],
    )
    report = run_comparison(spec)
    assert report.shared_segmenter == "mock"  # trackers compared on the mock segmentation
    assert [t.name for t in report.trackers] == ["mock", "strack"]
    for t in report.trackers:
        assert t.n_detections > 0
        assert t.n_tracks > 0
    # The report renders to a table mentioning both sections.
    text = format_report(report)
    assert "Segmentation:" in text and "Tracking (on 'mock' masks):" in text


def test_compare_requires_something_to_compare(tmp_path, dividing_stack: ImageStack):
    p = tmp_path / "stack.tif"
    _write_stack(p, dividing_stack)
    with pytest.raises(Exception):
        CompareSpec(input={"path": str(p)})  # neither segmenters nor trackers


def test_compare_trackers_without_segmenter_rejected(tmp_path):
    with pytest.raises(Exception):  # trackers need a segmenter to produce masks
        CompareSpec(input={"path": "x.tif"}, trackers=[{"name": "strack"}])


def test_track_on_must_be_a_compared_segmenter(tmp_path):
    with pytest.raises(Exception):
        CompareSpec(
            input={"path": "x.tif"},
            segmenters=[{"name": "mock"}],
            trackers=[{"name": "strack"}],
            track_on="cellpose-sam",  # not among compared segmenters
        )


def test_compare_selects_named_channel(tmp_path, dividing_stack: ImageStack):
    ph = tmp_path / "PH.tif"
    gfp = tmp_path / "GFP.tif"
    _write_stack(ph, dividing_stack)
    _write_stack(gfp, ImageStack(data=np.zeros_like(dividing_stack.data)))
    spec = CompareSpec(
        input={
            "channels": [
                {"name": "PH", "path": str(ph)},
                {"name": "GFP", "path": str(gfp)},
            ]
        },
        segmenters=[{"name": "mock"}],
        channel="PH",
    )
    report = run_comparison(spec)
    assert report.channel == "PH"
    assert report.segmenters[0].total_cells > 0  # PH has cells, GFP is blank
