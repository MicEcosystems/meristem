"""End-to-end pipeline smoke tests with the mock backends (zero ML deps)."""

from __future__ import annotations

import json

import numpy as np

from meristem.core import (
    BackendConfig,
    ImageStack,
    PipelineConfig,
    ROIConfig,
    run_on_stack,
    run_pipeline,
)
from meristem.core.io import read_image_stack
from meristem.core.segmentation.mock import MockSegmenter, MockSegmenterParams


def _mock_config(**overrides) -> PipelineConfig:
    base = dict(
        input={"path": "unused.tif", "name": "synthetic"},
        segmenter=BackendConfig(name="mock", params={"threshold": 0.5}),
        tracker=BackendConfig(name="mock", params={"min_iou": 0.05}),
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_mock_segmenter_labels_two_daughters(dividing_stack: ImageStack):
    seg = MockSegmenter()
    seg.load(MockSegmenterParams(threshold=0.5))
    masks = seg.segment(dividing_stack)
    assert masks.data.shape == dividing_stack.data.shape
    counts = masks.n_cells_per_frame()
    assert counts[0] == 1  # one cell early
    assert counts[-1] == 2  # two daughters after division


def test_run_on_stack_produces_lineage_with_division(dividing_stack: ImageStack):
    bundle = run_on_stack(dividing_stack, _mock_config())
    assert bundle.masks.n_frames == 5
    assert bundle.tracks.n_detections > 0
    # The synthetic stack encodes exactly one division; the tracker must recover at least one.
    assert len(bundle.tracks.divisions()) >= 1


def test_crop_stage_is_applied(dividing_stack: ImageStack):
    cfg = _mock_config(crop=ROIConfig(y=10, x=10, height=20, width=20))
    bundle = run_on_stack(dividing_stack, cfg)
    assert bundle.stack.shape_yx == (20, 20)  # downstream operates on the cropped FOV


def test_seamless_swap_changes_only_the_name(dividing_stack: ImageStack):
    # The whole point: selecting a backend is a name change, nothing else in the call site moves.
    cfg_a = _mock_config(segmenter=BackendConfig(name="mock"))
    bundle = run_on_stack(dividing_stack, cfg_a)
    assert bundle.segmenter == "mock"


def test_full_run_pipeline_writes_result_bundle(tmp_path, dividing_stack: ImageStack):
    # Persist the synthetic stack, then run the file-based pipeline against it.
    import tifffile

    stack_path = tmp_path / "stack.tif"
    tifffile.imwrite(str(stack_path), dividing_stack.data)

    out_dir = tmp_path / "results"
    cfg = _mock_config(
        input={"path": str(stack_path), "name": "synthetic", "pixel_size_um": 0.065},
        output={"dir": str(out_dir)},
    )
    bundle = run_pipeline(cfg, save=True)
    assert bundle.tracker == "mock"

    manifest_path = out_dir / "synthetic_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["segmenter"] == "mock"
    assert manifest["n_divisions"] >= 1
    assert (out_dir / manifest["files"]["masks"]).exists()
    assert (out_dir / manifest["files"]["ctc"]).exists()

    # Saved napari-tracks array round-trips to the expected 4-column shape.
    tracks = np.load(out_dir / manifest["files"]["tracks"])
    assert tracks.ndim == 2 and tracks.shape[1] == 4


def test_read_image_stack_promotes_2d(tmp_path):
    import tifffile

    p = tmp_path / "single.tif"
    tifffile.imwrite(str(p), np.zeros((8, 8), dtype=np.uint16))
    stack = read_image_stack(p)
    assert stack.data.shape == (1, 8, 8)
