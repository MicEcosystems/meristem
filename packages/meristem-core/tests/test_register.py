"""Drift registration: recover known shifts and correct them, and integrate before crop."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from meristem.core import (
    BackendConfig,
    PipelineConfig,
    apply_shifts,
    estimate_drift,
    run_segmentation,
)
from meristem.core.register import _shift_frame, crop_with_drift


def _blobby_frame(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)
    for _ in range(8):
        cy, cx = rng.integers(12, h - 12), rng.integers(12, w - 12)
        img[cy - 3 : cy + 3, cx - 3 : cx + 3] = 1.0
    return img


def test_estimate_drift_recovers_known_shifts():
    base = _blobby_frame()
    # Build a stack where each frame is the base drifted by a known cumulative amount.
    true_shifts = np.array([[0, 0], [0, 3], [-2, 5], [-2, 8]], dtype=float)
    stack = np.stack([_shift_frame(base, int(dy), int(dx)) for dy, dx in true_shifts])
    est = estimate_drift(stack, reference="previous")
    assert np.allclose(est, true_shifts, atol=1.0)  # integer-pixel recovery


def test_apply_shifts_realigns_to_reference():
    base = _blobby_frame(seed=1)
    true_shifts = np.array([[0, 0], [4, -3], [7, 2]], dtype=float)
    drifted = np.stack([_shift_frame(base, int(dy), int(dx)) for dy, dx in true_shifts])
    est = estimate_drift(drifted, reference="first")
    aligned = apply_shifts(drifted, est)
    # After correction, every frame's overlap region matches frame 0 far better than before.
    def overlap_err(stack):
        return np.mean([np.abs(stack[t] - stack[0]).mean() for t in range(1, stack.shape[0])])

    assert overlap_err(aligned) < overlap_err(drifted)


def test_crop_with_drift_follows_a_moving_object():
    # A small bright square drifts across frames; a following crop keeps it centered without
    # resampling (MiDAP's cutout-tracks-the-cells approach).
    h = w = 60
    base = np.zeros((h, w), dtype=np.uint16)
    base[28:32, 28:32] = 255  # 4x4 square centered at (30, 30)
    drift = [(0, 0), (5, -4), (10, 6)]
    frames = np.stack([_shift_frame(base, dy, dx) for dy, dx in drift]).astype(np.uint16)
    shifts = np.array(drift, dtype=float)  # true content drift

    # Crop a 10x10 box originally around the square; it should track the square every frame.
    cropped = crop_with_drift(frames, y=25, x=25, height=10, width=10, shifts=shifts)
    assert cropped.shape == (3, 10, 10)
    for t in range(3):
        assert cropped[t].max() == 255  # the square is inside the (moved) window every frame
        # centroid of the bright pixels stays near the window centre across frames
        ys, xs = np.nonzero(cropped[t])
        assert 3 <= ys.mean() <= 7 and 3 <= xs.mean() <= 7


def test_estimate_drift_requires_3d():
    with pytest.raises(ValueError):
        estimate_drift(np.zeros((10, 10)))


def test_registration_applied_before_crop_in_pipeline(tmp_path):
    # A drifting stack with a bright square; after registration the square sits still, so a fixed
    # crop keeps it — verify the pipeline runs registration and the cropped masks are non-empty.
    h = w = 80
    base = np.zeros((h, w), dtype=np.uint16)
    base[30:50, 30:50] = 4000  # bright cell block near center
    shifts = [(0, 0), (3, -3), (6, -6), (9, 0), (12, 4)]
    frames = np.stack([_shift_frame(base, dy, dx) for dy, dx in shifts]).astype(np.uint16)
    p = tmp_path / "PH.tif"
    # photometric=minisblack so tifffile stores frames as pages, not RGB planes.
    tifffile.imwrite(str(p), frames, photometric="minisblack")

    out = tmp_path / "out"
    cfg = PipelineConfig(
        input={"path": str(p), "name": "PH"},
        register={"on": "PH", "reference": "previous"},
        crop={"y": 28, "x": 28, "height": 24, "width": 24},  # tight box around the (registered) cell
        segmenter=BackendConfig(name="mock", params={"threshold": 0.5}),
        tracker=BackendConfig(name="strack"),
        output={"dir": str(out)},
    )
    bundle = run_segmentation(cfg, save=True)
    # Drift shifts were saved for the tracking stage to reuse.
    assert (out / "PH_drift.npy").exists()
    # The block stays in the crop across all frames thanks to registration.
    counts = bundle.channels[0].masks.n_cells_per_frame()
    assert all(c >= 1 for c in counts)


def test_register_on_must_be_a_channel(tmp_path):
    with pytest.raises(Exception, match="register.on"):
        PipelineConfig(
            input={"path": "x.tif", "name": "PH"},
            register={"on": "GFP"},  # not a channel in a single-path input (named PH)
            segmenter=BackendConfig(name="mock"),
            tracker=BackendConfig(name="mock"),
        )
