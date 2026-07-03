"""Shared fixtures: a synthetic monolayer time-lapse with a known division."""

from __future__ import annotations

import numpy as np
import pytest

from meristem.core.contracts import ImageStack


@pytest.fixture
def dividing_stack() -> ImageStack:
    """A 5-frame synthetic FOV containing exactly one cell that divides.

    Frames 0-1: one blob. Frames 2-4: two blobs (the daughters), spatially overlapping the parent
    so the greedy overlap tracker links them and records a division. Built as raw intensity so the
    mock segmenter's threshold path is exercised end-to-end.
    """
    t, h, w = 5, 40, 40
    data = np.zeros((t, h, w), dtype=np.float32)

    def disk(frame: int, cy: int, cx: int, r: int) -> None:
        ys, xs = np.ogrid[:h, :w]
        data[frame][(ys - cy) ** 2 + (xs - cx) ** 2 <= r * r] = 1.0

    # One growing cell for the first two frames.
    disk(0, 20, 20, 6)
    disk(1, 20, 20, 7)
    # Then it splits into two daughters far enough apart to be distinct connected components
    # (centers 16px apart, radius 5 => a gap between them) yet each still overlapping the
    # parent's frame-1 footprint (x in [13, 27]) so the overlap tracker links both to it.
    for f in (2, 3, 4):
        disk(f, 20, 12, 5)
        disk(f, 20, 28, 5)

    return ImageStack(data=data, pixel_size_um=0.065, frame_interval_s=60.0, name="synthetic")
