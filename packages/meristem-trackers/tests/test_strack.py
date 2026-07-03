"""S-track native port — fully exercised (pure NumPy, no heavy deps to skip).

Covers discovery/params plus the two behaviors that define S-track: linking a cell across frames,
and validating divisions against the mother's long axis (accept an axial split, reject a
perpendicular one).
"""

from __future__ import annotations

import numpy as np
import pytest

from meristem.core import get_tracker, list_trackers
from meristem.core.contracts import SegMasks
from meristem.core.tracking.base import TrackerBackend
from meristem.trackers import STrackTracker
from meristem.trackers.strack_tracker import STrackParams


def test_strack_registered_and_resolvable():
    assert "strack" in list_trackers()
    cls = get_tracker("strack")
    assert cls is STrackTracker
    assert issubclass(cls, TrackerBackend)
    assert cls.name == "strack"


def test_params_defaults_and_validation():
    p = STrackParams()
    assert p.max_dist == 30.0 and p.max_angle == 30.0
    with pytest.raises(Exception):  # extra="forbid"
        STrackParams(unknown=1)


def _horizontal_mother(frames: int) -> np.ndarray:
    """A horizontal rod (long axis along x) for the first two frames of a T=5 stack."""
    data = np.zeros((frames, 40, 60), dtype=np.int32)
    data[0, 18:23, 10:40] = 1
    data[1, 18:23, 10:42] = 1
    return data


def test_axial_division_is_accepted():
    # Daughters split left/right — along the mother's long axis => a valid division.
    data = _horizontal_mother(5)
    for t in (2, 3, 4):
        data[t, 18:23, 10:24] = 1  # left daughter
        data[t, 18:23, 28:42] = 2  # right daughter

    tracker = STrackTracker()
    tracker.load(STrackParams())
    tg = tracker.track(_dummy_stack(data), SegMasks(data=data))

    assert tg.n_detections == 8  # 1 + 1 + 2 + 2 + 2
    assert len(tg.divisions()) == 1  # the mother splits exactly once
    assert len(tg.roots()) == 1  # a single founder lineage


def test_perpendicular_division_is_rejected():
    # Daughters split top/bottom — across the mother's long axis => not a real division.
    data = _horizontal_mother(5)
    for t in (2, 3, 4):
        data[t, 8:20, 10:42] = 1  # top "daughter"
        data[t, 22:34, 10:42] = 2  # bottom "daughter"

    tracker = STrackTracker()
    tracker.load(STrackParams())
    tg = tracker.track(_dummy_stack(data), SegMasks(data=data))

    # The angle check refuses to call this a division; the second cell starts its own lineage.
    assert len(tg.divisions()) == 0
    assert len(tg.roots()) >= 2


def test_distance_gate_prevents_teleporting_links():
    # Two cells, far apart, no overlap: beyond max_dist they must not link.
    data = np.zeros((2, 40, 80), dtype=np.int32)
    data[0, 18:22, 5:10] = 1
    data[1, 18:22, 70:75] = 1  # ~65 px away
    tracker = STrackTracker()
    tracker.load(STrackParams(max_dist=10.0))
    tg = tracker.track(_dummy_stack(data), SegMasks(data=data))
    assert tg.graph.number_of_edges() == 0  # nothing linked across the gap


def _dummy_stack(mask_data: np.ndarray):
    from meristem.core.contracts import ImageStack

    return ImageStack(data=mask_data.astype("float32"))
