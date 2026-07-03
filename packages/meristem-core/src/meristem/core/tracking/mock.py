"""A dependency-free tracking backend for tests, demos, and pipeline wiring.

:class:`MockTracker` links segmentation masks across time with :func:`greedy_overlap_link`,
producing a real :class:`~meristem.core.contracts.TrackGraph` with divisions. It needs no ML
frameworks and is registered as the ``mock`` tracker via the package entry points.
"""

from __future__ import annotations

from ..contracts import ImageStack, SegMasks, TrackGraph
from .base import TrackerBackend, TrackerParams
from .lineage import greedy_overlap_link


class MockTrackerParams(TrackerParams):
    min_iou: float = 0.1  # minimum intersection-over-union to link a cell to a parent


class MockTracker(TrackerBackend):
    """Greedy mask-overlap tracker with division detection. A baseline, not a learned model."""

    name = "mock"
    Params = MockTrackerParams

    def load(self, params: MockTrackerParams) -> None:  # type: ignore[override]
        self._params = params

    def track(self, stack: ImageStack, masks: SegMasks) -> TrackGraph:
        params: MockTrackerParams = getattr(self, "_params", MockTrackerParams())
        return greedy_overlap_link(masks, min_iou=params.min_iou)
