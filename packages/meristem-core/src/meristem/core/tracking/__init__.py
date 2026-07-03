"""Tracking backend contract, lineage helpers, and built-in mock backend."""

from .base import TrackerBackend, TrackerParams
from .lineage import frame_centroids, greedy_overlap_link
from .mock import MockTracker, MockTrackerParams

__all__ = [
    "TrackerBackend",
    "TrackerParams",
    "MockTracker",
    "MockTrackerParams",
    "frame_centroids",
    "greedy_overlap_link",
]
