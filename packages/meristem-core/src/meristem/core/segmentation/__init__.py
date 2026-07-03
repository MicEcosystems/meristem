"""Segmentation backend contract and built-in mock backend."""

from .base import SegmenterBackend, SegmenterParams
from .mock import MockSegmenter, MockSegmenterParams

__all__ = ["SegmenterBackend", "SegmenterParams", "MockSegmenter", "MockSegmenterParams"]
