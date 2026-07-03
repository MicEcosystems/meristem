"""Registry: decorator registration, entry-point discovery, and error behavior."""

from __future__ import annotations

import pytest

from meristem.core import registry
from meristem.core.contracts import ImageStack, SegMasks, TrackGraph
from meristem.core.segmentation.base import SegmenterBackend
from meristem.core.tracking.base import TrackerBackend


def test_builtin_mocks_discovered_via_entry_points():
    # These come from the installed package's entry points, not from decorators in this test.
    assert "mock" in registry.list_segmenters()
    assert "mock" in registry.list_trackers()


def test_get_segmenter_returns_class_and_sets_name():
    cls = registry.get_segmenter("mock")
    assert issubclass(cls, SegmenterBackend)
    assert cls.name == "mock"


def test_unknown_backend_raises_with_helpful_message():
    with pytest.raises(registry.BackendNotFoundError) as exc:
        registry.get_segmenter("does-not-exist")
    assert "Available segmenters" in str(exc.value)


def test_decorator_registration_roundtrip():
    @registry.register_segmenter("unit-test-seg")
    class _Seg(SegmenterBackend):
        def load(self, params):  # noqa: D401
            pass

        def segment(self, stack: ImageStack) -> SegMasks:  # pragma: no cover - not run
            raise NotImplementedError

    try:
        assert "unit-test-seg" in registry.list_segmenters()
        assert registry.get_segmenter("unit-test-seg") is _Seg
    finally:
        registry._segmenters.pop("unit-test-seg", None)


def test_registering_non_subclass_is_rejected():
    with pytest.raises(TypeError):
        registry.register_tracker("bad")(object)  # type: ignore[arg-type]


def test_tracker_backend_registration():
    @registry.register_tracker("unit-test-track")
    class _Track(TrackerBackend):
        def load(self, params):
            pass

        def track(self, stack, masks) -> TrackGraph:  # pragma: no cover - not run
            raise NotImplementedError

    try:
        assert registry.get_tracker("unit-test-track") is _Track
    finally:
        registry._trackers.pop("unit-test-track", None)
