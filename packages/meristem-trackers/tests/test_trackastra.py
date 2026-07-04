"""Discovery/introspection without the heavy library, plus a real test of the output converter.

Trackastra itself isn't installed in this environment, so the backend can't run here — but its
discovery, parameter validation, and missing-dependency behavior are all testable, and the
napari->TrackGraph conversion (the logic most likely to harbor bugs) is fully unit-tested with a
synthetic lineage.
"""

from __future__ import annotations

import numpy as np
import pytest

from meristem.core import get_tracker, list_trackers
from meristem.core.tracking.base import TrackerBackend
from meristem.trackers import TrackastraTracker
from meristem.trackers._common import trackgraph_from_napari
from meristem.trackers.trackastra_tracker import TrackastraParams


def test_trackastra_registered_via_entry_point():
    assert "trackastra" in list_trackers()


def test_registry_returns_class_with_name():
    cls = get_tracker("trackastra")
    assert cls is TrackastraTracker
    assert issubclass(cls, TrackerBackend)
    assert cls.name == "trackastra"


def test_params_defaults_and_validation():
    p = TrackastraParams()
    assert p.model_name == "general_2d"
    assert p.mode == "greedy"
    with pytest.raises(Exception):  # invalid mode (not in Literal)
        TrackastraParams(mode="teleport")
    with pytest.raises(Exception):  # unknown key (extra="forbid")
        TrackastraParams(nonsense=1)


def test_load_without_library_gives_helpful_hint():
    import importlib.util

    if importlib.util.find_spec("trackastra") is not None:
        pytest.skip("trackastra is installed; missing-dependency path does not apply")
    with pytest.raises(ModuleNotFoundError) as exc:
        TrackastraTracker().load(TrackastraParams())
    msg = str(exc.value)
    assert "meristem-trackers" in msg and "trackastra" in msg


# --- the converter: the real logic worth testing here -----------------------
def _synthetic_division():
    """Track 1 (frames 0-1) divides into tracks 2 and 3 (frames 2-3)."""
    data = np.array(
        [
            [1, 0, 10, 10],
            [1, 1, 10, 10],
            [2, 2, 10, 6],
            [2, 3, 10, 6],
            [3, 2, 10, 14],
            [3, 3, 10, 14],
        ],
        dtype=float,
    )
    division_graph = {2: [1], 3: [1]}
    return data, division_graph


def test_trackgraph_from_napari_reconstructs_lineage():
    data, division_graph = _synthetic_division()
    tg = trackgraph_from_napari(data, division_graph)

    assert tg.n_detections == 6
    assert len(tg.roots()) == 1  # only track 1's first detection has no parent
    assert len(tg.divisions()) == 1  # the parent's last detection branches into two daughters

    # And it exports to CTC with exactly two child tracks naming a non-zero parent.
    parents = [row[3] for row in tg.to_ctc()]
    assert parents.count(0) >= 1
    assert sum(1 for p in parents if p != 0) == 2


def test_trackgraph_from_napari_empty():
    tg = trackgraph_from_napari(np.empty((0, 4)), {})
    assert tg.n_detections == 0


def test_trackgraph_from_napari_accepts_scalar_parent():
    # Trackastra's graph_to_napari_tracks maps child -> a single parent int (not a list); the
    # converter must accept that as well as napari's own child -> [parents] form.
    data, _ = _synthetic_division()
    tg = trackgraph_from_napari(data, {2: 1, 3: 1})  # scalar parents
    assert len(tg.divisions()) == 1
    assert tg.n_detections == 6


@pytest.mark.slow
def test_trackastra_real_tracking_if_available():
    pytest.importorskip("trackastra")
    from meristem.core.contracts import ImageStack, SegMasks

    rng = np.random.default_rng(0)
    stack = ImageStack(data=rng.random((3, 64, 64)).astype("float32"))
    masks = SegMasks(data=np.zeros((3, 64, 64), dtype=np.int32))
    masks.data[:, 20:30, 20:30] = 1  # one persistent blob
    tracker = TrackastraTracker()
    tracker.load(TrackastraParams(device="cpu"))
    tg = tracker.track(stack, masks)
    assert tg.n_detections >= 1
