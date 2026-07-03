"""Shared plumbing for tracking backends.

The most valuable piece here is :func:`trackgraph_from_napari`, which turns napari's Tracks-layer
representation (``[track_id, t, y, x]`` rows + a ``{child: [parents]}`` division graph) into
Meristem's per-detection :class:`~meristem.core.contracts.TrackGraph`. Many modern trackers
(Trackastra, ultrack, btrack) can already emit the napari format, so routing them all through this
converter means every tracker produces the identical lineage object — divisions and all — for the
rest of the pipeline. It is also fully testable *without* any tracker installed, which is why the
conversion logic lives here rather than inside a specific backend.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from meristem.core.contracts import TrackGraph


def import_or_hint(module: str, *, backend: str, extra: str):
    """Import a heavy tracking dependency by name, or raise a clear install hint.

    Called only inside ``load()`` so the backend stays discoverable without the library present.
    """
    try:
        return __import__(module, fromlist=["_"])
    except ModuleNotFoundError as exc:
        top = module.split(".")[0]
        missing = exc.name or ""
        # A missing *transitive* dependency should surface as-is, not be reported as the whole
        # backend library being absent.
        if missing != top and not missing.startswith(top + "."):
            raise
        raise ModuleNotFoundError(
            f"the '{backend}' tracking backend requires the '{module}' package, which is not "
            f"installed. Install it with:  pip install 'meristem-trackers[{extra}]'"
        ) from exc


def trackgraph_from_napari(
    data: np.ndarray, division_graph: Dict[int, Sequence[int]]
) -> TrackGraph:
    """Build a :class:`TrackGraph` from napari Tracks data plus a division graph.

    Parameters
    ----------
    data:
        ``(N, 4)`` array of ``[track_id, t, y, x]`` rows (one per detection).
    division_graph:
        Maps a child ``track_id`` to its parent ``track_id``(s), as produced by
        ``trackastra.tracking.graph_to_napari_tracks`` (and napari's own Tracks graph).

    Each row becomes a detection node (its ``label`` is the consistent ``track_id``). Detections
    of the same track in consecutive rows are linked; a parent track's last detection is linked to
    each child track's first detection, materializing divisions as branching edges.
    """
    tg = TrackGraph()
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return tg

    # Group detections by track, ordered in time; assign each a stable node id.
    order = np.lexsort((arr[:, 1], arr[:, 0]))  # sort by track_id, then t
    track_nodes: Dict[int, List[int]] = defaultdict(list)
    for node_id, row_idx in enumerate(order):
        track_id, t, y, x = arr[row_idx]
        track_id = int(track_id)
        tg.add_detection(node_id, frame=int(t), label=track_id, centroid=(y, x))
        track_nodes[track_id].append(node_id)

    # Intra-track links (consecutive detections of the same track).
    for nodes in track_nodes.values():
        for parent, child in zip(nodes, nodes[1:]):
            tg.link(parent, child)

    # Division links: last detection of the parent track -> first detection of each child track.
    for child_track, parents in division_graph.items():
        child_nodes = track_nodes.get(int(child_track))
        if not child_nodes:
            continue
        child_first = child_nodes[0]
        for parent_track in parents:
            parent_nodes = track_nodes.get(int(parent_track))
            if parent_nodes:
                tg.link(parent_nodes[-1], child_first)

    return tg
