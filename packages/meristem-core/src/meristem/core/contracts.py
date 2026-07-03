"""Typed data contracts shared by every backend.

These three types are the whole point of ``meristem``: if a segmentation model consumes an
:class:`ImageStack` and returns a :class:`SegMasks`, and a tracker consumes those and returns a
:class:`TrackGraph`, then *any* model can be swapped for *any* other without the surrounding
pipeline knowing or caring. The contracts pin down the things MiDAP left implicit — axis order,
dtype, and pixel-size metadata — exactly once, here.

The array-carrying types are deliberately *not* Pydantic models (Pydantic and large numpy arrays
mix poorly). They are frozen dataclasses that validate in ``__post_init__``. Pydantic is reserved
for configuration and backend parameters, where validation and (de)serialization actually help.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator

import networkx as nx
import numpy as np

# Canonical axis order for everything that flows through the pipeline. 2D + time only for v1
# (bacterial monolayers are imaged as single-plane time-lapses); a Z axis can be added later
# without changing this contract's consumers if we keep the order T-first.
AXES = "TYX"


@dataclass(frozen=True)
class ImageStack:
    """A single field of view over time: one channel, shape ``(T, Y, X)``.

    Parameters
    ----------
    data:
        ``(T, Y, X)`` array. Any real dtype; :meth:`normalized` rescales to float32 in [0, 1].
    pixel_size_um:
        Physical pixel size in microns (isotropic). Carried through so downstream analysis
        (growth rates, cell lengths) has real units instead of pixels. ``None`` if unknown.
    frame_interval_s:
        Seconds between frames. ``None`` if unknown.
    name:
        Human-readable identifier for the FOV / position (used in output filenames and logs).
    """

    data: np.ndarray
    pixel_size_um: float | None = None
    frame_interval_s: float | None = None
    name: str = "fov"

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"ImageStack.data must be a numpy array, got {type(self.data)!r}")
        if self.data.ndim != 3:
            raise ValueError(
                f"ImageStack.data must be 3D {tuple(AXES)!r}=(T, Y, X); got shape {self.data.shape}"
            )
        if self.pixel_size_um is not None and self.pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um must be positive, got {self.pixel_size_um}")
        if self.frame_interval_s is not None and self.frame_interval_s <= 0:
            raise ValueError(f"frame_interval_s must be positive, got {self.frame_interval_s}")

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape_yx(self) -> tuple[int, int]:
        return int(self.data.shape[1]), int(self.data.shape[2])

    def normalized(self) -> np.ndarray:
        """Return the stack rescaled per-stack to float32 in [0, 1].

        Backends that want a different normalization (percentile clipping, per-frame, etc.) are
        free to ignore this and work from :attr:`data`; this is just the common default so simple
        models don't each reimplement it.
        """
        arr = self.data.astype(np.float32, copy=False)
        lo = float(arr.min())
        hi = float(arr.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - lo) / (hi - lo)

    def crop(self, roi: "ROI") -> "ImageStack":
        """Return a new stack cropped to ``roi`` (applied to every frame)."""
        y0, x0, y1, x1 = roi.clamped_to(self.shape_yx).bounds
        return replace(self, data=self.data[:, y0:y1, x0:x1])


@dataclass(frozen=True)
class ROI:
    """A rectangular region of interest in pixel coordinates: the manual crop bounds.

    Stored as (y, x, height, width) — the same convention napari uses for a rectangle Shapes
    layer — so the interactive picker and the config file agree without conversion.
    """

    y: int
    x: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"ROI height/width must be positive, got {self.height}x{self.width}")
        if self.y < 0 or self.x < 0:
            raise ValueError(f"ROI origin must be non-negative, got (y={self.y}, x={self.x})")

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Half-open pixel bounds ``(y0, x0, y1, x1)`` suitable for array slicing."""
        return self.y, self.x, self.y + self.height, self.x + self.width

    def clamped_to(self, shape_yx: tuple[int, int]) -> "ROI":
        """Clamp the ROI so it stays inside an image of the given ``(Y, X)`` shape."""
        max_y, max_x = shape_yx
        y = min(self.y, max_y - 1)
        x = min(self.x, max_x - 1)
        height = min(self.height, max_y - y)
        width = min(self.width, max_x - x)
        return ROI(y=y, x=x, height=height, width=width)


@dataclass(frozen=True)
class SegMasks:
    """Instance-labeled segmentation over time: shape ``(T, Y, X)``, integer labels.

    Label ``0`` is background. Non-zero labels identify cell instances *within a frame*; they are
    **not** required to be consistent across frames — establishing cross-frame identity is the
    tracker's job. This mirrors what every segmentation backend natively produces and keeps the
    segmentation/tracking boundary clean.
    """

    data: np.ndarray
    source: str = "unknown"  # backend name that produced these masks (provenance)

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"SegMasks.data must be a numpy array, got {type(self.data)!r}")
        if self.data.ndim != 3:
            raise ValueError(
                f"SegMasks.data must be 3D (T, Y, X); got shape {self.data.shape}"
            )
        if not np.issubdtype(self.data.dtype, np.integer):
            raise ValueError(
                f"SegMasks.data must have an integer dtype (instance labels); got {self.data.dtype}"
            )

    @property
    def n_frames(self) -> int:
        return int(self.data.shape[0])

    def labels_in_frame(self, t: int) -> np.ndarray:
        """Sorted non-zero instance labels present in frame ``t``."""
        vals = np.unique(self.data[t])
        return vals[vals != 0]

    def n_cells_per_frame(self) -> list[int]:
        return [int(self.labels_in_frame(t).size) for t in range(self.n_frames)]


# ---------------------------------------------------------------------------
# Lineage / tracking result
# ---------------------------------------------------------------------------

# Node attribute keys used on the TrackGraph. Kept as constants so exporters and backends agree.
NODE_FRAME = "t"  # int frame index
NODE_LABEL = "label"  # int instance label in SegMasks at that frame
NODE_CENTROID = "centroid"  # (y, x) float centroid in pixels


@dataclass
class TrackGraph:
    """The unified tracking result: a lineage forest as a directed graph.

    Nodes are *cell detections* — one per (frame, instance-label). Directed edges point from a
    detection to its successor(s) in the next frame; a node with two out-edges is a **division**.
    This single structure replaces MiDAP's flattened ``(H, W, 2)`` daughter-assignment arrays and
    makes divisions and lineages first-class. Every tracker backend returns one of these, and it
    exports losslessly to napari's Tracks layer and to the Cell Tracking Challenge format.

    Node ids are opaque integers. Use :meth:`add_detection` to build the graph so the required
    node attributes (:data:`NODE_FRAME`, :data:`NODE_LABEL`, :data:`NODE_CENTROID`) are always set.
    """

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_detection(
        self, node_id: int, *, frame: int, label: int, centroid: tuple[float, float]
    ) -> None:
        self.graph.add_node(
            node_id,
            **{NODE_FRAME: int(frame), NODE_LABEL: int(label), NODE_CENTROID: (
                float(centroid[0]),
                float(centroid[1]),
            )},
        )

    def link(self, parent_node: int, child_node: int) -> None:
        """Link a detection to its successor in the next frame."""
        self.graph.add_edge(parent_node, child_node)

    # -- queries ------------------------------------------------------------
    @property
    def n_detections(self) -> int:
        return self.graph.number_of_nodes()

    def divisions(self) -> list[int]:
        """Node ids where a cell divides (out-degree >= 2)."""
        return [n for n in self.graph.nodes if self.graph.out_degree(n) >= 2]

    def roots(self) -> list[int]:
        """Detections with no parent (cells present at the start or entering the FOV)."""
        return [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]

    def iter_frames(self) -> Iterator[tuple[int, int]]:
        """Yield ``(node_id, frame)`` for every detection."""
        for n, data in self.graph.nodes(data=True):
            yield n, data[NODE_FRAME]

    # -- exporters ----------------------------------------------------------
    def to_napari_tracks(self) -> tuple[np.ndarray, dict[int, list[int]]]:
        """Export to napari's Tracks layer format.

        Returns ``(data, graph)`` where ``data`` is an ``(N, 4)`` array of
        ``[track_id, t, y, x]`` rows and ``graph`` maps a child track_id to its parent track_id(s),
        encoding divisions. A *track* is a maximal chain of detections between divisions; each such
        chain gets one stable ``track_id``.
        """
        track_of, parent_track = self._assign_track_ids()
        rows: list[tuple[int, int, float, float]] = []
        for node, ndata in self.graph.nodes(data=True):
            y, x = ndata[NODE_CENTROID]
            rows.append((track_of[node], ndata[NODE_FRAME], y, x))
        rows.sort(key=lambda r: (r[0], r[1]))
        data = (
            np.array(rows, dtype=float)
            if rows
            else np.empty((0, 4), dtype=float)
        )
        graph = {child: parents for child, parents in parent_track.items()}
        return data, graph

    def to_ctc(self) -> "list[tuple[int, int, int, int]]":
        """Export lineage to Cell Tracking Challenge ``res_track.txt`` rows.

        Each row is ``(track_id, start_frame, end_frame, parent_track_id)`` with
        ``parent_track_id == 0`` for tracks that start without a parent. This is the de-facto
        interchange format consumed by traccuracy and the CTC evaluation tools.
        """
        track_of, parent_track = self._assign_track_ids()
        frames_by_track: dict[int, list[int]] = {}
        for node, ndata in self.graph.nodes(data=True):
            frames_by_track.setdefault(track_of[node], []).append(ndata[NODE_FRAME])
        rows: list[tuple[int, int, int, int]] = []
        for track_id, frames in sorted(frames_by_track.items()):
            parents = parent_track.get(track_id, [])
            parent = parents[0] if parents else 0
            rows.append((track_id, min(frames), max(frames), parent))
        return rows

    # -- internals ----------------------------------------------------------
    def _assign_track_ids(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        """Segment the lineage graph into tracks (chains split at divisions/merges).

        A new track begins at a root, immediately after a division (each daughter), or wherever a
        node has no unique predecessor. Returns ``(track_of_node, parent_tracks_of_track)``.
        """
        track_of: dict[int, int] = {}
        parent_tracks: dict[int, list[int]] = {}
        next_track_id = 1

        # Deterministic order so track ids are stable across runs.
        for start in sorted(self._track_starts(), key=self._sort_key):
            track_id = next_track_id
            next_track_id += 1
            # Walk the unbranched chain forward from this start.
            node = start
            while True:
                track_of[node] = track_id
                succ = sorted(self.graph.successors(node), key=self._sort_key)
                # Continue the same track only through a non-dividing 1->1 link.
                if len(succ) == 1 and self.graph.in_degree(succ[0]) == 1:
                    node = succ[0]
                    continue
                break
            # Record parent track(s) for this track's start node.
            preds = list(self.graph.predecessors(start))
            if preds:
                parent_tracks[track_id] = sorted(
                    {track_of[p] for p in preds if p in track_of}
                )
        return track_of, parent_tracks

    def _track_starts(self) -> list[int]:
        starts: list[int] = []
        for n in self.graph.nodes:
            preds = list(self.graph.predecessors(n))
            if not preds:
                starts.append(n)  # root
            elif len(preds) == 1 and self.graph.out_degree(preds[0]) >= 2:
                starts.append(n)  # daughter of a division
            elif len(preds) > 1:
                starts.append(n)  # merge target (rare; treated as a fresh track)
        return starts

    def _sort_key(self, node: int) -> tuple[int, int]:
        d = self.graph.nodes[node]
        return d[NODE_FRAME], d[NODE_LABEL]
