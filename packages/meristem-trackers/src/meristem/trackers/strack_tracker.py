"""S-track: a native port of MiDAP's geometric bacterial tracker.

MiDAP's S-track links segmented cells frame-to-frame with a greedy strategy that prefers pixel
overlap, falls back to centroid distance (gated by ``max_dist``), and — crucially for bacteria —
only accepts a cell division when the two daughters separate roughly along the *mother's long
axis* (within ``max_angle``). A mother may gain at most two daughters.

This is a from-scratch reimplementation against Meristem's contracts: it works in-memory on a
:class:`~meristem.core.contracts.SegMasks` array (no TIF round-trips), depends only on NumPy (the
original used OpenCV + scikit-image), and emits a :class:`~meristem.core.contracts.TrackGraph`
with divisions as first-class edges instead of MiDAP's per-frame CSV tables. Keeping S-track
available preserves continuity for users who rely on it while dropping its file-based baggage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from meristem.core.contracts import ImageStack, SegMasks, TrackGraph
from meristem.core.tracking.base import TrackerBackend, TrackerParams


class STrackParams(TrackerParams):
    """Parameters for the S-track backend."""

    max_dist: float = 30.0  # max centroid displacement (pixels) to link a cell across frames
    max_angle: float = 30.0  # max deviation (degrees) of the division axis from the mother's axis
    min_overlap: float = 0.05  # min fraction of a cell covered by a candidate mother to count as overlap


@dataclass
class _Region:
    centroid: tuple[float, float]  # (y, x)
    orientation: float  # radians, major-axis angle from the x-axis
    area: int


class STrackTracker(TrackerBackend):
    """Overlap-first greedy tracker with bacterial division validation (native S-track port)."""

    name = "strack"
    Params = STrackParams

    def load(self, params: STrackParams) -> None:  # type: ignore[override]
        self._params = params

    def track(self, stack: ImageStack, masks: SegMasks) -> TrackGraph:
        p: STrackParams = getattr(self, "_params", STrackParams())
        data = masks.data
        big = int(data.max()) + 1
        if big < 1:
            big = 1

        def node_id(frame: int, label: int) -> int:
            return frame * (big + 1) + label

        graph = TrackGraph()
        regions_by_frame: list[Dict[int, _Region]] = []
        for t in range(masks.n_frames):
            regions = _frame_regions(data[t])
            regions_by_frame.append(regions)
            for label, reg in regions.items():
                graph.add_detection(node_id(t, label), frame=t, label=label, centroid=reg.centroid)

        for t in range(1, masks.n_frames):
            prev_img, curr_img = data[t - 1], data[t]
            prev_regions, curr_regions = regions_by_frame[t - 1], regions_by_frame[t]
            assignments = _match_frame(
                prev_img, curr_img, prev_regions, curr_regions, p
            )
            for mother_label, daughter_label in assignments:
                graph.link(node_id(t - 1, mother_label), node_id(t, daughter_label))

        return graph


def _frame_regions(labels: np.ndarray) -> Dict[int, _Region]:
    """Compute centroid, long-axis orientation, and area for every label in a frame."""
    out: Dict[int, _Region] = {}
    present = np.unique(labels)
    for lab in present[present != 0]:
        ys, xs = np.nonzero(labels == lab)
        cy, cx = float(ys.mean()), float(xs.mean())
        x = xs - cx
        y = ys - cy
        mu20 = float(np.mean(x * x))
        mu02 = float(np.mean(y * y))
        mu11 = float(np.mean(x * y))
        # Major-axis angle from the x-axis (second-moment orientation), matching regionprops.
        orientation = 0.5 * math.atan2(2.0 * mu11, (mu20 - mu02))
        out[int(lab)] = _Region(centroid=(cy, cx), orientation=orientation, area=int(ys.size))
    return out


def _match_frame(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    prev_regions: Dict[int, _Region],
    curr_regions: Dict[int, _Region],
    p: STrackParams,
) -> list[tuple[int, int]]:
    """Return (mother_label, daughter_label) links from frame t-1 to frame t."""
    # Build candidate (overlap_frac, dist, curr_label, prev_label) tuples.
    candidates: list[tuple[float, float, int, int]] = []
    for cl, cinfo in curr_regions.items():
        region = curr_img == cl
        # Overlap: how much of this current cell is covered by each previous-frame label.
        prev_vals = prev_img[region]
        prev_vals = prev_vals[prev_vals != 0]
        overlaps: Dict[int, float] = {}
        if prev_vals.size:
            vals, counts = np.unique(prev_vals, return_counts=True)
            for v, cnt in zip(vals, counts):
                overlaps[int(v)] = cnt / cinfo.area
        for pl, pinfo in prev_regions.items():
            frac = overlaps.get(pl, 0.0)
            dist = math.dist(cinfo.centroid, pinfo.centroid)
            if frac >= p.min_overlap or dist <= p.max_dist:
                candidates.append((frac, dist, cl, pl))

    # Greedy: strongest overlap first, then nearest.
    candidates.sort(key=lambda c: (-c[0], c[1]))

    assigned_curr: set[int] = set()
    daughters: Dict[int, list[int]] = {}
    links: list[tuple[int, int]] = []
    for frac, dist, cl, pl in candidates:
        if cl in assigned_curr:
            continue
        if frac < p.min_overlap and dist > p.max_dist:
            continue
        existing = daughters.get(pl, [])
        if len(existing) >= 2:
            continue  # a mother keeps at most two daughters
        if len(existing) == 1:
            # Second daughter => a division; accept only if it splits along the mother's long axis.
            if not _valid_division(
                prev_regions[pl], curr_regions[existing[0]], curr_regions[cl], p.max_angle
            ):
                continue
        daughters.setdefault(pl, []).append(cl)
        assigned_curr.add(cl)
        links.append((pl, cl))
    return links


def _valid_division(mother: _Region, d1: _Region, d2: _Region, max_angle_deg: float) -> bool:
    """True if the axis between the two daughters aligns with the mother's long axis.

    Bacteria elongate and split along their long axis, so a genuine division has the daughters
    displaced roughly parallel to the mother's orientation. Both are axes (180-degree symmetric),
    so the angular difference is reduced to [0, 90] degrees before comparison.
    """
    dy = d2.centroid[0] - d1.centroid[0]
    dx = d2.centroid[1] - d1.centroid[1]
    if dy == 0 and dx == 0:
        return False
    division_axis = math.atan2(dy, dx)
    diff = abs(division_axis - mother.orientation) % math.pi  # axes are pi-periodic
    diff = min(diff, math.pi - diff)  # fold to [0, pi/2]
    return math.degrees(diff) <= max_angle_deg
