"""Reusable lineage-construction helpers.

These are backend-agnostic building blocks for turning per-frame instance masks into a
:class:`~meristem.core.contracts.TrackGraph`. The centerpiece, :func:`greedy_overlap_link`, is a
mask-overlap tracker with division detection — a legitimate baseline in its own right and the
engine behind :class:`~meristem.core.tracking.mock.MockTracker`. Real learned trackers replace
the *linking* step but can still reuse :func:`frame_centroids` and the graph-building conventions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..contracts import SegMasks, TrackGraph

Centroid = Tuple[float, float]


def frame_centroids(labels: np.ndarray) -> Dict[int, Centroid]:
    """Return ``{label: (y, x)}`` centroids for every non-zero label in a 2D label image."""
    out: Dict[int, Centroid] = {}
    present = np.unique(labels)
    for lab in present[present != 0]:
        ys, xs = np.nonzero(labels == lab)
        out[int(lab)] = (float(ys.mean()), float(xs.mean()))
    return out


def greedy_overlap_link(masks: SegMasks, min_iou: float = 0.1) -> TrackGraph:
    """Build a lineage by linking cells with maximal mask overlap between consecutive frames.

    For each frame pair, every current-frame cell is linked to the previous-frame cell it overlaps
    most (above ``min_iou``). When two current cells both claim the same parent, that parent is
    treated as having **divided** — both become its daughters. Cells with no qualifying parent
    start new lineages. This is O(cells) per frame via a label-intersection histogram, not a full
    IoU matrix, so it scales to dense monolayers.
    """
    graph = TrackGraph()
    # Assign a unique node id per (frame, label). Encode as f * BIG + label for stability.
    big = int(masks.data.max()) + 1
    if big < 1:
        big = 1

    def node_id(frame: int, label: int) -> int:
        return frame * (big + 1) + label

    # Register every detection as a node.
    centroids_by_frame: list[Dict[int, Centroid]] = []
    for t in range(masks.n_frames):
        cents = frame_centroids(masks.data[t])
        centroids_by_frame.append(cents)
        for label, centroid in cents.items():
            graph.add_detection(node_id(t, label), frame=t, label=label, centroid=centroid)

    # Link consecutive frames by best overlap.
    for t in range(1, masks.n_frames):
        prev = masks.data[t - 1]
        curr = masks.data[t]
        for label in centroids_by_frame[t]:
            parent = _best_overlap_parent(prev, curr, label, min_iou)
            if parent is not None:
                graph.link(node_id(t - 1, parent), node_id(t, label))
    return graph


def _best_overlap_parent(
    prev: np.ndarray, curr: np.ndarray, label: int, min_iou: int | float
) -> int | None:
    """Return the previous-frame label overlapping ``curr == label`` the most (by IoU), or None."""
    region = curr == label
    overlap_labels = prev[region]
    overlap_labels = overlap_labels[overlap_labels != 0]
    if overlap_labels.size == 0:
        return None
    values, counts = np.unique(overlap_labels, return_counts=True)
    best_idx = int(np.argmax(counts))
    best_label = int(values[best_idx])
    intersection = int(counts[best_idx])
    union = int(region.sum()) + int((prev == best_label).sum()) - intersection
    iou = intersection / union if union else 0.0
    return best_label if iou >= min_iou else None
