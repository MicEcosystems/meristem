"""A dependency-free segmentation backend for tests, demos, and pipeline wiring.

:class:`MockSegmenter` "segments" by thresholding the normalized image and labeling connected
components with a tiny pure-numpy 2-pass algorithm (no scipy/skimage). It lets the whole pipeline
— crop, segment, track, export — run and be tested with zero ML dependencies, which is exactly
what MiDAP lacked. It is registered as the ``mock`` segmenter via the package entry points.
"""

from __future__ import annotations

import numpy as np

from ..contracts import ImageStack, SegMasks
from .base import SegmenterBackend, SegmenterParams


class MockSegmenterParams(SegmenterParams):
    threshold: float = 0.5  # relative threshold on the per-stack normalized image, in [0, 1]
    min_size: int = 1  # drop components smaller than this many pixels


class MockSegmenter(SegmenterBackend):
    """Threshold + connected-components. Not a real model; a stand-in with the real contract."""

    name = "mock"
    Params = MockSegmenterParams

    def load(self, params: MockSegmenterParams) -> None:  # type: ignore[override]
        self._params = params

    def segment(self, stack: ImageStack) -> SegMasks:
        params: MockSegmenterParams = getattr(self, "_params", MockSegmenterParams())
        norm = stack.normalized()
        out = np.zeros(stack.data.shape, dtype=np.int32)
        for t in range(stack.n_frames):
            out[t] = _label_components(norm[t] >= params.threshold, params.min_size)
        return SegMasks(data=out, source=self.name)


def _label_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Label 4-connected foreground components. Pure numpy union-find, no external deps."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]  # parent[0] unused; union-find over provisional labels

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    next_label = 1
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up and left:
                labels[y, x] = min(up, left)
                union(up, left)
            elif up:
                labels[y, x] = up
            elif left:
                labels[y, x] = left
            else:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1

    # Resolve equivalences and compact labels to a contiguous 1..K range.
    remap: dict[int, int] = {}
    out = np.zeros_like(labels)
    counts: dict[int, int] = {}
    for y in range(h):
        for x in range(w):
            lab = labels[y, x]
            if not lab:
                continue
            root = find(lab)
            counts[root] = counts.get(root, 0) + 1
    keep = {root for root, c in counts.items() if c >= min_size}
    for y in range(h):
        for x in range(w):
            lab = labels[y, x]
            if not lab:
                continue
            root = find(lab)
            if root not in keep:
                continue
            if root not in remap:
                remap[root] = len(remap) + 1
            out[y, x] = remap[root]
    return out
