"""Segmentation post-processing — size filtering (MiDAP parity).

MiDAP's ``postprocess_seg`` drops spurious regions that are too small to be cells: per frame it
keeps only components whose area exceeds ``mean(area) * 0.01`` (1% of the mean cell size). We do the
same, with two differences suited to our contracts: we work on the *existing instance labels*
(computing areas by label count, so touching cells are never merged the way re-running connected
components would), and we optionally also drop oversized blobs — useful when a segmenter fuses a
clump of cells into one giant label.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .contracts import SegMasks


def filter_by_size(
    masks: SegMasks,
    *,
    min_size_frac: float = 0.01,
    max_size_frac: Optional[float] = None,
    min_size_px: Optional[int] = None,
) -> SegMasks:
    """Remove instance labels whose area is out of range, per frame.

    Parameters
    ----------
    min_size_frac:
        Drop labels smaller than ``min_size_frac * mean_area`` (MiDAP uses 0.01). Set 0 to disable.
    max_size_frac:
        If given, also drop labels larger than ``max_size_frac * mean_area`` (e.g. 5.0 removes
        merged clumps). MiDAP has no upper bound; ``None`` keeps that behavior.
    min_size_px:
        Optional absolute floor in pixels, applied in addition to ``min_size_frac``.

    Surviving labels keep their original ids; the result is a new :class:`SegMasks`.
    """
    out = np.zeros_like(masks.data)
    for t in range(masks.n_frames):
        labels = masks.data[t]
        sizes = np.bincount(labels.ravel())  # sizes[i] = pixel count of label i (0 = background)
        if sizes.size <= 1:
            continue  # only background in this frame
        present = sizes[1:][sizes[1:] > 0]
        if present.size == 0:
            continue
        mean_area = float(present.mean())

        low = mean_area * min_size_frac
        if min_size_px is not None:
            low = max(low, float(min_size_px))
        keep = sizes > low
        if max_size_frac is not None:
            keep &= sizes < mean_area * max_size_frac
        keep[0] = False  # background is never a cell

        out[t] = labels * keep[labels]
    return SegMasks(data=out, source=f"{masks.source}+sizefilter")
