"""Drift registration — correct stage/XY drift.

Long microfluidic time-lapses drift, so a fixed crop rectangle slowly loses the cells it started
on. This module estimates per-frame translation by FFT phase cross-correlation (pure NumPy, no
scikit-image), estimated on one channel (e.g. PH) and applied identically to every channel so the
fluorescence stays registered to the masks.

This follows MiDAP's approach (``skimage.registration.phase_cross_correlation`` to the first frame,
translation only): drift is corrected by **moving the crop window per frame**
(:func:`crop_with_drift`) so the box tracks the cells — no image resampling or zero-fill artifacts.
For the (less common) no-crop case, :func:`apply_shifts` resamples the whole frame instead.
Rotation/scale drift is out of scope for v1.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Reference = Literal["previous", "first"]


def estimate_drift(images: np.ndarray, reference: Reference = "previous") -> np.ndarray:
    """Estimate per-frame drift of a ``(T, Y, X)`` stack.

    Returns an ``(T, 2)`` array of ``(dy, dx)`` giving each frame's displacement **relative to
    frame 0** (frame 0 is ``(0, 0)``). ``reference="previous"`` accumulates frame-to-frame shifts
    (robust to gradual drift over long movies); ``"first"`` correlates every frame directly against
    frame 0 (robust when total drift is small).
    """
    if images.ndim != 3:
        raise ValueError(f"estimate_drift expects a (T, Y, X) stack, got shape {images.shape}")
    n = images.shape[0]
    shifts = np.zeros((n, 2), dtype=float)
    if n <= 1:
        return shifts

    if reference == "first":
        ref_fft = np.fft.fft2(images[0].astype(np.float32))
        for t in range(1, n):
            shifts[t] = _phase_shift(ref_fft, images[t])
    elif reference == "previous":
        for t in range(1, n):
            rel = _phase_shift(np.fft.fft2(images[t - 1].astype(np.float32)), images[t])
            shifts[t] = shifts[t - 1] + rel
    else:
        raise ValueError(f"reference must be 'previous' or 'first', got {reference!r}")
    return shifts


def crop_with_drift(
    images: np.ndarray, y: int, x: int, height: int, width: int, shifts: np.ndarray
) -> np.ndarray:
    """Crop a ``(T, Y, X)`` stack with the window following the drift (MiDAP's cutout approach).

    For frame ``t`` the crop origin is offset by that frame's drift, so the fixed-size ``(height,
    width)`` box tracks the same cells across the movie without resampling the image. Regions that
    fall outside the frame (only near the image border for large drift) are zero-filled to keep the
    output size constant.
    """
    n, h, w = images.shape
    out = np.zeros((n, height, width), dtype=images.dtype)
    for t in range(n):
        oy = y + int(round(shifts[t, 0]))
        ox = x + int(round(shifts[t, 1]))
        sy0, sy1 = max(0, oy), min(h, oy + height)
        sx0, sx1 = max(0, ox), min(w, ox + width)
        if sy1 <= sy0 or sx1 <= sx0:
            continue  # window entirely outside the frame
        out[t, sy0 - oy : sy1 - oy, sx0 - ox : sx1 - ox] = images[t, sy0:sy1, sx0:sx1]
    return out


def apply_shifts(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Align a ``(T, Y, X)`` stack by undoing each frame's drift (integer-pixel, zero-filled).

    Used for the no-crop case; when a crop is present, prefer :func:`crop_with_drift`, which moves
    the crop window instead of resampling the frame.
    """
    if shifts.shape[0] != images.shape[0]:
        raise ValueError(
            f"shifts ({shifts.shape[0]}) and images ({images.shape[0]}) frame counts differ"
        )
    out = np.zeros_like(images)
    for t in range(images.shape[0]):
        dy, dx = int(round(shifts[t, 0])), int(round(shifts[t, 1]))
        out[t] = _shift_frame(images[t], -dy, -dx)  # move the frame back to the reference
    return out


def _phase_shift(ref_fft: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Integer-pixel shift of ``img`` relative to the reference (whose FFT is ``ref_fft``)."""
    img_fft = np.fft.fft2(img.astype(np.float32))
    cross = ref_fft * np.conj(img_fft)
    cross /= np.abs(cross) + 1e-8  # phase-only (whitened) cross-power spectrum
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    shift = np.array(peak, dtype=float)
    # The correlation peak is periodic; map indices in the upper half to negative shifts.
    for i, dim in enumerate(corr.shape):
        if shift[i] > dim // 2:
            shift[i] -= dim
    # The peak locates how to shift `img` back onto `ref`; negate to return img's displacement
    # (its drift) relative to `ref`, which is what estimate_drift accumulates.
    return -shift


def _shift_frame(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a 2D image by (dy, dx) with zero fill (no wraparound)."""
    h, w = img.shape
    out = np.zeros_like(img)
    y_src0, y_src1 = max(0, -dy), min(h, h - dy)
    x_src0, x_src1 = max(0, -dx), min(w, w - dx)
    y_dst0, x_dst0 = max(0, dy), max(0, dx)
    out[y_dst0 : y_dst0 + (y_src1 - y_src0), x_dst0 : x_dst0 + (x_src1 - x_src0)] = img[
        y_src0:y_src1, x_src0:x_src1
    ]
    return out
