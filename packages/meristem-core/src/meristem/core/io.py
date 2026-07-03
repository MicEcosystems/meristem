"""Reading image stacks and writing a self-describing result bundle.

Kept small and format-light: TIFF/OME-TIFF via ``tifffile`` covers the common case, and the
result bundle is plain files (TIFF masks, ``.npy`` napari tracks, CTC ``res_track.txt``, a JSON
manifest) rather than MiDAP's scattered ``.npz``/CSV/HDF5. Richer readers (ND2/CZI via ``bioio``)
belong in an optional extra, not the core.

A run may involve several channels (PH/GFP/RFP), each segmented — and optionally tracked —
independently. Results are therefore a list of :class:`ChannelResult`, saved with channel-prefixed
filenames under one manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import tifffile

from .contracts import ImageStack, SegMasks, TrackGraph


def read_image_stack(
    path: str | Path,
    *,
    pixel_size_um: float | None = None,
    frame_interval_s: float | None = None,
    name: str = "fov",
    max_frames: int | None = None,
) -> ImageStack:
    """Read a single-channel (T, Y, X) TIFF/OME-TIFF into an :class:`ImageStack`.

    A 2D image is promoted to a single-frame stack. ``max_frames`` reads only the first N frames
    (via ``tifffile`` page keys) — useful for quick runs on very large stacks without loading the
    whole file. Each imaging channel is read separately; the pipeline handles them per-channel.
    """
    key = range(max_frames) if max_frames else None
    arr = tifffile.imread(str(path), key=key)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected a 2D or 3D (T, Y, X) image at {path}, got shape {arr.shape}")
    return ImageStack(
        data=arr,
        pixel_size_um=pixel_size_um,
        frame_interval_s=frame_interval_s,
        name=name,
    )


@dataclass
class ChannelResult:
    """Segmentation (and optional tracking) outcome for a single channel."""

    name: str
    stack: ImageStack
    masks: SegMasks
    tracks: Optional[TrackGraph] = None  # None if this channel was segmented but not tracked

    @property
    def tracked(self) -> bool:
        return self.tracks is not None


@dataclass
class ResultBundle:
    """Everything a pipeline run produces across all channels, plus how to persist it."""

    channels: List[ChannelResult]
    segmenter: str
    tracker: str

    def channel(self, name: str) -> ChannelResult:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(f"no channel named {name!r}; have {[c.name for c in self.channels]}")

    def save(self, out_dir: str | Path, *, save_masks: bool = True, save_tracks: bool = True) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "segmenter": self.segmenter,
            "tracker": self.tracker,
            "channels": [
                self._save_channel(ch, out, save_masks=save_masks, save_tracks=save_tracks)
                for ch in self.channels
            ],
        }
        manifest_path = out / "manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest_path

    def _save_channel(
        self, ch: ChannelResult, out: Path, *, save_masks: bool, save_tracks: bool
    ) -> dict:
        prefix = ch.name
        entry: dict = {
            "name": ch.name,
            "n_frames": ch.stack.n_frames,
            "shape_yx": list(ch.stack.shape_yx),
            "pixel_size_um": ch.stack.pixel_size_um,
            "frame_interval_s": ch.stack.frame_interval_s,
            "tracked": ch.tracked,
            "files": {},
        }
        if save_masks:
            mask_path = out / f"{prefix}_masks.tif"
            tifffile.imwrite(str(mask_path), ch.masks.data)
            entry["files"]["masks"] = mask_path.name

        if save_tracks and ch.tracks is not None:
            data, graph = ch.tracks.to_napari_tracks()
            tracks_path = out / f"{prefix}_tracks.npy"
            np.save(tracks_path, data)
            graph_path = out / f"{prefix}_tracks_graph.json"
            with open(graph_path, "w") as fh:
                json.dump({str(k): v for k, v in graph.items()}, fh, indent=2)
            ctc_path = out / f"{prefix}_res_track.txt"
            with open(ctc_path, "w") as fh:
                for row in ch.tracks.to_ctc():
                    fh.write(" ".join(str(v) for v in row) + "\n")
            entry.update(
                n_detections=ch.tracks.n_detections,
                n_divisions=len(ch.tracks.divisions()),
            )
            entry["files"].update(
                tracks=tracks_path.name, tracks_graph=graph_path.name, ctc=ctc_path.name
            )
        return entry
