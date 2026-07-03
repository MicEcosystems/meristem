"""Reading image stacks and writing a self-describing result bundle.

Kept small and format-light: TIFF/OME-TIFF via ``tifffile`` covers the common case, and the
result bundle is plain files (TIFF masks, ``.npy`` napari tracks, CTC ``res_track.txt``, a JSON
manifest) rather than MiDAP's scattered ``.npz``/CSV/HDF5. Richer readers (ND2/CZI via ``bioio``)
belong in an optional extra, not the core.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .contracts import ImageStack, SegMasks, TrackGraph


def read_image_stack(
    path: str | Path,
    *,
    pixel_size_um: float | None = None,
    frame_interval_s: float | None = None,
    name: str = "fov",
) -> ImageStack:
    """Read a (T, Y, X) TIFF/OME-TIFF into an :class:`ImageStack`.

    A 2D image is promoted to a single-frame stack. Multi-channel data is not handled here yet —
    v1 is single-channel monolayers; channel selection will be a reader option later.
    """
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"expected a 2D or 3D (T, Y, X) image at {path}, got shape {arr.shape}"
        )
    return ImageStack(
        data=arr,
        pixel_size_um=pixel_size_um,
        frame_interval_s=frame_interval_s,
        name=name,
    )


@dataclass
class ResultBundle:
    """Everything a pipeline run produces, plus how to persist it."""

    stack: ImageStack
    masks: SegMasks
    tracks: TrackGraph
    segmenter: str
    tracker: str

    def save(self, out_dir: str | Path, *, save_masks: bool = True, save_tracks: bool = True) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "name": self.stack.name,
            "segmenter": self.segmenter,
            "tracker": self.tracker,
            "n_frames": self.stack.n_frames,
            "shape_yx": list(self.stack.shape_yx),
            "pixel_size_um": self.stack.pixel_size_um,
            "frame_interval_s": self.stack.frame_interval_s,
            "n_detections": self.tracks.n_detections,
            "n_divisions": len(self.tracks.divisions()),
            "files": {},
        }

        if save_masks:
            mask_path = out / f"{self.stack.name}_masks.tif"
            tifffile.imwrite(str(mask_path), self.masks.data)
            manifest["files"]["masks"] = mask_path.name

        if save_tracks:
            data, graph = self.tracks.to_napari_tracks()
            tracks_path = out / f"{self.stack.name}_tracks.npy"
            np.save(tracks_path, data)
            graph_path = out / f"{self.stack.name}_tracks_graph.json"
            with open(graph_path, "w") as fh:
                json.dump({str(k): v for k, v in graph.items()}, fh, indent=2)
            ctc_path = out / f"{self.stack.name}_res_track.txt"
            with open(ctc_path, "w") as fh:
                for row in self.tracks.to_ctc():
                    fh.write(" ".join(str(v) for v in row) + "\n")
            manifest["files"].update(
                tracks=tracks_path.name, tracks_graph=graph_path.name, ctc=ctc_path.name
            )

        manifest_path = out / f"{self.stack.name}_manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest_path
