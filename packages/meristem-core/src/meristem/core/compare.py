"""Compare mode: run several models on the *same* input and tabulate them.

MiDAP's GUI lets you preview a few segmentation models and pick the best before committing. Because
every Meristem backend shares one contract, comparison here is just looping registry names over
identical inputs — no per-model glue. Two independent axes:

- **Segmenter comparison** — segment the chosen channel with each backend; report cells/frame,
  total cells, median cell area, and wall-clock time.
- **Tracker comparison** — take one shared segmentation (the ``track_on`` backend, or the first
  segmenter) and run each tracker on those identical masks; report detections, divisions, tracks,
  and time. Comparing trackers on the *same* masks is what makes it a fair fight.

Ground-truth metrics (CTC TRA/DET via ``traccuracy``) are a natural extension and slot in wherever a
ground-truth lineage is provided; this v1 focuses on the descriptive, no-ground-truth comparison
that mirrors MiDAP's eyeball-and-choose workflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .config import BackendConfig, InputConfig, ROIConfig
from .contracts import ImageStack, SegMasks, TrackGraph
from .io import read_image_stack
from .pipeline import segment as _segment
from .pipeline import track as _track


class CompareSpec(BaseModel):
    """What to compare, and on what input."""

    model_config = {"extra": "forbid"}

    input: InputConfig
    crop: Optional[ROIConfig] = None
    segmenters: List[BackendConfig] = Field(default_factory=list)
    trackers: List[BackendConfig] = Field(default_factory=list)
    channel: Optional[str] = None  # multichannel: which channel to compare on (default: first)
    track_on: Optional[str] = None  # segmenter whose masks feed the tracker comparison (default: first)

    @model_validator(mode="after")
    def _check(self) -> "CompareSpec":
        if not self.segmenters and not self.trackers:
            raise ValueError("compare requires at least one segmenter or tracker to compare")
        if self.trackers and not self.segmenters:
            raise ValueError("comparing trackers needs at least one segmenter to produce masks")
        names = [s.name for s in self.segmenters]
        if self.track_on and self.track_on not in names:
            raise ValueError(f"track_on={self.track_on!r} is not among the compared segmenters {names}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CompareSpec":
        import yaml

        with open(path, "r") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})


@dataclass
class SegmenterComparison:
    name: str
    cells_per_frame: List[int]
    total_cells: int
    median_area_px: float
    median_area_um2: Optional[float]
    seconds: float
    masks: SegMasks = field(repr=False)


@dataclass
class TrackerComparison:
    name: str
    n_detections: int
    n_divisions: int
    n_tracks: int
    seconds: float
    tracks: TrackGraph = field(repr=False)


@dataclass
class ComparisonReport:
    channel: str
    n_frames: int
    shape_yx: tuple
    segmenters: List[SegmenterComparison]
    trackers: List[TrackerComparison]
    shared_segmenter: Optional[str]  # which segmentation the trackers were compared on


def run_comparison(spec: CompareSpec) -> ComparisonReport:
    """Run the comparison described by ``spec`` and return a :class:`ComparisonReport`."""
    stack = _load_channel(spec)

    seg_results: List[SegmenterComparison] = []
    masks_by_name: dict[str, SegMasks] = {}
    for bc in spec.segmenters:
        t0 = time.perf_counter()
        masks = _segment(stack, bc)
        seconds = time.perf_counter() - t0
        masks_by_name[bc.name] = masks
        seg_results.append(_summarize_segmentation(bc.name, masks, seconds, stack.pixel_size_um))

    trk_results: List[TrackerComparison] = []
    shared: Optional[str] = None
    if spec.trackers:
        shared = spec.track_on or spec.segmenters[0].name
        masks = masks_by_name[shared]
        for bc in spec.trackers:
            t0 = time.perf_counter()
            tg = _track(stack, masks, bc)
            seconds = time.perf_counter() - t0
            trk_results.append(_summarize_tracking(bc.name, tg, seconds))

    return ComparisonReport(
        channel=stack.name,
        n_frames=stack.n_frames,
        shape_yx=stack.shape_yx,
        segmenters=seg_results,
        trackers=trk_results,
        shared_segmenter=shared,
    )


def _load_channel(spec: CompareSpec) -> ImageStack:
    channels = spec.input.resolved_channels()
    if spec.channel is not None:
        chosen = next((c for c in channels if c.name == spec.channel), None)
        if chosen is None:
            raise ValueError(f"channel {spec.channel!r} not in input {[c.name for c in channels]}")
    else:
        chosen = next((c for c in channels if c.segment), channels[0])
    stack = read_image_stack(
        chosen.path,
        pixel_size_um=spec.input.pixel_size_um,
        frame_interval_s=spec.input.frame_interval_s,
        name=chosen.name,
        max_frames=spec.input.max_frames,
    )
    if spec.crop is not None:
        stack = stack.crop(spec.crop.to_roi())
    return stack


def _summarize_segmentation(
    name: str, masks: SegMasks, seconds: float, pixel_size_um: Optional[float]
) -> SegmenterComparison:
    cells = masks.n_cells_per_frame()
    areas = [int((masks.data[t] == lab).sum()) for t in range(masks.n_frames) for lab in masks.labels_in_frame(t)]
    median_px = float(np.median(areas)) if areas else 0.0
    median_um2 = median_px * pixel_size_um**2 if pixel_size_um else None
    return SegmenterComparison(
        name=name,
        cells_per_frame=cells,
        total_cells=int(sum(cells)),
        median_area_px=median_px,
        median_area_um2=median_um2,
        seconds=seconds,
        masks=masks,
    )


def _summarize_tracking(name: str, tg: TrackGraph, seconds: float) -> TrackerComparison:
    data, _ = tg.to_napari_tracks()
    n_tracks = int(np.unique(data[:, 0]).size) if len(data) else 0
    return TrackerComparison(
        name=name,
        n_detections=tg.n_detections,
        n_divisions=len(tg.divisions()),
        n_tracks=n_tracks,
        seconds=seconds,
        tracks=tg,
    )


def format_report(report: ComparisonReport) -> str:
    """Render a report as an aligned text table for the CLI."""
    h, w = report.shape_yx
    lines = [f"Comparison on channel '{report.channel}'  ({report.n_frames} frames, {h}x{w})", ""]

    if report.segmenters:
        lines.append("Segmentation:")
        lines.append(f"  {'backend':16s} {'total':>6s} {'median area':>18s} {'time(s)':>8s}")
        for s in report.segmenters:
            area = f"{s.median_area_px:.0f}px"
            if s.median_area_um2 is not None:
                area += f" ({s.median_area_um2:.2f}um2)"
            lines.append(f"  {s.name:16s} {s.total_cells:>6d} {area:>18s} {s.seconds:>8.2f}")
        lines.append("")

    if report.trackers:
        lines.append(f"Tracking (on '{report.shared_segmenter}' masks):")
        lines.append(
            f"  {'backend':16s} {'detections':>10s} {'divisions':>9s} {'tracks':>6s} {'time(s)':>8s}"
        )
        for t in report.trackers:
            lines.append(
                f"  {t.name:16s} {t.n_detections:>10d} {t.n_divisions:>9d} "
                f"{t.n_tracks:>6d} {t.seconds:>8.2f}"
            )
    return "\n".join(lines)
