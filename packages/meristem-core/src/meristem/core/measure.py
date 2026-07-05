"""Per-cell intensity measurement — read fluorescence channels through a segmentation.

The canonical bacterial-reporter workflow: segment cells on phase contrast (PH), then quantify
GFP/RFP *per cell* using those PH masks, and tie each measurement to the cell's track so you get a
per-lineage intensity time-series. This module does the measurement half; the pipeline wires it to
the PH masks + tracks.

Measurement is keyed by the segmentation's per-frame instance labels. Each cell's ``track_id`` is
resolved by nearest-centroid match to the tracking result, which is tracker-agnostic: the tracks
were computed from these same masks, so centroids line up regardless of whether the tracker relabels
(Trackastra) or keeps the mask labels (strack).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .contracts import SegMasks, TrackGraph

UNTRACKED = -1


@dataclass
class CellMeasurement:
    frame: int
    label: int  # instance label in the segmentation for this frame
    track_id: int  # lineage track id, or UNTRACKED (-1) if no track matched
    area_px: int
    centroid_y: float
    centroid_x: float
    # channel name -> {"mean", "total", "median"} intensity over the cell's pixels
    intensities: Dict[str, Dict[str, float]]


@dataclass
class MeasurementTable:
    rows: List[CellMeasurement]
    channels: List[str]  # measured channel names, in column order
    pixel_size_um: Optional[float] = None

    def to_records(self) -> List[dict]:
        recs = []
        for r in self.rows:
            rec = {
                "frame": r.frame,
                "label": r.label,
                "track_id": r.track_id,
                "area_px": r.area_px,
                "area_um2": (r.area_px * self.pixel_size_um**2) if self.pixel_size_um else "",
                "centroid_y": round(r.centroid_y, 3),
                "centroid_x": round(r.centroid_x, 3),
            }
            for ch in self.channels:
                stats = r.intensities.get(ch, {})
                rec[f"{ch}_mean"] = round(stats.get("mean", 0.0), 3)
                rec[f"{ch}_total"] = round(stats.get("total", 0.0), 3)
                rec[f"{ch}_median"] = round(stats.get("median", 0.0), 3)
            recs.append(rec)
        return recs

    def to_csv(self, path: str | Path) -> None:
        records = self.to_records()
        fieldnames = list(records[0].keys()) if records else ["frame", "label", "track_id"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def measure_intensities(
    masks: SegMasks,
    channel_images: Dict[str, np.ndarray],
    *,
    tracks: Optional[TrackGraph] = None,
    pixel_size_um: Optional[float] = None,
    match_tolerance_px: float = 3.0,
) -> MeasurementTable:
    """Measure per-cell intensity of each channel through ``masks``.

    Parameters
    ----------
    masks:
        The segmentation (from the ``measure_on`` channel, e.g. PH) that defines the cells.
    channel_images:
        ``{channel_name: (T, Y, X) array}`` for each channel to measure (e.g. GFP, RFP). Must
        match the masks' shape.
    tracks:
        Optional lineage for the same masks; used to attach a ``track_id`` per cell.
    pixel_size_um:
        For reporting cell area in physical units.
    """
    for name, img in channel_images.items():
        if img.shape != masks.data.shape:
            raise ValueError(
                f"channel {name!r} shape {img.shape} != masks shape {masks.data.shape}"
            )

    frame_tracks = _tracks_by_frame(tracks) if tracks is not None else {}
    channels = list(channel_images)
    rows: List[CellMeasurement] = []

    for t in range(masks.n_frames):
        frame_labels = masks.labels_in_frame(t)
        track_points = frame_tracks.get(t, [])
        for lab in frame_labels:
            ys, xs = np.nonzero(masks.data[t] == lab)
            cy, cx = float(ys.mean()), float(xs.mean())
            intensities: Dict[str, Dict[str, float]] = {}
            for name in channels:
                pix = channel_images[name][t][ys, xs].astype(np.float64)
                intensities[name] = {
                    "mean": float(pix.mean()),
                    "total": float(pix.sum()),
                    "median": float(np.median(pix)),
                }
            rows.append(
                CellMeasurement(
                    frame=t,
                    label=int(lab),
                    track_id=_nearest_track(cy, cx, track_points, match_tolerance_px),
                    area_px=int(ys.size),
                    centroid_y=cy,
                    centroid_x=cx,
                    intensities=intensities,
                )
            )

    return MeasurementTable(rows=rows, channels=channels, pixel_size_um=pixel_size_um)


@dataclass
class TrackSummary:
    """One cell lineage collapsed to a single row."""

    track_id: int
    parent: int  # parent track id (0 = founder / no parent, CTC convention)
    start_frame: int
    end_frame: int
    n_frames: int  # number of frames the track is observed
    n_daughters: int
    divides: bool
    area_first_px: int
    area_last_px: int
    area_mean_px: float
    displacement_px: float  # centroid distance from first to last frame
    growth_rate_per_hr: Optional[float]  # specific growth rate ln(A_last/A_first)/hours
    channel_means: Dict[str, float]  # per measured channel, mean of the per-frame mean intensities


@dataclass
class TrackSummaryTable:
    rows: List[TrackSummary]
    channels: List[str]
    pixel_size_um: Optional[float] = None

    def to_records(self) -> List[dict]:
        recs = []
        um2 = (self.pixel_size_um**2) if self.pixel_size_um else None
        for r in sorted(self.rows, key=lambda s: s.track_id):
            rec = {
                "track_id": r.track_id,
                "parent": r.parent,
                "start_frame": r.start_frame,
                "end_frame": r.end_frame,
                "n_frames": r.n_frames,
                "n_daughters": r.n_daughters,
                "divides": int(r.divides),
                "area_first_px": r.area_first_px,
                "area_last_px": r.area_last_px,
                "area_mean_px": round(r.area_mean_px, 2),
                "area_mean_um2": round(r.area_mean_px * um2, 4) if um2 else "",
                "displacement_px": round(r.displacement_px, 3),
                "growth_rate_per_hr": (
                    round(r.growth_rate_per_hr, 5) if r.growth_rate_per_hr is not None else ""
                ),
            }
            for ch in self.channels:
                rec[f"{ch}_mean"] = round(r.channel_means.get(ch, 0.0), 3)
            recs.append(rec)
        return recs

    def to_csv(self, path: str | Path) -> None:
        records = self.to_records()
        fieldnames = list(records[0].keys()) if records else ["track_id", "parent"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def summarize_tracks(
    measurements: MeasurementTable,
    tracks: TrackGraph,
    *,
    frame_interval_s: Optional[float] = None,
) -> TrackSummaryTable:
    """Collapse long-format per-cell measurements into one row per track (lineage).

    Aggregates area/intensity over each track and attaches lineage from ``tracks`` (parent,
    daughter count, division flag). If ``frame_interval_s`` is known, adds a specific growth rate
    ``ln(area_last / area_first) / hours`` per track.
    """
    parent_of, daughters_of = _lineage_maps(tracks)

    by_track: Dict[int, list] = defaultdict(list)
    for r in measurements.rows:
        if r.track_id != UNTRACKED:
            by_track[r.track_id].append(r)

    rows: List[TrackSummary] = []
    for tid, cell_rows in by_track.items():
        cell_rows.sort(key=lambda r: r.frame)
        first, last = cell_rows[0], cell_rows[-1]
        areas = [r.area_px for r in cell_rows]
        span_frames = last.frame - first.frame
        growth = _growth_rate(first.area_px, last.area_px, span_frames, frame_interval_s)
        channel_means = {
            ch: float(np.mean([r.intensities[ch]["mean"] for r in cell_rows]))
            for ch in measurements.channels
        }
        n_daughters = len(daughters_of.get(tid, []))
        rows.append(
            TrackSummary(
                track_id=tid,
                parent=parent_of.get(tid, 0),
                start_frame=first.frame,
                end_frame=last.frame,
                n_frames=len(cell_rows),
                n_daughters=n_daughters,
                divides=n_daughters >= 2,
                area_first_px=first.area_px,
                area_last_px=last.area_px,
                area_mean_px=float(np.mean(areas)),
                displacement_px=math.dist(
                    (first.centroid_y, first.centroid_x), (last.centroid_y, last.centroid_x)
                ),
                growth_rate_per_hr=growth,
                channel_means=channel_means,
            )
        )
    return TrackSummaryTable(
        rows=rows, channels=list(measurements.channels), pixel_size_um=measurements.pixel_size_um
    )


def _lineage_maps(tracks: TrackGraph):
    """Return (parent_of_track, daughters_of_track) from the CTC lineage rows."""
    parent_of: Dict[int, int] = {}
    daughters_of: Dict[int, list] = defaultdict(list)
    for track_id, _start, _end, parent in tracks.to_ctc():
        parent_of[track_id] = parent
        if parent:
            daughters_of[parent].append(track_id)
    return parent_of, daughters_of


def _growth_rate(
    area_first: int, area_last: int, span_frames: int, frame_interval_s: Optional[float]
) -> Optional[float]:
    if not frame_interval_s or span_frames <= 0 or area_first <= 0 or area_last <= 0:
        return None
    hours = span_frames * frame_interval_s / 3600.0
    if hours <= 0:
        return None
    return math.log(area_last / area_first) / hours


def _tracks_by_frame(tracks: TrackGraph) -> Dict[int, list]:
    """Group napari-tracks rows [track_id, t, y, x] by frame for centroid lookup."""
    data, _ = tracks.to_napari_tracks()
    by_frame: Dict[int, list] = {}
    for track_id, t, y, x in data:
        by_frame.setdefault(int(t), []).append((int(track_id), float(y), float(x)))
    return by_frame


def _nearest_track(cy: float, cx: float, points: list, tolerance: float) -> int:
    """Return the track id whose centroid is nearest (within tolerance) to (cy, cx)."""
    best_id, best_d2 = UNTRACKED, tolerance * tolerance
    for track_id, y, x in points:
        d2 = (y - cy) ** 2 + (x - cx) ** 2
        if d2 <= best_d2:
            best_id, best_d2 = track_id, d2
    return best_id
