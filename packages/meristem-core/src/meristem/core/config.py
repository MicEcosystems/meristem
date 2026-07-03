"""Typed pipeline configuration — the replacement for MiDAP's ``settings.ini``.

A whole experiment is described by one validated YAML file. Crucially, *which model runs* is just
the ``segmenter.name`` / ``tracker.name`` field: change the string, rerun, done. Backend-specific
options live in the free-form ``params`` dicts and are validated against the chosen backend's own
Pydantic ``Params`` model when the pipeline resolves the backend — so the config stays decoupled
from any particular backend's option set while still being checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .contracts import ROI


class ROIConfig(BaseModel):
    """Manual crop bounds, matching :class:`~meristem.core.contracts.ROI` (y, x, height, width)."""

    model_config = {"extra": "forbid"}

    y: int = Field(ge=0)
    x: int = Field(ge=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)

    def to_roi(self) -> ROI:
        return ROI(y=self.y, x=self.x, height=self.height, width=self.width)


class BackendConfig(BaseModel):
    """Selects a backend by registry name and carries its (unvalidated-here) params."""

    model_config = {"extra": "forbid"}

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ChannelConfig(BaseModel):
    """One imaging channel plus what to do with it.

    A channel is segmented independently when ``segment`` is set, and its resulting masks are
    tracked when ``track`` is set. A ``segment=False`` channel is measure-only (carried for future
    per-cell intensity readout, not segmented). ``track`` requires ``segment`` — you cannot track
    without masks.
    """

    model_config = {"extra": "forbid"}

    name: str  # channel label, e.g. "PH", "GFP", "RFP" (must be unique within an input)
    path: str  # single-channel (T, Y, X) TIFF stack for this channel
    segment: bool = True
    track: bool = True

    @model_validator(mode="after")
    def _track_requires_segment(self) -> "ChannelConfig":
        if self.track and not self.segment:
            raise ValueError(
                f"channel {self.name!r}: track=True requires segment=True (cannot track without masks)"
            )
        return self


class InputConfig(BaseModel):
    """The input field of view: either a single stack (``path``) or several ``channels``."""

    model_config = {"extra": "forbid"}

    path: Optional[str] = None  # single-channel shorthand: one (T, Y, X) TIFF
    name: str = "fov"  # FOV/position name (used for the single-channel shorthand and output prefix)
    channels: Optional[List[ChannelConfig]] = None  # multi-channel: per-channel files + roles
    pixel_size_um: Optional[float] = None
    frame_interval_s: Optional[float] = None
    max_frames: Optional[int] = Field(default=None, gt=0)  # cap frames read (handy for quick runs)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "InputConfig":
        if bool(self.path) == bool(self.channels):
            raise ValueError("InputConfig requires exactly one of 'path' or 'channels'")
        if self.channels is not None:
            names = [c.name for c in self.channels]
            if len(names) != len(set(names)):
                raise ValueError(f"channel names must be unique, got {names}")
            if not any(c.segment for c in self.channels):
                raise ValueError("at least one channel must have segment=True")
        return self

    def resolved_channels(self) -> List[ChannelConfig]:
        """Normalize to a channel list — the single-channel shorthand becomes one channel."""
        if self.channels is not None:
            return self.channels
        return [ChannelConfig(name=self.name, path=self.path or "", segment=True, track=True)]


class OutputConfig(BaseModel):
    model_config = {"extra": "forbid"}

    dir: str = "results"
    save_masks: bool = True
    save_tracks: bool = True  # writes napari-tracks .npy + CTC res_track.txt


class PipelineConfig(BaseModel):
    """Top-level experiment description loaded from YAML."""

    model_config = {"extra": "forbid"}

    input: InputConfig
    segmenter: BackendConfig
    tracker: BackendConfig
    crop: Optional[ROIConfig] = None  # manual ROI; None = use the full field of view
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), fh, sort_keys=False)


def example_config() -> PipelineConfig:
    """A minimal, valid config using the built-in mock backends (used in docs and tests)."""
    return PipelineConfig(
        input=InputConfig(path="stack.tif", pixel_size_um=0.065, frame_interval_s=60.0),
        segmenter=BackendConfig(name="mock", params={"threshold": 0.5}),
        tracker=BackendConfig(name="mock", params={"min_iou": 0.1}),
        crop=ROIConfig(y=0, x=0, height=256, width=256),
    )
