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
    track: bool = False  # opt-in; the single-channel shorthand sets this True explicitly
    measure: bool = False  # read out per-cell intensity (via the PipelineConfig.measure_on masks)

    @model_validator(mode="after")
    def _validate_roles(self) -> "ChannelConfig":
        if self.track and not self.segment:
            raise ValueError(
                f"channel {self.name!r}: track=True requires segment=True (cannot track without masks)"
            )
        if not (self.segment or self.measure):
            raise ValueError(
                f"channel {self.name!r}: must have at least one role (segment or measure)"
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


class PostprocessConfig(BaseModel):
    """Size-filter cleanup applied to segmentation masks (MiDAP-style)."""

    model_config = {"extra": "forbid"}

    min_size_frac: float = Field(default=0.01, ge=0)  # drop labels < this * mean area (MiDAP: 0.01)
    max_size_frac: Optional[float] = Field(default=None, gt=0)  # optional: drop labels > this * mean
    min_size_px: Optional[int] = Field(default=None, gt=0)  # optional absolute floor in pixels


class RegisterConfig(BaseModel):
    """Drift-registration settings, applied to all channels before cropping."""

    model_config = {"extra": "forbid"}

    on: str  # channel used to estimate drift (e.g. "PH"); its shifts apply to every channel
    # "previous" accumulates frame-to-frame shifts; "first" aligns every frame to frame 0 (MiDAP's
    # choice). Default is "previous": in a growing monolayer the field decorrelates from frame 0, so
    # "first" tends to return ~0 drift on this kind of data, whereas consecutive frames stay similar.
    # Use "first" for static-background / mother-machine data (MiDAP-compatible).
    reference: str = "previous"

    @model_validator(mode="after")
    def _check_reference(self) -> "RegisterConfig":
        if self.reference not in ("previous", "first"):
            raise ValueError(f"register.reference must be 'previous' or 'first', got {self.reference!r}")
        return self


class OutputConfig(BaseModel):
    model_config = {"extra": "forbid"}

    dir: str = "results"
    save_masks: bool = True  # instance-labeled masks (the primary output), as uint16 TIFF
    save_binary: bool = True  # also write a 0/255 binary foreground TIFF (MiDAP's _seg_bin parity)
    save_tracks: bool = True  # writes napari-tracks .npy + CTC res_track.txt


class PipelineConfig(BaseModel):
    """Top-level experiment description loaded from YAML."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    input: InputConfig
    segmenter: BackendConfig
    tracker: BackendConfig
    # YAML key is `register`; the attribute is `registration` to avoid shadowing ABCMeta.register.
    registration: Optional[RegisterConfig] = Field(default=None, alias="register")
    crop: Optional[ROIConfig] = None  # manual ROI; None = use the full field of view
    postprocess: Optional[PostprocessConfig] = None  # size-filter cleanup after segmentation
    # Name of the segmented channel whose masks define cells for intensity measurement of the
    # `measure` channels (e.g. "PH"). Required when any channel has measure=True.
    measure_on: Optional[str] = None
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _validate_register(self) -> "PipelineConfig":
        if self.registration is not None:
            names = {c.name for c in self.input.resolved_channels()}
            if self.registration.on not in names:
                raise ValueError(
                    f"register.on={self.registration.on!r} must name an input channel; "
                    f"have {sorted(names)}"
                )
        return self

    @model_validator(mode="after")
    def _validate_measurement(self) -> "PipelineConfig":
        channels = self.input.resolved_channels()
        if any(c.measure for c in channels):
            if not self.measure_on:
                raise ValueError(
                    "measure_on is required when any channel has measure=True "
                    "(name the segmented channel whose masks to measure through, e.g. 'PH')"
                )
            segmented = {c.name for c in channels if c.segment}
            if self.measure_on not in segmented:
                raise ValueError(
                    f"measure_on={self.measure_on!r} must name a segmented channel; "
                    f"segmented channels are {sorted(segmented)}"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(
                self.model_dump(mode="json", exclude_none=True, by_alias=True), fh, sort_keys=False
            )


def example_config() -> PipelineConfig:
    """A minimal, valid config using the built-in mock backends (used in docs and tests)."""
    return PipelineConfig(
        input=InputConfig(path="stack.tif", pixel_size_um=0.065, frame_interval_s=60.0),
        segmenter=BackendConfig(name="mock", params={"threshold": 0.5}),
        tracker=BackendConfig(name="mock", params={"min_iou": 0.1}),
        crop=ROIConfig(y=0, x=0, height=256, width=256),
    )
