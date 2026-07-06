"""Batch processing — run the same pipeline across many positions (FOVs).

MiDAP scanned a folder of ``..._pos4_...`` / ``..._pos5_...`` files and processed every position.
This does the same from a filename *pattern* with ``{pos}`` and ``{channel}`` tokens: it discovers
the positions present, builds a :class:`~meristem.core.config.PipelineConfig` per position (sharing
the segmenter/tracker/register/crop/measurement settings), and runs each into its own output
subfolder.

```yaml
# batch.yaml
folder: /data/experiment1
pattern: "Timelapse_pos{pos}_{channel}.tif"    # {pos} and {channel} are filled in per file
channels:
  - {name: PH,  match: PH,    segment: false}  # 'match' = the {channel} token in the filename
  - {name: GFP, match: GFP,   measure: true}
  - {name: RFP, match: TxRed, measure: true}
register: {channel: PH}
segmenter: {name: midap_omni_phase_v01}
tracker:   {name: strack}
measure_on: PH
output_dir: results            # -> results/pos4/, results/pos5/, ...
```
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from .config import (
    BackendConfig,
    ChannelConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PostprocessConfig,
    RegisterConfig,
    ROIConfig,
)
from .io import ResultBundle
from .pipeline import run_pipeline


class BatchChannel(BaseModel):
    """A channel in a batch: its role plus the ``{channel}`` token used to find its file."""

    model_config = {"extra": "forbid"}

    name: str
    # the {channel} token in the filename; defaults to `name`. Set it when they differ, e.g. a
    # channel named "RFP" whose files use "TxRed".
    match: Optional[str] = None
    segment: bool = True
    track: bool = False
    measure: bool = False

    @property
    def token(self) -> str:
        return self.match or self.name


class BatchSpec(BaseModel):
    """A batch run: where the files are, how they're named, and the shared pipeline settings."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    folder: str
    pattern: str  # filename pattern with {pos} and {channel} tokens
    channels: List[BatchChannel]
    positions: Optional[List[str]] = None  # explicit list; None = auto-discover from the folder

    segmenter: BackendConfig
    tracker: BackendConfig
    registration: Optional[RegisterConfig] = Field(default=None, alias="register")
    crop: Optional[ROIConfig] = None
    postprocess: Optional[PostprocessConfig] = None
    measure_on: Optional[str] = None
    pixel_size_um: Optional[float] = None
    frame_interval_s: Optional[float] = None
    max_frames: Optional[int] = None
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BatchSpec":
        with open(path, "r") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})


def _pattern_to_regex(pattern: str) -> re.Pattern:
    esc = re.escape(pattern)
    esc = esc.replace(re.escape("{pos}"), r"(?P<pos>[^/]+?)")
    esc = esc.replace(re.escape("{channel}"), r"(?P<channel>[^/]+?)")
    return re.compile("^" + esc + "$")


def discover_positions(spec: BatchSpec) -> List[str]:
    """Scan ``spec.folder`` for positions that have a file for every required channel."""
    folder = Path(spec.folder)
    rx = _pattern_to_regex(spec.pattern)
    nested = "/" in spec.pattern
    wanted = {c.token for c in spec.channels}
    found: Dict[str, set] = {}
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(folder)) if nested else p.name
        m = rx.match(rel)
        if m:
            found.setdefault(m.group("pos"), set()).add(m.group("channel"))
    return sorted(pos for pos, chans in found.items() if wanted <= chans)


def build_config(spec: BatchSpec, pos: str) -> PipelineConfig:
    """Build the single-position :class:`PipelineConfig` for one position id."""
    folder = Path(spec.folder)
    channels = [
        ChannelConfig(
            name=c.name,
            path=str(folder / spec.pattern.format(pos=pos, channel=c.token)),
            segment=c.segment,
            track=c.track,
            measure=c.measure,
        )
        for c in spec.channels
    ]
    subdir = f"pos{pos}" if pos.isdigit() else pos
    return PipelineConfig(
        input=InputConfig(
            channels=channels,
            pixel_size_um=spec.pixel_size_um,
            frame_interval_s=spec.frame_interval_s,
            max_frames=spec.max_frames,
        ),
        segmenter=spec.segmenter,
        tracker=spec.tracker,
        registration=spec.registration,
        crop=spec.crop,
        postprocess=spec.postprocess,
        measure_on=spec.measure_on,
        output=OutputConfig(dir=str(Path(spec.output_dir) / subdir)),
    )


def run_batch(
    spec: BatchSpec, *, save: bool = True, positions: Optional[List[str]] = None
) -> Dict[str, ResultBundle]:
    """Run the pipeline for every position; return ``{position: ResultBundle}``.

    Positions come from ``positions`` (arg), else ``spec.positions``, else auto-discovery. Each
    position writes to ``output_dir/pos<id>/``. One position failing does not stop the rest.
    """
    poss = positions or spec.positions or discover_positions(spec)
    results: Dict[str, ResultBundle] = {}
    for pos in poss:
        results[pos] = run_pipeline(build_config(spec, pos), save=save)
    return results
