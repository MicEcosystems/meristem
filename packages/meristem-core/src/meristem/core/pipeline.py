"""The orchestrator: config in, :class:`ResultBundle` out.

The flow is deliberately linear and backend-blind, and runs per channel:

    for each channel:  load  →  manual crop (ROI)  →  segment  →  (optional) track  →  result

Backends are resolved *by name* from the registry, and their params dicts from the config are
validated against each backend's own ``Params`` model here — the one place config meets code. No
model name is ever hardcoded; adding a backend never touches this file.
"""

from __future__ import annotations

from .config import BackendConfig, PipelineConfig
from .contracts import ImageStack, SegMasks, TrackGraph
from .io import ChannelResult, ResultBundle, read_image_stack
from .registry import get_segmenter, get_tracker
from .segmentation.base import SegmenterBackend, SegmenterParams
from .tracking.base import TrackerBackend, TrackerParams


def run_pipeline(config: PipelineConfig, *, save: bool = True) -> ResultBundle:
    """Run the full pipeline described by ``config`` and (optionally) persist the results.

    Every channel marked ``segment`` is read, cropped, and segmented independently; those also
    marked ``track`` are then tracked. Measure-only channels are skipped here (reserved for future
    per-cell intensity readout).
    """
    roi = config.crop.to_roi() if config.crop is not None else None
    results = []
    for ch in config.input.resolved_channels():
        if not ch.segment:
            continue
        stack = read_image_stack(
            ch.path,
            pixel_size_um=config.input.pixel_size_um,
            frame_interval_s=config.input.frame_interval_s,
            name=ch.name,
            max_frames=config.input.max_frames,
        )
        if roi is not None:
            stack = stack.crop(roi)
        results.append(_process_channel(ch.name, stack, config, do_track=ch.track))

    bundle = ResultBundle(
        channels=results, segmenter=config.segmenter.name, tracker=config.tracker.name
    )
    if save:
        bundle.save(
            config.output.dir,
            save_masks=config.output.save_masks,
            save_tracks=config.output.save_tracks,
        )
    return bundle


def run_on_stack(
    stack: ImageStack, config: PipelineConfig, *, do_track: bool = True
) -> ResultBundle:
    """Run crop → segment → (optional) track on one already-loaded stack (no file IO).

    Exposed separately so the napari plugin can feed an in-memory stack (and an interactively
    drawn ROI) straight through the same code path the headless pipeline uses. Returns a bundle
    with a single :class:`ChannelResult`.
    """
    if config.crop is not None:
        stack = stack.crop(config.crop.to_roi())
    result = _process_channel(stack.name, stack, config, do_track=do_track)
    return ResultBundle(
        channels=[result], segmenter=config.segmenter.name, tracker=config.tracker.name
    )


def _process_channel(
    name: str, stack: ImageStack, config: PipelineConfig, *, do_track: bool
) -> ChannelResult:
    masks = segment(stack, config.segmenter)
    tracks = track(stack, masks, config.tracker) if do_track else None
    return ChannelResult(name=name, stack=stack, masks=masks, tracks=tracks)


def segment(stack: ImageStack, backend_config: BackendConfig) -> SegMasks:
    """Resolve, configure, and run a segmentation backend by name."""
    backend, params = _resolve_segmenter(backend_config)
    backend.load(params)
    return backend.segment(stack)


def track(stack: ImageStack, masks: SegMasks, backend_config: BackendConfig) -> TrackGraph:
    """Resolve, configure, and run a tracking backend by name."""
    backend, params = _resolve_tracker(backend_config)
    backend.load(params)
    return backend.track(stack, masks)


def _resolve_segmenter(cfg: BackendConfig) -> "tuple[SegmenterBackend, SegmenterParams]":
    cls = get_segmenter(cfg.name)
    params = cls.Params.model_validate(cfg.params)  # validates config against the backend's schema
    return cls(), params


def _resolve_tracker(cfg: BackendConfig) -> "tuple[TrackerBackend, TrackerParams]":
    cls = get_tracker(cfg.name)
    params = cls.Params.model_validate(cfg.params)
    return cls(), params
