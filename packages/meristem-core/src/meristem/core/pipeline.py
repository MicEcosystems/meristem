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
from .measure import measure_intensities
from .registry import get_segmenter, get_tracker
from .segmentation.base import SegmenterBackend, SegmenterParams
from .tracking.base import TrackerBackend, TrackerParams


def run_pipeline(config: PipelineConfig, *, save: bool = True) -> ResultBundle:
    """Run the full pipeline described by ``config`` and (optionally) persist the results.

    Every channel marked ``segment`` is read, cropped, and segmented independently; those also
    marked ``track`` are then tracked. ``measure`` channels are not segmented — instead their
    per-cell intensity is read out through the ``measure_on`` channel's masks (and joined to its
    tracks), producing a measurements table.
    """
    roi = config.crop.to_roi() if config.crop is not None else None
    channels = config.input.resolved_channels()

    def load(ch) -> ImageStack:
        stack = read_image_stack(
            ch.path,
            pixel_size_um=config.input.pixel_size_um,
            frame_interval_s=config.input.frame_interval_s,
            name=ch.name,
            max_frames=config.input.max_frames,
        )
        return stack.crop(roi) if roi is not None else stack

    results = []
    result_by_name = {}
    for ch in channels:
        if not ch.segment:
            continue
        cr = _process_channel(ch.name, load(ch), config, do_track=ch.track)
        results.append(cr)
        result_by_name[ch.name] = cr

    measurements = None
    measure_channels = [c for c in channels if c.measure]
    if measure_channels:
        base = result_by_name[config.measure_on]  # validated to be a segmented channel
        channel_images = {c.name: load(c).data for c in measure_channels}
        measurements = measure_intensities(
            base.masks,
            channel_images,
            tracks=base.tracks,
            pixel_size_um=config.input.pixel_size_um,
        )

    bundle = ResultBundle(
        channels=results,
        segmenter=config.segmenter.name,
        tracker=config.tracker.name,
        measurements=measurements,
    )
    if save:
        bundle.save(
            config.output.dir,
            save_masks=config.output.save_masks,
            save_binary=config.output.save_binary,
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
