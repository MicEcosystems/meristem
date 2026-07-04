"""The orchestrator: config in, :class:`ResultBundle` out.

The flow is deliberately linear and backend-blind, and runs per channel:

    for each channel:  load  →  manual crop (ROI)  →  segment  →  (optional) track  →  result

Backends are resolved *by name* from the registry, and their params dicts from the config are
validated against each backend's own ``Params`` model here — the one place config meets code. No
model name is ever hardcoded; adding a backend never touches this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .config import BackendConfig, ChannelConfig, PipelineConfig
from .contracts import ImageStack, SegMasks, TrackGraph
from .io import ChannelResult, ResultBundle, read_image_stack, read_masks
from .measure import MeasurementTable, measure_intensities
from .registry import get_segmenter, get_tracker
from .segmentation.base import SegmenterBackend, SegmenterParams
from .tracking.base import TrackerBackend, TrackerParams

FrameRange = Tuple[int, int]  # (start, stop), half-open


def run_pipeline(config: PipelineConfig, *, save: bool = True) -> ResultBundle:
    """Run the full pipeline (segment + track + measure) in one pass and persist the results.

    Every channel marked ``segment`` is read, cropped, and segmented; those also marked ``track``
    are tracked. ``measure`` channels are read out per-cell through the ``measure_on`` masks. For
    the modular workflow — segment, inspect, then track chosen frames — use :func:`run_segmentation`
    and :func:`run_tracking` instead.
    """
    channels = config.input.resolved_channels()
    results = []
    result_by_name = {}
    for ch in channels:
        if not ch.segment:
            continue
        cr = _process_channel(ch.name, _load_stack(config, ch), config, do_track=ch.track)
        results.append(cr)
        result_by_name[ch.name] = cr

    measurements = _measure_channels(config, result_by_name, frames=None)
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


def run_segmentation(config: PipelineConfig, *, save: bool = True) -> ResultBundle:
    """Stage 1: segment the ``segment`` channels only, and save masks (no tracking).

    This is the standalone segmentation step — run it, inspect the masks visually, then decide
    which frames are worth tracking before running :func:`run_tracking`.
    """
    results = []
    for ch in config.input.resolved_channels():
        if not ch.segment:
            continue
        stack = _load_stack(config, ch)
        masks = segment(stack, config.segmenter)
        results.append(ChannelResult(name=ch.name, stack=stack, masks=masks, tracks=None))
    bundle = ResultBundle(
        channels=results, segmenter=config.segmenter.name, tracker=config.tracker.name
    )
    if save:
        # Write masks + binary only; there are no tracks yet.
        bundle.save(
            config.output.dir,
            save_masks=config.output.save_masks,
            save_binary=config.output.save_binary,
            save_tracks=False,
        )
    return bundle


def run_tracking(
    config: PipelineConfig,
    *,
    masks_dir: Optional[str] = None,
    frames: Optional[FrameRange] = None,
    save: bool = True,
) -> ResultBundle:
    """Stage 2: track previously-saved masks, optionally restricted to a frame window.

    Loads ``{channel}_masks.tif`` from ``masks_dir`` (default: ``output.dir`` — where
    :func:`run_segmentation` wrote them), optionally slices to ``frames=(start, stop)`` so you can
    track only the frames where segmentation looked good, tracks the ``track`` channels, and
    measures the ``measure`` channels. Does not overwrite the saved masks.
    """
    md = Path(masks_dir) if masks_dir else Path(config.output.dir)
    fr = range(*frames) if frames is not None else None

    results = []
    result_by_name = {}
    for ch in config.input.resolved_channels():
        if not ch.segment:
            continue
        masks = _slice_masks(read_masks(md / f"{ch.name}_masks.tif"), fr)
        stack = _load_stack(config, ch, frames=fr)
        _check_aligned(ch.name, stack, masks)
        tracks = track(stack, masks, config.tracker) if ch.track else None
        cr = ChannelResult(name=ch.name, stack=stack, masks=masks, tracks=tracks)
        results.append(cr)
        result_by_name[ch.name] = cr

    measurements = _measure_channels(config, result_by_name, frames=fr)
    bundle = ResultBundle(
        channels=results,
        segmenter=config.segmenter.name,
        tracker=config.tracker.name,
        measurements=measurements,
    )
    if save:
        # Masks already exist from the segment stage (and may be a full-length superset of this
        # window) — write only tracks + measurements here.
        bundle.save(
            config.output.dir,
            save_masks=False,
            save_binary=False,
            save_tracks=config.output.save_tracks,
        )
    return bundle


def _load_stack(
    config: PipelineConfig, ch: ChannelConfig, frames: Optional[range] = None
) -> ImageStack:
    """Load one channel's stack with the config's crop, honoring a frame window or max_frames."""
    stack = read_image_stack(
        ch.path,
        pixel_size_um=config.input.pixel_size_um,
        frame_interval_s=config.input.frame_interval_s,
        name=ch.name,
        max_frames=None if frames is not None else config.input.max_frames,
        frames=frames,
    )
    if config.crop is not None:
        stack = stack.crop(config.crop.to_roi())
    return stack


def _measure_channels(
    config: PipelineConfig, result_by_name: dict, frames: Optional[range]
) -> Optional[MeasurementTable]:
    measure_channels = [c for c in config.input.resolved_channels() if c.measure]
    if not measure_channels:
        return None
    base = result_by_name.get(config.measure_on)
    if base is None:  # the measure_on channel wasn't segmented in this run
        return None
    channel_images = {c.name: _load_stack(config, c, frames=frames).data for c in measure_channels}
    return measure_intensities(
        base.masks, channel_images, tracks=base.tracks, pixel_size_um=config.input.pixel_size_um
    )


def _slice_masks(masks: SegMasks, frames: Optional[range]) -> SegMasks:
    if frames is None:
        return masks
    sliced = masks.data[frames.start : frames.stop]
    if sliced.shape[0] == 0:
        raise ValueError(
            f"frame window {frames.start}:{frames.stop} selects no frames from masks with "
            f"{masks.n_frames} frames"
        )
    return SegMasks(data=sliced, source=masks.source)


def _check_aligned(name: str, stack: ImageStack, masks: SegMasks) -> None:
    if stack.n_frames != masks.n_frames:
        raise ValueError(
            f"channel {name!r}: image has {stack.n_frames} frames but masks have {masks.n_frames}; "
            "the tracking frame window must lie within the segmented frames"
        )
    masks_yx = (int(masks.data.shape[1]), int(masks.data.shape[2]))
    if stack.shape_yx != masks_yx:
        raise ValueError(
            f"channel {name!r}: image {stack.shape_yx} and masks {masks_yx} differ in Y/X "
            "(crop must match the one used for segmentation)"
        )


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
