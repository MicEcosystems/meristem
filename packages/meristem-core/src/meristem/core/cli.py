"""Command-line entry point: ``meristem``.

Subcommands cover the headless workflow:

    meristem list                     # show installed segmentation & tracking backends
    meristem run experiment.yaml      # segment + track + measure in one pass
    meristem segment experiment.yaml  # stage 1: segment only, write masks (then inspect them)
    meristem track experiment.yaml    # stage 2: track saved masks (--frames START:STOP)
    meristem compare compare.yaml     # run several models on the same input and tabulate them

Segment and track are separate stages on purpose: run segmentation, evaluate the masks visually,
then track only the frames worth tracking. All logic lives in :mod:`meristem.core.pipeline` and
:mod:`meristem.core.compare`; the napari plugin is a parallel front-end over the same functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .compare import CompareSpec, format_report, run_comparison
from .config import PipelineConfig
from .pipeline import run_pipeline, run_segmentation, run_tracking
from .registry import list_segmenters, list_trackers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meristem", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list installed segmentation and tracking backends")

    run_p = sub.add_parser("run", help="segment + track + measure in one pass")
    run_p.add_argument("config", type=Path, help="path to the experiment YAML config")
    run_p.add_argument("--no-save", action="store_true", help="do not write result files")

    seg_p = sub.add_parser("segment", help="stage 1: segment only, write masks")
    seg_p.add_argument("config", type=Path, help="path to the experiment YAML config")
    seg_p.add_argument("--no-save", action="store_true", help="do not write result files")

    trk_p = sub.add_parser("track", help="stage 2: track saved masks")
    trk_p.add_argument("config", type=Path, help="path to the experiment YAML config")
    trk_p.add_argument("--masks-dir", type=str, default=None, help="dir with saved masks (default: output.dir)")
    trk_p.add_argument("--frames", type=str, default=None, help="frame window START:STOP (e.g. 5:40)")
    trk_p.add_argument("--no-save", action="store_true", help="do not write result files")

    cmp_p = sub.add_parser("compare", help="compare several models on the same input")
    cmp_p.add_argument("spec", type=Path, help="path to the comparison YAML spec")

    args = parser.parse_args(argv)

    if args.command == "list":
        return _cmd_list()
    if args.command == "run":
        return _cmd_run(args.config, save=not args.no_save)
    if args.command == "segment":
        return _cmd_segment(args.config, save=not args.no_save)
    if args.command == "track":
        return _cmd_track(args.config, args.masks_dir, args.frames, save=not args.no_save)
    if args.command == "compare":
        return _cmd_compare(args.spec)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _cmd_compare(spec_path: Path) -> int:
    report = run_comparison(CompareSpec.from_yaml(spec_path))
    print(format_report(report))
    return 0


def _cmd_segment(config_path: Path, *, save: bool) -> int:
    config = PipelineConfig.from_yaml(config_path)
    bundle = run_segmentation(config, save=save)
    print(f"Segmented [segmenter={bundle.segmenter}]")
    for ch in bundle.channels:
        h, w = ch.stack.shape_yx
        print(f"  {ch.name}: {ch.stack.n_frames} frames, {h}x{w}, cells/frame {ch.masks.n_cells_per_frame()}")
    if save:
        print(f"Masks written to: {config.output.dir}  (inspect, then `meristem track`)")
    return 0


def _cmd_track(config_path: Path, masks_dir: Optional[str], frames: Optional[str], *, save: bool) -> int:
    config = PipelineConfig.from_yaml(config_path)
    window = _parse_frames(frames)
    bundle = run_tracking(config, masks_dir=masks_dir, frames=window, save=save)
    span = f" frames {window[0]}:{window[1]}" if window else ""
    print(f"Tracked [tracker={bundle.tracker}]{span}")
    for ch in bundle.channels:
        if ch.tracked:
            print(f"  {ch.name}: {ch.tracks.n_detections} detections, {len(ch.tracks.divisions())} divisions")
    if bundle.measurements is not None:
        m = bundle.measurements
        print(f"  measured {', '.join(m.channels)} over {len(m.rows)} cell-frames")
    if bundle.track_summary is not None:
        print(f"  summarized {len(bundle.track_summary.rows)} tracks")
    if save:
        print(f"Results written to: {config.output.dir}")
    return 0


def _parse_frames(frames: Optional[str]) -> Optional[tuple]:
    if not frames:
        return None
    if ":" not in frames:
        raise SystemExit(f"--frames must be START:STOP, got {frames!r}")
    start_s, stop_s = frames.split(":", 1)
    return (int(start_s), int(stop_s))


def _cmd_list() -> int:
    print("Segmenters:")
    for name in list_segmenters():
        print(f"  - {name}")
    print("Trackers:")
    for name in list_trackers():
        print(f"  - {name}")
    return 0


def _cmd_run(config_path: Path, *, save: bool) -> int:
    config = PipelineConfig.from_yaml(config_path)
    bundle = run_pipeline(config, save=save)
    print(f"Done [segmenter={bundle.segmenter}, tracker={bundle.tracker}]")
    for ch in bundle.channels:
        h, w = ch.stack.shape_yx
        line = f"  {ch.name}: {ch.stack.n_frames} frames, {h}x{w}"
        if ch.tracked:
            line += f", {ch.tracks.n_detections} detections, {len(ch.tracks.divisions())} divisions"
        else:
            line += ", segmented only (not tracked)"
        print(line)
    if bundle.measurements is not None:
        m = bundle.measurements
        print(f"  measured {', '.join(m.channels)} over {len(m.rows)} cell-frames")
    if save:
        print(f"Results written to: {config.output.dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
