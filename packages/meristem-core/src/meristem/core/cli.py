"""Command-line entry point: ``meristem``.

Subcommands cover the headless workflow:

    meristem list                     # show installed segmentation & tracking backends
    meristem run experiment.yaml      # run the pipeline described by a config file
    meristem compare compare.yaml     # run several models on the same input and tabulate them

This is intentionally thin — all logic lives in :mod:`meristem.core.pipeline` and
:mod:`meristem.core.compare`. The napari plugin is a parallel front-end over the same functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compare import CompareSpec, format_report, run_comparison
from .config import PipelineConfig
from .pipeline import run_pipeline
from .registry import list_segmenters, list_trackers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meristem", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list installed segmentation and tracking backends")

    run_p = sub.add_parser("run", help="run the pipeline from a YAML config")
    run_p.add_argument("config", type=Path, help="path to the experiment YAML config")
    run_p.add_argument("--no-save", action="store_true", help="do not write result files")

    cmp_p = sub.add_parser("compare", help="compare several models on the same input")
    cmp_p.add_argument("spec", type=Path, help="path to the comparison YAML spec")

    args = parser.parse_args(argv)

    if args.command == "list":
        return _cmd_list()
    if args.command == "run":
        return _cmd_run(args.config, save=not args.no_save)
    if args.command == "compare":
        return _cmd_compare(args.spec)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _cmd_compare(spec_path: Path) -> int:
    report = run_comparison(CompareSpec.from_yaml(spec_path))
    print(format_report(report))
    return 0


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
    if save:
        print(f"Results written to: {config.output.dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
