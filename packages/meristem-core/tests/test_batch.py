"""Batch processing across positions discovered from a filename pattern."""

from __future__ import annotations

import numpy as np
import tifffile

from meristem.core import BatchSpec, discover_positions, run_batch


def _spec(folder, out, **overrides):
    base = dict(
        folder=str(folder),
        pattern="Timelapse_pos{pos}_{channel}.tif",
        channels=[
            {"name": "PH", "segment": True, "track": True},
            {"name": "RFP", "match": "TxRed", "segment": False, "measure": True},
        ],
        segmenter={"name": "mock", "params": {"threshold": 0.5}},
        tracker={"name": "strack"},
        measure_on="PH",
        pixel_size_um=0.065,
        output_dir=str(out),
    )
    base.update(overrides)
    return BatchSpec(**base)


def test_discover_positions(tmp_path, dividing_stack):
    folder = tmp_path / "data"
    folder.mkdir()
    for pos in ["4", "5"]:
        for tok in ["PH", "TxRed"]:
            tifffile.imwrite(str(folder / f"Timelapse_pos{pos}_{tok}.tif"), dividing_stack.data)
    # pos6 is missing its TxRed file -> not a complete position
    tifffile.imwrite(str(folder / "Timelapse_pos6_PH.tif"), dividing_stack.data)

    spec = _spec(folder, tmp_path / "out")
    assert discover_positions(spec) == ["4", "5"]  # pos6 excluded (missing RFP/TxRed)


def test_run_batch_per_position_outputs(tmp_path, dividing_stack):
    folder = tmp_path / "data"
    folder.mkdir()
    for pos in ["4", "5"]:
        tifffile.imwrite(str(folder / f"Timelapse_pos{pos}_PH.tif"), dividing_stack.data)
        tifffile.imwrite(
            str(folder / f"Timelapse_pos{pos}_TxRed.tif"),
            (dividing_stack.data * 300).astype(np.uint16),
        )
    out = tmp_path / "out"
    results = run_batch(_spec(folder, out), save=True)

    assert set(results) == {"4", "5"}
    for pos in ["4", "5"]:
        assert (out / f"pos{pos}" / "manifest.json").exists()
        assert (out / f"pos{pos}" / "PH_masks.tif").exists()
        assert (out / f"pos{pos}" / "measurements.csv").exists()
        assert results[pos].channel("PH").tracked  # PH segmented + tracked per position
        assert results[pos].measurements.channels == ["RFP"]  # RFP measured through PH


def test_match_token_differs_from_name(tmp_path, dividing_stack):
    # The 'RFP' channel is stored as 'TxRed' in filenames — `match` handles the rename.
    folder = tmp_path / "data"
    folder.mkdir()
    tifffile.imwrite(str(folder / "Timelapse_pos1_PH.tif"), dividing_stack.data)
    tifffile.imwrite(str(folder / "Timelapse_pos1_TxRed.tif"), dividing_stack.data)
    spec = _spec(folder, tmp_path / "out")
    cfg = __import__("meristem.core.batch", fromlist=["build_config"]).build_config(spec, "1")
    paths = {c.name: c.path for c in cfg.input.channels}
    assert paths["RFP"].endswith("Timelapse_pos1_TxRed.tif")  # name RFP -> file token TxRed
    assert paths["PH"].endswith("Timelapse_pos1_PH.tif")
