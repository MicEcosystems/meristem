# Meristem — User Manual (v1.0)

Meristem segments and tracks bacterial cells growing as 2D monolayers in microfluidic devices. It
is a modular successor to [MiDAP](https://github.com/Microbial-Systems-Ecology/midap): you pick a
segmentation model and a tracker **by name**, and swapping either is a one-line config change.

- **Pipeline:** `register drift → crop ROI → segment → track → measure`
- **Two ways to run it:** one-shot (`meristem run`) or as separate stages you can inspect between
  (`meristem segment`, then `meristem track`).
- **Interactive:** a napari plugin exposes the same steps as dock widgets.

---

## 0. For biologists — the 3-line start

No YAML. In a fresh Python 3.11 environment, from the project folder:

```bash
python3.11 -m venv meristem-env && source meristem-env/bin/activate   # make + activate an env
cd path/to/Phusion                                                   # the folder with packages/
pip install -e packages/meristem-core \
            -e 'packages/meristem-models-cellpose[cellpose]' \
            -e 'packages/meristem-trackers[trackastra]' \
            -e 'packages/meristem-napari[gui]'
meristem-gui                                                         # launch
```

> Once the packages are published to PyPI this collapses to one line —
> `pip install 'meristem-napari[all]'` — but today they install from these local folders.

napari opens with a **Run pipeline** panel already docked. Then:

1. **Drag your image** (a TIFF stack) into the napari window.
2. Pick a **Segmentation model** and a **Tracker** from the dropdowns.
3. *(optional)* Add a **Shapes** layer, draw a rectangle to crop; leave **correct drift** ticked.
4. Press **Run ▶**.

Masks and tracks appear as layers you can scroll through. That's it — the dropdowns do the rest.
For the segment-then-inspect-then-track workflow, use the **1. Segment** / **2. Track** panels below
it. Everything past this point is for scripting and batch runs.

> Model weights download on first run (Cellpose-SAM ~1.15 GB), so the first segmentation is slow;
> after that they're cached. On Apple Silicon the GPU (MPS) is used automatically.

---

## 1. Installation

Meristem is five packages. The core is dependency-light (NumPy/Pydantic/networkx/tifffile) and runs
on Python ≥ 3.9. The model backends pull heavy ML stacks (PyTorch, Cellpose, Trackastra) and want
Python ≥ 3.10.

```bash
# Core only — mock + strack backends, full CLI, no ML:
pip install -e packages/meristem-core

# Real segmentation models (needs a Python 3.10+ env with torch):
pip install -e 'packages/meristem-models-cellpose[cellpose]'   # Cellpose-SAM
pip install -e 'packages/meristem-models-cellpose[omnipose]'   # Omnipose

# Modern tracker:
pip install -e 'packages/meristem-trackers[trackastra]'

# Interactive GUI:
pip install -e 'packages/meristem-napari[gui]'                 # napari + magicgui
```

Model **weights are not bundled** — Cellpose-SAM (~1.15 GB) and Trackastra download on first use and
cache to `~/.cellpose` / `~/.trackastra`. On Apple Silicon, segmentation uses the MPS GPU
automatically (`device: auto`).

Check what's installed:

```bash
meristem list      # lists the segmenters and trackers the registry can see
```

---

## 2. Quickstart

`strack` and the `mock` backends are pure NumPy, so this runs with only the core installed:

```bash
cat > experiment.yaml <<'YAML'
input: { path: /path/to/movie.tif, name: pos1 }
crop:  { y: 700, x: 700, height: 512, width: 512 }
segmenter: { name: mock }
tracker:   { name: strack }
YAML

meristem run experiment.yaml
```

For real segmentation, change `segmenter.name` to `cellpose-sam` (in a torch-enabled env).

---

## 3. The config file

One YAML describes an experiment. Only `input`, `segmenter`, and `tracker` are required.

```yaml
input:
  # Either a single stack:
  path: movie.tif
  name: pos1
  # …or several channels (each a single-channel TIFF stack):
  channels:
    - { name: PH,  path: pos1_PH.tif,  segment: true,  track: true }
    - { name: GFP, path: pos1_GFP.tif, segment: false, measure: true }
    - { name: RFP, path: pos1_RFP.tif, segment: false, measure: true }
  pixel_size_um: 0.065       # enables cell areas in µm²
  frame_interval_s: 300.0    # enables growth rates (per hour)
  max_frames: 50             # optional: only read the first N frames

register:                    # optional drift correction (applied before crop)
  channel: PH                # estimate drift on this channel, apply to all
  reference: previous        # "previous" (default) or "first" (MiDAP-style)

crop: { y: 700, x: 700, height: 512, width: 512 }   # optional manual ROI

segmenter:
  name: cellpose-sam         # or omnipose, mock, …  (see `meristem list`)
  params: { device: auto }   # backend-specific; validated against the backend

tracker:
  name: strack               # or trackastra, mock, …
  params: {}

postprocess:                 # optional size-filter cleanup (MiDAP-style)
  min_size_frac: 0.01        # drop labels < 1% of mean area
  max_size_frac: 5.0         # optional: drop merged clumps > 5× mean
  min_size_px: 20            # optional absolute floor

measure_on: PH               # required if any channel has measure: true

output:
  dir: results
  save_masks: true
  save_binary: true          # 0/255 foreground TIFF (MiDAP _seg_bin parity)
  save_tracks: true
```

**Per-channel roles.** `segment` → produce masks; `track` → track those masks (requires `segment`);
`measure` → read per-cell intensity through the `measure_on` channel's masks (not segmented itself).
A dual-reporter experiment segments PH and measures GFP+RFP; two strains you want tracked separately
would each get `segment: true, track: true`.

---

## 4. Two workflows

### One-shot
```bash
meristem run experiment.yaml
```
Runs register → crop → segment → track → measure and writes everything.

### Modular (segment, inspect, then track chosen frames)
This is the recommended workflow for real analysis — segmentation quality varies across a movie, so
you inspect before committing to tracking.

```bash
meristem segment experiment.yaml            # writes masks only (no tracks)
#   … open results/PH_masks.tif in napari/Fiji, scroll, judge quality …
meristem track experiment.yaml --frames 5:180   # track only the good frames, then measure
```

`track` reads the saved masks (it does not re-segment), tracks the `[start, stop)` window, and never
overwrites the masks. Drift shifts saved by `segment` are reused so alignment matches.

> Note: in a windowed `track` run, output frame indices are **window-local** (0-based).

---

## 5. CLI reference

| Command | Does |
|---|---|
| `meristem list` | Show installed segmentation & tracking backends |
| `meristem run <cfg>` | Segment + track + measure in one pass |
| `meristem segment <cfg>` | Stage 1: segment only, write masks |
| `meristem track <cfg> [--frames A:B] [--masks-dir DIR]` | Stage 2: track saved masks |
| `meristem compare <spec>` | Run several models on the same input and tabulate them |

`--no-save` skips writing files (any command).

---

## 6. Outputs

All written to `output.dir`. Per segmented channel (prefixed by name, e.g. `PH_`):

| File | Contents |
|---|---|
| `PH_masks.tif` | Instance masks — `(T,Y,X)`, each cell a unique integer label. |
| `PH_seg_bin.tif` | Binary foreground `0/255` (MiDAP `_seg_bin`). |
| `PH_tracks.npy` | napari Tracks array `(N,4)` = `[track_id, t, y, x]`. |
| `PH_tracks_graph.json` | Lineage `{child_track: [parents]}` (divisions). |
| `PH_res_track.txt` | Cell Tracking Challenge lineage (`id start end parent`). |
| `PH_drift.npy` | Per-frame drift shifts (if registration on). |

Analysis tables:

| File | Grain | Key columns |
|---|---|---|
| `measurements.csv` | one row per **cell-frame** | `frame, label, track_id, area_px, area_um2, centroid_y/x`, `{chan}_mean/total/median` |
| `track_summary.csv` | one row per **track/lineage** | `track_id, parent, start/end/n_frames, divides, n_daughters, area_first/last/mean, displacement_px, growth_rate_per_hr, {chan}_mean` |

`manifest.json` indexes everything (backends used, per-channel stats, file names).

A PH-only run (no fluorescence) still produces `measurements.csv` (areas) and `track_summary.csv`
(growth/lineage) — intensity columns are simply absent.

---

## 7. napari plugin

```bash
napari         # Plugins → Meristem
```

- **1. Segment** — drop a channel TIFF in, pick a backend from the dropdown, optionally draw a
  rectangle on a Shapes layer for the crop, run. Masks appear as a Labels layer to inspect.
- **2. Track + measure** — pick a tracker and a frame window; the lineage appears as a Tracks layer.

The widgets call the same `meristem.core` functions as the CLI, so results match.

---

## 8. Backends

**Segmentation** (`meristem list`):
- `cellpose-sam` — Cellpose-SAM super-generalist; strong default, diameter-agnostic.
- `omnipose` — purpose-built for elongated/dense bacteria; exposes MiDAP's four bacterial models
  (`bact_phase_omni` default, `bact_fluor_omni`, `bact_phase_cp`, `bact_fluor_cp`).
- `mock` — thresholding, no ML (testing/plumbing).

**Tracking:**
- `trackastra` — transformer linker, division-aware, tuning-free. Best modern default for dense,
  dividing monolayers. Defaults to CPU (see performance notes).
- `strack` — native port of MiDAP's S-track (overlap + distance + division-angle validation).
- `mock` — greedy overlap baseline.

**Choosing between them** — `meristem compare` runs several on the same input:

```yaml
# compare.yaml
input: { path: movie.tif, max_frames: 20 }
crop:  { y: 700, x: 700, height: 512, width: 512 }
segmenters: [ { name: cellpose-sam }, { name: omnipose } ]
trackers:   [ { name: strack }, { name: trackastra } ]
```
```bash
meristem compare compare.yaml    # tabulates cells/area/time and detections/divisions/tracks
```

---

## 9. Performance & limitations

Measured on a 217-frame, 512×512 crop, Cellpose-SAM on Apple-Silicon MPS + strack:

- **~34 min**, peak ~4.2 GB. Segmentation dominates; the per-cell measurement loop is the second
  cost (72k cell-frames).
- Drift over that movie was **45 px** — registration is not optional at this length.

Known limitations in v1:
- **strack fragments** on dense dividing colonies (many short tracks). Prefer `trackastra`, and use
  `compare` to check. Ground-truth tracking metrics (CTC TRA/DET) are not built in yet.
- Windowed `track` runs report **window-local** frame indices.
- The measurement loop is per-label NumPy; a `regionprops` rewrite would speed up huge frames.
- Registration is **translation-only** (no rotation), matching MiDAP.
- The napari GUI needs a real OpenGL session (it can't render headless).

---

## 10. Python API

Everything the CLI does is available as functions:

```python
from meristem.core import PipelineConfig, run_pipeline, run_segmentation, run_tracking

cfg = PipelineConfig.from_yaml("experiment.yaml")

# one-shot
bundle = run_pipeline(cfg)
ch = bundle.channel("PH")
print(ch.masks.data.shape, ch.tracks.n_detections, len(ch.tracks.divisions()))
print(bundle.track_summary.rows[0].growth_rate_per_hr)

# or staged
run_segmentation(cfg)
bundle = run_tracking(cfg, frames=(5, 180))
```

`bundle.channels[i]` gives `ChannelResult(name, stack, masks, tracks)`; `tracks` is a `TrackGraph`
with `.to_napari_tracks()` and `.to_ctc()`. `bundle.measurements` and `bundle.track_summary` expose
`.to_csv(path)`.

Write a new backend by subclassing `SegmenterBackend` / `TrackerBackend` and registering it via an
entry point (`meristem.segmenters` / `meristem.trackers`) — the core never changes.
