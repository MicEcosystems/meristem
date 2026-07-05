# Changelog

## v1.0.0 — 2026-07

First release of **Meristem**, a modular, model-agnostic pipeline for segmenting and tracking
bacterial cells growing as 2D monolayers in microfluidic devices. A ground-up successor to
[MiDAP](https://github.com/Microbial-Systems-Ecology/midap).

### Pipeline
`register drift → crop ROI → segment → track → measure`, runnable as one shot (`meristem run`) or as
separate, inspectable stages (`meristem segment` → look at the masks → `meristem track --frames a:b`).

### Highlights
- **Model-agnostic, swap-by-name.** Segmentation and tracking backends register via entry points;
  choosing one is a config field, not a code change. `meristem compare` runs several on the same
  input and tabulates them.
- **Segmentation backends:** Cellpose-SAM, Omnipose (with MiDAP's four bacterial models), plus a
  dependency-free `mock`.
- **Tracking backends:** Trackastra (transformer, division-aware), `strack` (native pure-NumPy port
  of MiDAP's S-track), plus a `mock` baseline.
- **Multi-channel, per-channel roles:** each channel can `segment` / `track` / `measure`. Segment on
  PH, measure GFP/RFP per cell through those masks.
- **Drift registration** (MiDAP-style: phase-correlation, translation-only, crop window follows the
  cells — no resampling).
- **MiDAP parity:** binary + instance masks, size-filter post-processing.
- **Outputs:** instance masks, binary masks, napari Tracks, CTC lineage, lineage graph,
  `measurements.csv` (per cell-frame), `track_summary.csv` (per lineage: growth rate, division,
  intensity), `manifest.json`.
- **Two interfaces:** a napari plugin with a one-click **Run** panel (`meristem-gui`) for biologists,
  and a full CLI / YAML / Python API for pipelines and batch.

### Validated on real data
End-to-end on a 217-frame *E. coli* monolayer movie (512×512 crop, Cellpose-SAM on Apple-Silicon
MPS + strack): **45 px of stage drift corrected**, colony grew **46 → 372 cells**, 72,483 detections
/ 11,550 divisions / 23,506 tracks, median specific growth rate **0.61 /hr**, ~34 min runtime.

### Quality
- 91 automated tests (3 skip without the heavy ML libraries).
- CI runs lint + tests on Python 3.9 / 3.10 / 3.11.

### Known limitations
- `strack` fragments on dense dividing colonies — prefer `trackastra`; use `compare` to check.
  Ground-truth tracking metrics (CTC TRA/DET) are not built in yet.
- Windowed `track` runs report window-local (0-based) frame indices.
- The per-cell measurement loop is per-label NumPy; a `regionprops` rewrite would speed up very
  large frames.
- Registration is translation-only (no rotation), matching MiDAP.
- The napari GUI needs a real OpenGL session (it cannot render headless).

### Install
Not yet on PyPI — install from source (see [docs/MANUAL.md](docs/MANUAL.md)) or straight from GitHub
(`git+https://…#subdirectory=…`, see [docs/PUBLISHING.md](docs/PUBLISHING.md)).

### Credits
Co-created with the [Microbial Ecosystems Lab](https://github.com/MicEcosystems) at Arizona State
University (Glen D'souza). BSD-3-Clause.
