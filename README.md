# Meristem

> A next-generation, modular pipeline for **segmenting and tracking bacterial cells growing as
> 2D monolayers in microfluidic devices**.

In plant biology a *meristem* is the tissue of continuously dividing cells that gives rise to
whole lineages — a fitting name for a tool whose job is to follow dividing bacterial cells and
reconstruct their genealogies. Meristem is a ground-up successor to
[MiDAP](https://github.com/Microbial-Systems-Ecology/midap), built around three ideas:

1. **Model-agnostic contracts.** Images, masks, and lineages have one typed representation
   (`ImageStack`, `SegMasks`, `TrackGraph`) that every backend speaks. Swapping a segmentation
   model or a tracker is a *config change*, not a code change.
2. **A plugin registry.** Segmentation and tracking backends register themselves via decorators
   and Python entry points, so a new model is a new pip package — the core never changes.
3. **A clean core / adapter split.** `meristem-core` has no GUI or heavy-ML dependencies and
   runs headlessly (CLI / notebook / cluster). Foundation and classic models live in separate
   backend packages; a napari plugin sits on top for interactive work.

**→ Full usage guide: [docs/MANUAL.md](docs/MANUAL.md)**

## Repository layout

```
packages/
  meristem-core/            # contracts, registry, pipeline, mocks (no ML deps)
  meristem-models-cellpose/ # Cellpose-SAM + Omnipose backends (MiDAP's 4 bacterial models)
  meristem-trackers/        # Trackastra + strack (native MiDAP S-track port)
  meristem-napari/          # napari (npe2) interactive plugin
```

## Quickstart

```bash
pip install -e packages/meristem-core        # dependency-light core + mock/strack backends
meristem list                                # show installed segmentation & tracking backends
meristem run experiment.yaml                 # segment + track + measure in one pass
```

```python
from meristem.core import run_pipeline, PipelineConfig
results = run_pipeline(PipelineConfig.from_yaml("experiment.yaml"))
```

## Status — v1.0

Complete and tested end-to-end on real *E. coli* monolayer movies:

- **register → crop → segment → track → measure** pipeline, as one-shot (`run`) or modular stages
  (`segment` → inspect → `track --frames`).
- Segmentation: **Cellpose-SAM**, **Omnipose** (+ MiDAP's 4 bacterial models); Tracking:
  **Trackastra**, **strack**. Swappable by name; `compare` mode to pick the best.
- Drift registration (MiDAP-style crop-follow), manual crop, per-channel segment/track/measure
  roles, size-filter cleanup, binary + instance masks.
- Outputs: masks, tracks (napari + CTC), lineage graph, `measurements.csv` (per cell-frame),
  `track_summary.csv` (per lineage: growth rate, division, intensity), `manifest.json`.
- Interactive **napari** plugin over the same functions.

See [docs/MANUAL.md](docs/MANUAL.md) for everything. Roadmap: ultrack/btrack/DeLTA trackers,
micro-sam/StarDist segmenters, CTC tracking metrics, absolute frame numbering.
