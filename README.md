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

## Repository layout

```
packages/
  meristem-core/            # contracts, registry, pipeline, mocks (no ML deps)
  meristem-models-cellpose/ # Cellpose-SAM + Omnipose backends
  meristem-models-microsam/ # SAM2 / micro-sam backends                   (planned)
  meristem-models-classic/  # StarDist / U-Net (MiDAP parity)             (planned)
  meristem-trackers/        # Trackastra / ultrack / btrack / DeLTA       (planned)
  meristem-napari/          # napari (npe2) interactive adapter           (planned)
```

## Quickstart

```bash
pip install -e packages/meristem-core        # dependency-light core + mock backends
meristem list                                # show installed segmentation & tracking backends
meristem run experiment.yaml                 # run the pipeline from a YAML config
```

```python
from meristem.core import run_pipeline, PipelineConfig
results = run_pipeline(PipelineConfig.from_yaml("experiment.yaml"))
```

## Status

- **`meristem-core`** — complete: contracts, registry, pipeline (with a manual-crop ROI stage),
  mock backends, CLI, and tests. Runs with zero ML dependencies.
- **`meristem-models-cellpose`** — Cellpose-SAM and Omnipose segmentation backends.

Trackers (Trackastra first) and the napari plugin are next. See the design plan for the roadmap.
