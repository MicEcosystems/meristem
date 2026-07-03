# meristem-models-cellpose

Cellpose-family segmentation backends for [Meristem](../../README.md):

- **`cellpose-sam`** — Cellpose-SAM (super-generalist, `cellpose >= 4.0`). Diameter-agnostic,
  strong on novel bacterial morphologies with little/no retraining. The recommended default.
- **`omnipose`** — Omnipose (`cellpose-omni`). Purpose-built for elongated, filamentous, and
  densely packed bacteria — the rod-shaped-monolayer case Cellpose handles less well. Ships the
  four bacterial models MiDAP curated, using MiDAP's proven inference settings:

  | `model_type` | Imaging | Architecture |
  |---|---|---|
  | `bact_phase_omni` *(default)* | phase contrast | Omnipose |
  | `bact_fluor_omni` | fluorescence | Omnipose |
  | `bact_phase_cp` | phase contrast | Cellpose |
  | `bact_fluor_cp` | fluorescence | Cellpose |

  Any other built-in Omnipose model name also works, or point `model_path` at custom weights.

Both register with the Meristem registry via the `meristem.segmenters` entry-point group, so once
installed they appear anywhere backends are listed (CLI, napari dropdown) with no code changes.

## Install

```bash
pip install 'meristem-models-cellpose[cellpose]'   # Cellpose-SAM
pip install 'meristem-models-cellpose[omnipose]'   # Omnipose
pip install 'meristem-models-cellpose[all]'        # both
```

The package itself is dependency-light: the heavy ML stacks (torch, cellpose, omnipose) are
imported lazily inside each backend's `load()`, so the backends are discoverable — and their
parameters introspectable — even before the model libraries are installed.

## Use

```python
from meristem.core import PipelineConfig, BackendConfig, run_pipeline

cfg = PipelineConfig(
    input={"path": "movie.tif", "pixel_size_um": 0.065},
    segmenter=BackendConfig(name="omnipose", params={"model_type": "bact_phase_omni"}),
    tracker=BackendConfig(name="mock"),
)
run_pipeline(cfg)
```

Switching to Cellpose-SAM is a one-line change: `name="cellpose-sam"`.
