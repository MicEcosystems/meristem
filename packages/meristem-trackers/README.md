# meristem-trackers

Modern cell-tracking backends for [Meristem](../../README.md).

- **`trackastra`** — [Trackastra](https://github.com/weigertlab/trackastra), a transformer that
  learns cell-to-cell association across time. Tuning-free, division-aware, and decoupled from
  segmentation (it links whatever masks you give it). Winner of the 7th Cell Tracking Challenge
  (ISBI 2024). The recommended modern default for dividing bacterial monolayers.
- **`strack`** — a native, pure-NumPy port of MiDAP's S-track: overlap-first greedy matching,
  distance-gated, accepting a division only when the daughters separate along the mother's long
  axis. Fast, dependency-free, and works out of the box — kept for continuity with MiDAP workflows.

Registers with the Meristem tracker registry via the `meristem.trackers` entry-point group, so it
appears anywhere trackers are listed (CLI, napari) once installed.

## What's new vs. MiDAP's trackers

MiDAP shipped btrack (Bayesian motion model), DeLTA (frame-to-frame U-Net), and S-track. Trackastra
differs in three ways that matter for dense, dividing bacteria:

1. **Learned association, not a hand-tuned model.** A transformer attends over cell appearance and
   position across a temporal window to predict links — no `btrack_conf.json` motion parameters to
   tune, and no per-dataset retraining like DeLTA.
2. **Tuning-free generalization.** Pretrained models (including a bacteria-oriented variant) work
   out of the box; you can start tracking without training anything.
3. **Segmentation-agnostic by design.** It links the masks from *any* Meristem segmentation
   backend, so Cellpose-SAM or Omnipose masks flow straight into it.

Output is a Meristem `TrackGraph` (lineage forest with first-class divisions), which exports to
napari Tracks and the Cell Tracking Challenge format.

## Install

```bash
pip install 'meristem-trackers[trackastra]'       # greedy / greedy_nodiv modes
pip install 'meristem-trackers[trackastra-ilp]'   # + globally-optimal ILP mode (needs Gurobi/SCIP)
```

The wrapper itself is dependency-light; Trackastra (and torch) load lazily when the backend runs.

### Device / GPU

Trackastra runs on `cpu` (default), `cuda`, or `mps`. It's a lightweight linker, so **CPU is
usually fastest at monolayer scale** — on Apple Silicon MPS is often *slower* (partial Metal op
coverage + transfer overhead), and only large workloads on CUDA clearly benefit. Set
`tracker.params.device` to choose. When you request `mps`, the backend enables
`PYTORCH_ENABLE_MPS_FALLBACK` automatically so the GPU is actually used instead of silently falling
back to CPU.

## Use

```python
from meristem.core import PipelineConfig, BackendConfig, run_pipeline

cfg = PipelineConfig(
    input={"path": "movie.tif", "pixel_size_um": 0.065},
    segmenter=BackendConfig(name="omnipose"),
    tracker=BackendConfig(name="trackastra", params={"mode": "greedy"}),
)
run_pipeline(cfg)
```
