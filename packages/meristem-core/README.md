# meristem-core

The dependency-light core of [Meristem](../../README.md): typed data contracts, the backend
plugin registry, the pipeline orchestrator (with a manual-crop ROI stage), and mock backends.

No napari and no heavy ML frameworks are imported here, so this package runs headlessly and is
safe to depend on from anywhere. Real segmentation/tracking models live in separate backend
packages that register themselves through the `meristem.segmenters` / `meristem.trackers`
entry-point groups.

```python
from meristem.core import run_pipeline, PipelineConfig

results = run_pipeline(PipelineConfig.from_yaml("experiment.yaml"))
```
