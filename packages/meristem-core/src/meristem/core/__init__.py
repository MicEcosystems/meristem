"""meristem-core: modular, model-agnostic segmentation + tracking for bacterial monolayers.

Public API. Import backends by name through the registry; never reference a concrete backend
class from application code.
"""

from __future__ import annotations

from .batch import BatchSpec, discover_positions, run_batch
from .compare import ComparisonReport, CompareSpec, format_report, run_comparison
from .config import (
    BackendConfig,
    ChannelConfig,
    PipelineConfig,
    PostprocessConfig,
    RegisterConfig,
    ROIConfig,
    example_config,
)
from .contracts import ImageStack, ROI, SegMasks, TrackGraph
from .io import ChannelResult, ResultBundle, read_image_stack
from .measure import (
    CellMeasurement,
    MeasurementTable,
    TrackSummaryTable,
    measure_intensities,
    summarize_tracks,
)
from .models import ModelSpec, load_model_specs, resolve_weights
from .postprocess import filter_by_size
from .register import apply_shifts, crop_with_drift, estimate_drift
from .pipeline import (
    run_on_stack,
    run_pipeline,
    run_segmentation,
    run_tracking,
    segment,
    track,
)
from .registry import (
    BackendNotFoundError,
    get_segmenter,
    get_tracker,
    list_segmenters,
    list_trackers,
    register_segmenter,
    register_tracker,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # contracts
    "ImageStack",
    "ROI",
    "SegMasks",
    "TrackGraph",
    # config
    "PipelineConfig",
    "BackendConfig",
    "ChannelConfig",
    "RegisterConfig",
    "PostprocessConfig",
    "ROIConfig",
    "example_config",
    # registration
    "estimate_drift",
    "apply_shifts",
    "crop_with_drift",
    # postprocess
    "filter_by_size",
    # io
    "ResultBundle",
    "ChannelResult",
    "read_image_stack",
    # measurement
    "measure_intensities",
    "MeasurementTable",
    "CellMeasurement",
    "summarize_tracks",
    "TrackSummaryTable",
    # custom models
    "ModelSpec",
    "load_model_specs",
    "resolve_weights",
    # pipeline
    "run_pipeline",
    "run_segmentation",
    "run_tracking",
    "run_on_stack",
    "segment",
    "track",
    # compare
    "CompareSpec",
    "ComparisonReport",
    "run_comparison",
    "format_report",
    # batch
    "BatchSpec",
    "run_batch",
    "discover_positions",
    # registry
    "register_segmenter",
    "register_tracker",
    "get_segmenter",
    "get_tracker",
    "list_segmenters",
    "list_trackers",
    "BackendNotFoundError",
]
