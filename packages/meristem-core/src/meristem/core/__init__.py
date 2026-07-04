"""meristem-core: modular, model-agnostic segmentation + tracking for bacterial monolayers.

Public API. Import backends by name through the registry; never reference a concrete backend
class from application code.
"""

from __future__ import annotations

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
from .measure import CellMeasurement, MeasurementTable, measure_intensities
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

__version__ = "0.0.1"

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
    # registry
    "register_segmenter",
    "register_tracker",
    "get_segmenter",
    "get_tracker",
    "list_segmenters",
    "list_trackers",
    "BackendNotFoundError",
]
