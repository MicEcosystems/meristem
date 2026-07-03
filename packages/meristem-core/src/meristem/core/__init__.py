"""meristem-core: modular, model-agnostic segmentation + tracking for bacterial monolayers.

Public API. Import backends by name through the registry; never reference a concrete backend
class from application code.
"""

from __future__ import annotations

from .config import (
    BackendConfig,
    ChannelConfig,
    PipelineConfig,
    ROIConfig,
    example_config,
)
from .contracts import ImageStack, ROI, SegMasks, TrackGraph
from .io import ChannelResult, ResultBundle, read_image_stack
from .pipeline import run_on_stack, run_pipeline, segment, track
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
    "ROIConfig",
    "example_config",
    # io
    "ResultBundle",
    "ChannelResult",
    "read_image_stack",
    # pipeline
    "run_pipeline",
    "run_on_stack",
    "segment",
    "track",
    # registry
    "register_segmenter",
    "register_tracker",
    "get_segmenter",
    "get_tracker",
    "list_segmenters",
    "list_trackers",
    "BackendNotFoundError",
]
