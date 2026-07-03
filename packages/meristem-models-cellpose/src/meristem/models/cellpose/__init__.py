"""Cellpose-family segmentation backends for Meristem: Cellpose-SAM and Omnipose.

Importing this package is cheap — the heavy ML libraries are only imported when a backend's
``load()`` is called. The backends are normally reached through the registry
(``meristem.core.get_segmenter("cellpose-sam")``), not imported directly.
"""

from .cellpose_sam import CellposeSAMParams, CellposeSAMSegmenter
from .omnipose import MIDAP_OMNIPOSE_MODELS, OmniposeParams, OmniposeSegmenter

__all__ = [
    "CellposeSAMSegmenter",
    "CellposeSAMParams",
    "OmniposeSegmenter",
    "OmniposeParams",
    "MIDAP_OMNIPOSE_MODELS",
]
