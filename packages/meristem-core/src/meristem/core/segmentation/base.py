"""The segmentation backend contract.

Every segmentation model — Cellpose-SAM, Omnipose, micro-sam/SAM2, StarDist, U-Net — is wrapped
as a :class:`SegmenterBackend` subclass. The contract is intentionally tiny: configure once via
:meth:`load`, then turn an :class:`~meristem.core.contracts.ImageStack` into a
:class:`~meristem.core.contracts.SegMasks`. Everything model-specific (weights, flow thresholds,
diameters) lives behind the backend's own typed ``Params`` model, so those knobs are *validated
config*, not hardcoded constants scattered through the code the way MiDAP had them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel

from ..contracts import ImageStack, SegMasks


class SegmenterParams(BaseModel):
    """Base class for a backend's parameters. Subclasses add model-specific fields.

    Using Pydantic here means a config file / UI form can be generated and validated from the
    field definitions, and unknown keys are rejected rather than silently ignored.
    """

    model_config = {"extra": "forbid"}


class SegmenterBackend(ABC):
    """Abstract base for all segmentation backends.

    Attributes
    ----------
    name:
        Registry key (set by :func:`~meristem.core.registry.register_segmenter`). Shown in the UI.
    Params:
        The Pydantic model describing this backend's parameters. Defaults to the empty
        :class:`SegmenterParams`; override in subclasses to expose model-specific options.
    """

    name: str = "unnamed"
    Params: Type[SegmenterParams] = SegmenterParams

    def __init__(self) -> None:
        self._loaded = False

    @abstractmethod
    def load(self, params: SegmenterParams) -> None:
        """Acquire weights and prepare for inference.

        Called once before :meth:`segment`. Implementations should resolve the compute device
        here (use :func:`meristem.core.device.select_device`) and cache the model on ``self``.
        """

    @abstractmethod
    def segment(self, stack: ImageStack) -> SegMasks:
        """Segment every frame of ``stack`` and return instance-labeled masks.

        Must return a :class:`SegMasks` whose ``data`` has the same ``(T, Y, X)`` shape as the
        input and integer instance labels (0 = background), with ``source`` set to ``self.name``.
        """

    # Convenience so callers don't have to track load state themselves.
    def ensure_loaded(self, params: SegmenterParams | None = None) -> None:
        if not self._loaded:
            self.load(params if params is not None else self.Params())
            self._loaded = True
