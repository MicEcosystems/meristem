"""The tracking backend contract.

Every tracker — Trackastra, ultrack, btrack, DeLTA — is wrapped as a :class:`TrackerBackend`
subclass. It consumes the image stack plus the segmentation masks and returns one
:class:`~meristem.core.contracts.TrackGraph`: a lineage forest with divisions as first-class
edges. Because the output type is fixed, downstream visualization and export (napari Tracks, CTC)
are shared across all trackers, and switching trackers changes nothing downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from pydantic import BaseModel

from ..contracts import ImageStack, SegMasks, TrackGraph


class TrackerParams(BaseModel):
    """Base class for a tracker's parameters. Subclasses add tracker-specific fields."""

    model_config = {"extra": "forbid"}


class TrackerBackend(ABC):
    """Abstract base for all tracking backends."""

    name: str = "unnamed"
    Params: Type[TrackerParams] = TrackerParams

    def __init__(self) -> None:
        self._loaded = False

    @abstractmethod
    def load(self, params: TrackerParams) -> None:
        """Prepare the tracker (load any weights/config). Called once before :meth:`track`."""

    @abstractmethod
    def track(self, stack: ImageStack, masks: SegMasks) -> TrackGraph:
        """Link ``masks`` across time into a :class:`TrackGraph` lineage forest.

        ``stack`` is provided because learned trackers (e.g. Trackastra) use image appearance in
        addition to the masks; mask-only trackers may ignore it.
        """

    def ensure_loaded(self, params: TrackerParams | None = None) -> None:
        if not self._loaded:
            self.load(params if params is not None else self.Params())
            self._loaded = True
