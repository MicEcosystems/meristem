"""Trackastra tracking backend.

Trackastra (Gallusser & Weigert, ECCV 2024; winner of the 7th Cell Tracking Challenge) is a
transformer that *learns* to associate segmented cells across time from their appearance and
position, rather than relying on a hand-tuned motion model. That makes it a strong, tuning-free
default for dense dividing bacterial monolayers: a pretrained model links masks out of the box,
including cell divisions, and works on top of whatever segmentation backend produced the masks.

This adapter is deliberately thin — it loads a pretrained model, runs ``model.track`` on the
image + mask stacks, and converts Trackastra's output through the shared napari-format converter
so the result is an ordinary Meristem :class:`~meristem.core.contracts.TrackGraph`.
"""

from __future__ import annotations

from typing import Literal

from meristem.core.contracts import ImageStack, SegMasks, TrackGraph
from meristem.core.device import Device, select_device
from meristem.core.tracking.base import TrackerBackend, TrackerParams

from ._common import import_or_hint, trackgraph_from_napari


class TrackastraParams(TrackerParams):
    """Parameters for the Trackastra backend."""

    # Pretrained model. "general_2d" generalizes broadly; "general_2d_w_SAM2_features" is the
    # bacteria-oriented variant (needs SAM2 features). Custom checkpoints are loadable by name/path.
    model_name: str = "general_2d"
    # Linking strategy: "greedy" (division-aware, fast), "greedy_nodiv" (no divisions), or
    # "ilp" (globally optimal via an ILP solver; requires the [trackastra-ilp] extra).
    mode: Literal["greedy", "greedy_nodiv", "ilp"] = "greedy"
    device: Device = "auto"


class TrackastraTracker(TrackerBackend):
    """Wraps Trackastra behind the Meristem tracker contract."""

    name = "trackastra"
    Params = TrackastraParams

    def load(self, params: TrackastraParams) -> None:  # type: ignore[override]
        self._params = params
        model_mod = import_or_hint("trackastra.model", backend=self.name, extra="trackastra")
        self._tracking = import_or_hint("trackastra.tracking", backend=self.name, extra="trackastra")
        device = select_device(params.device)
        self._model = model_mod.Trackastra.from_pretrained(params.model_name, device=device)

    def track(self, stack: ImageStack, masks: SegMasks) -> TrackGraph:
        p: TrackastraParams = self._params
        # Trackastra expects imgs and masks as (T, Y, X) arrays — exactly our contract's layout.
        track_graph, _masks_tracked = self._model.track(stack.data, masks.data, mode=p.mode)
        # graph_to_napari_tracks returns (data, division_graph, properties); we need the first two.
        result = self._tracking.graph_to_napari_tracks(track_graph)
        data, division_graph = result[0], result[1]
        return trackgraph_from_napari(data, division_graph)
