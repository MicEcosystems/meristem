"""Cellpose-SAM segmentation backend.

Cellpose-SAM (the super-generalist model shipped in ``cellpose >= 4.0``) is a strong default for
bacterial monolayers: it is diameter-agnostic and generalizes to novel morphologies with little
or no retraining, which is exactly the "foundation model" behavior we want as the out-of-the-box
choice. Custom fine-tuned weights are supported via ``model_path``.
"""

from __future__ import annotations

from typing import Optional

from meristem.core.contracts import ImageStack, SegMasks
from meristem.core.device import Device
from meristem.core.segmentation.base import SegmenterBackend, SegmenterParams

from ._common import import_or_hint, resolve_gpu, segment_per_frame


class CellposeSAMParams(SegmenterParams):
    """Parameters for the Cellpose-SAM backend (validated; unknown keys rejected)."""

    diameter: Optional[float] = None  # None => let Cellpose-SAM estimate (it is diameter-agnostic)
    flow_threshold: float = 0.4  # max allowed flow error per mask; lower = stricter
    cellprob_threshold: float = 0.0  # raise to shrink masks / drop dim cells
    normalize: bool = True  # cellpose's internal per-image normalization
    device: Device = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    model_path: Optional[str] = None  # custom fine-tuned weights; None => pretrained cpsam


class CellposeSAMSegmenter(SegmenterBackend):
    """Wraps ``cellpose.models.CellposeModel`` (Cellpose-SAM) behind the Meristem contract."""

    name = "cellpose-sam"
    Params = CellposeSAMParams

    def load(self, params: CellposeSAMParams) -> None:  # type: ignore[override]
        self._params = params
        models = import_or_hint("cellpose.models", backend=self.name, extra="cellpose")
        self._device, gpu = resolve_gpu(params.device)
        if params.model_path:
            self._model = models.CellposeModel(gpu=gpu, pretrained_model=params.model_path)
        else:
            # Default construction loads the Cellpose-SAM super-generalist ("cpsam") weights.
            self._model = models.CellposeModel(gpu=gpu)

    def segment(self, stack: ImageStack) -> SegMasks:
        p: CellposeSAMParams = self._params

        def per_frame(img):
            # eval returns (masks, flows, styles); arity has varied across versions, so index [0].
            result = self._model.eval(
                img,
                diameter=p.diameter,
                flow_threshold=p.flow_threshold,
                cellprob_threshold=p.cellprob_threshold,
                normalize=p.normalize,
            )
            return result[0]

        return segment_per_frame(stack, per_frame, source=self.name)
