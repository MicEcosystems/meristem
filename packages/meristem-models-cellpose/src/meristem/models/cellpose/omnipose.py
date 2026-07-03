"""Omnipose segmentation backend, with MiDAP's bacterial models.

Omnipose is purpose-built for elongated, filamentous, and densely packed bacterial cells — the
regime where Cellpose's round-cell assumptions break down — so it is the natural companion default
for rod-shaped bacteria in monolayers. It runs through a cellpose fork (``cellpose-omni``) with
``omni=True``.

MiDAP curated four bacterial models for this backend, and this port exposes the same set plus the
same inference call MiDAP settled on (``mask_threshold=-1``, ``flow_threshold=0``, ``omni=True``,
``resample=True``), so results carry over for existing MiDAP users. Custom fine-tuned weights are
still loadable via ``model_path``.
"""

from __future__ import annotations

from typing import Optional

from meristem.core.contracts import ImageStack, SegMasks
from meristem.core.device import Device
from meristem.core.segmentation.base import SegmenterBackend, SegmenterParams

from ._common import import_or_hint, resolve_gpu, segment_per_frame

# The bacterial models MiDAP offers for Omnipose. The "_omni" variants use the Omnipose
# distance-field architecture; the "_cp" variants are Cellpose-architecture bacterial models that
# MiDAP nonetheless evaluates through cellpose-omni with omni=True. Phase = phase-contrast,
# fluor = fluorescence. Exposed as a tuple so the UI can populate a dropdown.
MIDAP_OMNIPOSE_MODELS: dict[str, str] = {
    "bact_phase_omni": "Bacteria, phase contrast (Omnipose) — default",
    "bact_fluor_omni": "Bacteria, fluorescence (Omnipose)",
    "bact_phase_cp": "Bacteria, phase contrast (Cellpose architecture)",
    "bact_fluor_cp": "Bacteria, fluorescence (Cellpose architecture)",
}


class OmniposeParams(SegmenterParams):
    """Parameters for the Omnipose backend.

    ``model_type`` defaults to MiDAP's phase-contrast bacterial model. Any of
    :data:`MIDAP_OMNIPOSE_MODELS` (or any other built-in Omnipose model name) is accepted; set
    ``model_path`` instead to load custom weights, which overrides ``model_type``.
    """

    model_type: str = "bact_phase_omni"  # one of MIDAP_OMNIPOSE_MODELS (or any omnipose builtin)
    mask_threshold: float = -1.0  # distance-field threshold; MiDAP uses -1 to recover thin cells
    flow_threshold: float = 0.0  # Omnipose disables the flow-error filter (0.0)
    resample: bool = True  # resample dynamics for smoother masks (MiDAP default)
    device: Device = "auto"
    model_path: Optional[str] = None  # custom weights; overrides model_type


class OmniposeSegmenter(SegmenterBackend):
    """Wraps Omnipose (via the ``cellpose-omni`` fork) behind the Meristem contract."""

    name = "omnipose"
    Params = OmniposeParams

    def load(self, params: OmniposeParams) -> None:  # type: ignore[override]
        self._params = params
        # Omnipose patches/forks cellpose; the models live in cellpose_omni.models.
        models = import_or_hint("cellpose_omni.models", backend=self.name, extra="omnipose")
        self._device, gpu = resolve_gpu(params.device)
        # Match MiDAP's construction: model_type (or a weights path) only — omni is set at eval time.
        if params.model_path:
            self._model = models.CellposeModel(gpu=gpu, pretrained_model=params.model_path)
        else:
            self._model = models.CellposeModel(gpu=gpu, model_type=params.model_type)

    def segment(self, stack: ImageStack) -> SegMasks:
        p: OmniposeParams = self._params

        def per_frame(img):
            # The exact eval call MiDAP settled on for bacterial Omnipose segmentation.
            result = self._model.eval(
                img,
                channels=[0, 0],
                rescale=None,
                mask_threshold=p.mask_threshold,
                transparency=True,
                flow_threshold=p.flow_threshold,
                omni=True,
                resample=p.resample,
                verbose=0,
            )
            return result[0]

        return segment_per_frame(stack, per_frame, source=self.name)
