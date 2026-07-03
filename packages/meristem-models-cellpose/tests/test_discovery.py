"""The backends must be *discoverable and introspectable* without the heavy ML stack installed.

These tests run in an environment where cellpose/omnipose are NOT installed. They verify that the
plugin still shows up in the registry, that its parameters validate, and that trying to actually
run it fails with a helpful install hint rather than an obscure ImportError. The real inference
path is covered separately and skips itself when the model library is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from meristem.core import get_segmenter, list_segmenters
from meristem.core.contracts import ImageStack
from meristem.core.segmentation.base import SegmenterBackend
from meristem.models.cellpose import (
    MIDAP_OMNIPOSE_MODELS,
    CellposeSAMSegmenter,
    OmniposeSegmenter,
)
from meristem.models.cellpose.cellpose_sam import CellposeSAMParams
from meristem.models.cellpose.omnipose import OmniposeParams


def test_backends_registered_via_entry_points():
    seg = list_segmenters()
    assert "cellpose-sam" in seg
    assert "omnipose" in seg


@pytest.mark.parametrize(
    "name, cls", [("cellpose-sam", CellposeSAMSegmenter), ("omnipose", OmniposeSegmenter)]
)
def test_registry_returns_class_with_name(name, cls):
    resolved = get_segmenter(name)
    assert resolved is cls
    assert issubclass(resolved, SegmenterBackend)
    assert resolved.name == name


def test_params_validate_and_reject_unknown_keys():
    p = CellposeSAMParams(flow_threshold=0.2)
    assert p.flow_threshold == 0.2
    assert p.diameter is None  # diameter-agnostic default
    with pytest.raises(Exception):  # pydantic ValidationError (extra="forbid")
        CellposeSAMParams(not_a_real_option=1)


@pytest.mark.parametrize(
    "cls, extra", [(CellposeSAMSegmenter, "cellpose"), (OmniposeSegmenter, "omnipose")]
)
def test_load_without_library_gives_helpful_hint(cls, extra):
    # Skip if the heavy library happens to be installed (then load() wouldn't raise here).
    lib = "cellpose" if extra == "cellpose" else "cellpose_omni"
    if _installed(lib):
        pytest.skip(f"{lib} is installed; the missing-dependency path does not apply")
    backend = cls()
    with pytest.raises(ModuleNotFoundError) as exc:
        backend.load(cls.Params())
    msg = str(exc.value)
    assert "meristem-models-cellpose" in msg and extra in msg


def test_midap_omnipose_models_available():
    # The four bacterial models MiDAP curated must be offered, with the phase-contrast Omnipose
    # model as the default.
    assert set(MIDAP_OMNIPOSE_MODELS) == {
        "bact_phase_omni",
        "bact_fluor_omni",
        "bact_phase_cp",
        "bact_fluor_cp",
    }
    assert OmniposeParams().model_type == "bact_phase_omni"


@pytest.mark.parametrize("model", list(MIDAP_OMNIPOSE_MODELS))
def test_each_midap_model_is_a_valid_param(model):
    # Selecting any MiDAP model is a validated config change; MiDAP's inference defaults carry over.
    p = OmniposeParams(model_type=model)
    assert p.model_type == model
    assert p.mask_threshold == -1.0  # MiDAP's proven bacterial threshold
    assert p.flow_threshold == 0.0


def test_omnipose_rejects_unknown_param_key():
    with pytest.raises(Exception):  # extra="forbid"
        OmniposeParams(cluster=True)  # removed to match MiDAP's eval call


@pytest.mark.slow
def test_cellpose_sam_real_inference_if_available():
    pytest.importorskip("cellpose")
    backend = CellposeSAMSegmenter()
    backend.load(CellposeSAMParams(device="cpu"))
    stack = ImageStack(data=np.random.default_rng(0).random((2, 64, 64)).astype("float32"))
    masks = backend.segment(stack)
    assert masks.data.shape == stack.data.shape
    assert masks.source == "cellpose-sam"


def _installed(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None
