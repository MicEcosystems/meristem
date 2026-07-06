"""Custom model registry: named models from ~/.meristem/models.yaml wrap installed backends."""

from __future__ import annotations

import pytest

from meristem.core import get_segmenter, list_segmenters
from meristem.core import models as models_mod
from meristem.core import registry
from meristem.core.models import ModelSpec, load_model_specs, resolve_weights
from meristem.core.segmentation.mock import MockSegmenter


def test_model_spec_requires_exactly_one_source():
    ModelSpec(name="a", backend="omnipose", path="/w")  # ok
    ModelSpec(name="a", backend="omnipose", url="http://x")  # ok
    with pytest.raises(Exception):
        ModelSpec(name="a", backend="omnipose")  # neither
    with pytest.raises(Exception):
        ModelSpec(name="a", backend="omnipose", path="/w", url="http://x")  # both


def test_load_specs_from_yaml(tmp_path):
    y = tmp_path / "models.yaml"
    y.write_text("models:\n  - {name: m1, backend: omnipose, path: /w1}\n")
    specs = load_model_specs(y)
    assert len(specs) == 1 and specs[0].name == "m1" and specs[0].backend == "omnipose"
    assert load_model_specs(tmp_path / "missing.yaml") == []  # absent file -> empty


def test_resolve_weights_local_path(tmp_path):
    w = tmp_path / "weights"
    w.write_bytes(b"fake")
    assert resolve_weights(ModelSpec(name="m", backend="omnipose", path=str(w))) == w
    with pytest.raises(FileNotFoundError):
        resolve_weights(ModelSpec(name="m", backend="omnipose", path=str(tmp_path / "nope")))


def test_builtin_midap_models_offered():
    from meristem.core.models import builtin_model_specs

    names = {s.name for s in builtin_model_specs()}
    assert names == {"midap_omni_phase_v01", "midap_omni_fluor_v01"}
    for s in builtin_model_specs():
        assert s.backend == "omnipose" and s.url and "releases/download" in s.url


def test_custom_model_registers_and_wraps_backend(tmp_path, monkeypatch):
    # A custom model backed by the (dependency-free) mock segmenter, so the whole path is testable.
    weights = tmp_path / "our_weights"
    weights.write_bytes(b"fake-omnipose-weights")
    y = tmp_path / "models.yaml"
    y.write_text(
        f"models:\n  - {{name: our_phase_v2, backend: mock, path: {weights}}}\n"
    )
    monkeypatch.setattr(models_mod, "MODELS_YAML", y)

    # Re-run discovery so the custom model is picked up; clean the registry afterwards.
    registry.discover_plugins(force=True)
    try:
        assert "our_phase_v2" in list_segmenters()
        cls = get_segmenter("our_phase_v2")
        assert issubclass(cls, MockSegmenter)  # it wraps the named backend
        assert cls.name == "our_phase_v2"
        # It loads (resolving the local weights) and segments like the wrapped backend.
        from meristem.core.contracts import ImageStack
        import numpy as np

        backend = cls()
        backend.load(cls.Params())
        masks = backend.segment(ImageStack(data=np.zeros((1, 8, 8), dtype=np.float32)))
        assert masks.data.shape == (1, 8, 8)
    finally:
        registry._segmenters.pop("our_phase_v2", None)
        registry._entry_points_loaded = False
        registry.discover_plugins(force=True)
