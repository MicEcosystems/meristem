"""Custom model registry — make trained model weights selectable by name.

MiDAP shipped custom Omnipose models (better Omnipose trained on the lab's data) and let you pick
them from a dropdown. This restores that: you list your models once in ``~/.meristem/models.yaml``,
and each becomes a named segmenter in the registry (and the napari dropdown), wrapping an installed
backend (e.g. ``omnipose``) with your weights.

```yaml
# ~/.meristem/models.yaml
models:
  - name: midap_omni_phase_v01        # appears as a segmenter you can select
    backend: omnipose                 # which installed backend runs it
    path: ~/.meristem/models/midap_omni_phase_v01   # local weights file
  - name: our_phase_v2
    backend: omnipose
    url: https://.../our_phase_v2     # downloaded + cached on first use
    version: v2                       # re-download only when this changes
```

Weights are resolved lazily (only when the model is actually run), downloaded once, and cached under
``~/.meristem/cache``. This module has no registry dependency — the registry imports it.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, model_validator

MERISTEM_HOME = Path.home() / ".meristem"
MODELS_YAML = MERISTEM_HOME / "models.yaml"
CACHE_DIR = MERISTEM_HOME / "cache"


class ModelSpec(BaseModel):
    """One named custom model: a backend + a weights source (local path or URL)."""

    model_config = {"extra": "forbid"}

    name: str
    backend: str  # a registered segmenter name, e.g. "omnipose" or "cellpose-sam"
    path: Optional[str] = None  # local weights file
    url: Optional[str] = None  # download URL (a weights file, or a zip + `subpath`)
    subpath: Optional[str] = None  # member to extract when `url` points at a zip
    version: Optional[str] = None  # bump to force a re-download

    @model_validator(mode="after")
    def _one_source(self) -> "ModelSpec":
        if bool(self.path) == bool(self.url):
            raise ValueError(f"model {self.name!r}: set exactly one of 'path' or 'url'")
        return self


def load_model_specs(yaml_path: Optional[Path] = None) -> List[ModelSpec]:
    """Read the user's model list (``~/.meristem/models.yaml``); empty if the file is absent."""
    path = yaml_path or MODELS_YAML
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    return [ModelSpec.model_validate(m) for m in raw.get("models", [])]


def resolve_weights(spec: ModelSpec) -> Path:
    """Return a local path to the model's weights, downloading + caching if needed."""
    if spec.path:
        p = Path(spec.path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"model {spec.name!r}: weights not found at {p}")
        return p
    return _download_cached(spec)


def _download_cached(spec: ModelSpec) -> Path:
    dest_dir = CACHE_DIR / spec.name
    marker = dest_dir / ".version"
    weights = dest_dir / "weights"
    if weights.exists() and marker.exists() and marker.read_text().strip() == (spec.version or ""):
        return weights

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / "download.tmp"
    with urllib.request.urlopen(spec.url) as r, open(tmp, "wb") as fh:  # noqa: S310 (trusted config)
        shutil.copyfileobj(r, fh)

    if spec.subpath:  # the URL is a zip; extract the requested member
        with zipfile.ZipFile(tmp) as z:
            with z.open(spec.subpath) as src, open(weights, "wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp.unlink()
    else:
        tmp.replace(weights)

    marker.write_text(spec.version or "")
    return weights
