"""Backend discovery: how a segmentation model or tracker becomes selectable by name.

Two mechanisms, one registry:

1. **Decorator registration** — a backend class in *this* process calls ``@register_segmenter``
   / ``@register_tracker``. Used by the built-in mocks and convenient in notebooks/tests.
2. **Entry-point discovery** — installed backend *packages* advertise themselves under the
   ``meristem.segmenters`` / ``meristem.trackers`` entry-point groups in their ``pyproject``.
   :func:`discover_plugins` scans those lazily on first access.

Either way, the rest of the system only ever asks the registry ``get_segmenter("cellpose-sam")``
or ``list_trackers()`` — there is no ``if/elif`` on model names anywhere. Adding a model is a new
package, never an edit here.
"""

from __future__ import annotations

from importlib import metadata
from typing import Callable, Dict, List, Type, TypeVar

from .segmentation.base import SegmenterBackend
from .tracking.base import TrackerBackend

SEGMENTER_GROUP = "meristem.segmenters"
TRACKER_GROUP = "meristem.trackers"

_segmenters: Dict[str, Type[SegmenterBackend]] = {}
_trackers: Dict[str, Type[TrackerBackend]] = {}
_entry_points_loaded = False

T = TypeVar("T")


class BackendNotFoundError(KeyError):
    """Raised when a requested backend name is not registered or installed."""


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------
def register_segmenter(name: str) -> Callable[[Type[SegmenterBackend]], Type[SegmenterBackend]]:
    """Class decorator: register a :class:`SegmenterBackend` subclass under ``name``."""

    def _register(cls: Type[SegmenterBackend]) -> Type[SegmenterBackend]:
        _check_subclass(cls, SegmenterBackend, name)
        cls.name = name
        _segmenters[name] = cls
        return cls

    return _register


def register_tracker(name: str) -> Callable[[Type[TrackerBackend]], Type[TrackerBackend]]:
    """Class decorator: register a :class:`TrackerBackend` subclass under ``name``."""

    def _register(cls: Type[TrackerBackend]) -> Type[TrackerBackend]:
        _check_subclass(cls, TrackerBackend, name)
        cls.name = name
        _trackers[name] = cls
        return cls

    return _register


def _check_subclass(cls: type, base: type, name: str) -> None:
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise TypeError(
            f"cannot register {cls!r} as {base.__name__} '{name}': not a {base.__name__} subclass"
        )


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------
def discover_plugins(force: bool = False) -> None:
    """Import backends advertised by installed packages via entry points.

    Idempotent and lazy: called automatically by the accessors below on first use. Pass
    ``force=True`` to rescan (e.g. after installing a backend in a running session).
    """
    global _entry_points_loaded
    if _entry_points_loaded and not force:
        return
    for group, register in ((SEGMENTER_GROUP, register_segmenter), (TRACKER_GROUP, register_tracker)):
        for ep in _iter_entry_points(group):
            try:
                cls = ep.load()
            except Exception as exc:  # a broken backend must not sink the whole registry
                import warnings

                warnings.warn(f"failed to load backend '{ep.name}' from {group}: {exc}")
                continue
            # Entry-point name is authoritative even if the class wasn't decorated.
            register(ep.name)(cls)
    _entry_points_loaded = True


def _iter_entry_points(group: str):
    """Yield entry points for ``group`` across importlib.metadata API versions."""
    eps = metadata.entry_points()
    # Python 3.10+ returns a SelectableGroups object supporting select(); 3.9 returns a dict.
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Accessors — the only API the pipeline / CLI / napari use
# ---------------------------------------------------------------------------
def get_segmenter(name: str) -> Type[SegmenterBackend]:
    discover_plugins()
    return _lookup(_segmenters, name, "segmenter")


def get_tracker(name: str) -> Type[TrackerBackend]:
    discover_plugins()
    return _lookup(_trackers, name, "tracker")


def list_segmenters() -> List[str]:
    discover_plugins()
    return sorted(_segmenters)


def list_trackers() -> List[str]:
    discover_plugins()
    return sorted(_trackers)


def _lookup(table: Dict[str, Type[T]], name: str, kind: str) -> Type[T]:
    try:
        return table[name]
    except KeyError:
        available = ", ".join(sorted(table)) or "(none installed)"
        raise BackendNotFoundError(
            f"no {kind} named {name!r}. Available {kind}s: {available}"
        ) from None
