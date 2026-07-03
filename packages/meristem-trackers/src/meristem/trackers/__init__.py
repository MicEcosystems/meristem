"""Modern tracking backends for Meristem (Trackastra; more to come).

Heavy tracking libraries are imported lazily inside each backend's ``load()``, so this package is
cheap to import and the backends are discoverable through the registry
(``meristem.core.get_tracker("trackastra")``) before the libraries are installed.
"""

from .strack_tracker import STrackParams, STrackTracker
from .trackastra_tracker import TrackastraParams, TrackastraTracker

__all__ = [
    "TrackastraTracker",
    "TrackastraParams",
    "STrackTracker",
    "STrackParams",
]
