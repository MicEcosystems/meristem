"""napari plugin for Meristem — a front-end over meristem.core's segment/track/measure functions.

The GUI code (widgets, reader) imports napari/magicgui lazily; the napari-free helpers in
``_core`` are importable and testable without a Qt stack.
"""

__version__ = "1.0.0"
