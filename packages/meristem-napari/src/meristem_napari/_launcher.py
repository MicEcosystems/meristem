"""`meristem-gui` — the one-command, biologist-facing launcher.

Opens napari with the guided "Run pipeline" panel already docked, so the whole workflow is: run
one command, drag in an image, pick a segmenter and tracker from the dropdowns, press Run. No YAML,
no CLI. The step-by-step Segment / Track widgets are docked too for the inspect-then-track workflow.
"""

from __future__ import annotations


def main() -> None:
    import napari

    from ._widgets import run_widget, segment_widget, track_widget

    viewer = napari.Viewer(title="Meristem")
    viewer.window.add_dock_widget(run_widget(), name="Run pipeline", area="right")
    viewer.window.add_dock_widget(segment_widget(), name="1. Segment", area="right")
    viewer.window.add_dock_widget(track_widget(), name="2. Track", area="right")
    napari.run()


if __name__ == "__main__":  # pragma: no cover
    main()
