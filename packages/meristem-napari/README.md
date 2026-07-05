# meristem-napari

A [napari](https://napari.org) plugin for the [Meristem](../../README.md) pipeline — the interactive
home for the *crop → segment → inspect → track* workflow.

Three dock widgets:

0. **Run pipeline** — the one-click, no-YAML entry point: pick a segmentation model and a tracker
   from dropdowns, (optionally) draw a crop rectangle, press Run. Masks + tracks appear as layers.
1. **Segment** — segment only, into a Labels layer you can scroll through and evaluate.
2. **Track + measure** — once the segmentation looks good, pick a tracker and a frame window and
   link the masks; the lineage appears as a Tracks layer.

## For biologists

```bash
pip install 'meristem-napari[all]'   # GUI + real segmentation/tracking models
meristem-gui                         # opens napari with the Run panel docked
```

Then drag in a TIFF, pick the dropdowns, press Run.

Both widgets are thin front-ends over `meristem.core` — the same functions the CLI uses — so the GUI
and headless runs behave identically.

## Install

```bash
pip install 'meristem-napari[gui]'   # pulls napari + magicgui
napari                               # then: Plugins -> Meristem
```

The package's non-GUI helpers import without a Qt stack; only the widgets require napari/magicgui.

## Status

Widgets and npe2 manifest are in place. GUI interaction is validated manually / with napari's
`make_napari_viewer` fixture; the napari-free helpers and the manifest are covered by unit tests.
