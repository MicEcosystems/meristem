# Publishing Meristem to PyPI

Once published, biologists get the true one-line install (`pip install 'meristem-napari[all]'`).
The packages already build into valid wheels + sdists (verified). This is the release runbook.

> These steps upload to a public index under **your** PyPI account. Run them yourself — they are
> outward-facing and need your credentials.

## 0. One-time setup

1. Create accounts on [PyPI](https://pypi.org/account/register/) and
   [TestPyPI](https://test.pypi.org/account/register/).
2. Create an **API token** on each (Account settings → API tokens).
3. Install the tools (in any env): `pip install build twine` — or use `uv build` / `uv publish`.

**Name availability:** the distributions are `meristem-core`, `meristem-models-cellpose`,
`meristem-trackers`, `meristem-napari`. Check each is free at `https://pypi.org/project/<name>/`
(404 = available). If any is taken, rename it in that package's `pyproject.toml` (and update the
sibling `dependencies`).

## 1. Build all packages

```bash
cd path/to/Phusion
rm -rf dist
for pkg in meristem-core meristem-models-cellpose meristem-trackers meristem-napari; do
  python -m build packages/$pkg --outdir dist      # or: uv build packages/$pkg --out-dir dist
done
twine check dist/*                                  # validates metadata/README rendering
```

## 2. Dry-run on TestPyPI

Publish **core first** — the others declare `meristem-core` as a dependency, so it must exist on the
index before they resolve.

```bash
twine upload --repository testpypi dist/meristem_core-*
twine upload --repository testpypi dist/meristem_models_cellpose-* dist/meristem_trackers-* dist/meristem_napari-*

# verify a clean install from TestPyPI (falling back to real PyPI for third-party deps):
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ 'meristem-napari[all]'
meristem-gui
```

## 3. Real release

```bash
twine upload dist/meristem_core-*
twine upload dist/meristem_models_cellpose-* dist/meristem_trackers-* dist/meristem_napari-*
```

Then the biologist install is genuinely:

```bash
pip install 'meristem-napari[all]'
meristem-gui
```

## 4. Future releases

1. Bump `version` in every changed package's `pyproject.toml` (and `__version__`). Keep the four in
   lockstep for a coordinated release; PyPI forbids re-uploading an existing version.
2. Rebuild, `twine check`, upload (core first if its version changed).
3. Tag it: `git tag -a vX.Y.Z -m "..." && git push --tags`.

## Alternative: install from GitHub (no PyPI)

If the repo is pushed to GitHub, users can install directly from it without PyPI — useful for a
private/early release:

```bash
pip install "meristem-core @ git+https://github.com/<you>/meristem#subdirectory=packages/meristem-core"
pip install "meristem-napari[gui] @ git+https://github.com/<you>/meristem#subdirectory=packages/meristem-napari"
```
