# GaussianSplat_videoTools

Per-frame Gaussian-splat generation from image sequences (via Apple ML-SHARP),
plus splat cleanup utilities. Designed to feed the companion V-Ray splat
sequence loader for use in 3ds Max.

## What this is

A small, self-contained Python pipeline that:

1. Takes an image sequence (PNG, JPG, TIFF)
2. Runs Apple ML-SHARP per-frame to regress a 3D Gaussian splat (.ply) per frame
3. Cleans up the splats (cull background by brightness/opacity, optional bbox)
4. Outputs a `Frame.0001.ply`, `Frame.0002.ply`, ... sequence
5. The V-Ray `splat_sequence_loader` MAXScript (a separate repo) consumes the
   sequence in 3ds Max for shadow casting, reflections, and relighting

Tools provided (CLI entry points exposed via uv):

| Command                    | What it does                                          |
| -------------------------- | ----------------------------------------------------- |
| `gs-footage-to-splats`     | Per-frame SHARP inference on an image sequence        |
| `gs-filter`                | Cull background/transparent/oversized splats          |
| `gs-version-check`         | Diagnose the install, write a version report          |

## Requirements

- Windows 10 or 11
- NVIDIA GPU with CUDA 12.4+ driver support (RTX 3050 minimum tested)
- Git
- [uv](https://docs.astral.sh/uv/) (the installer offers a one-liner if missing)
- ~10GB free disk for the install (checkpoint + venv + SHARP repo)
- ~50GB+ free disk for typical per-shot output

## Install (on a fresh machine)

```powershell
# 1. Install uv (one-time, machine-wide)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone this repo
git clone <YOUR_TRUENAS_REPO_PATH> C:\ml\GaussianSplat_videoTools
cd C:\ml\GaussianSplat_videoTools

# 3. Run setup
.\setup.ps1
```

The setup script handles:
- Installing Python 3.13 (via uv) if not present
- Creating the venv from `uv.lock`
- Installing PyTorch with CUDA wheels
- Cloning Apple ML-SHARP at a pinned commit
- Fetching the SHARP checkpoint (~2GB) from the first available mirror
- Verifying everything with `gs-version-check`

## Quick start

After install, in a fresh PowerShell from the repo root:

```powershell
# Inference on an image sequence
uv run gs-footage-to-splats `
    --input  D:\projects\shot_01\plates\v003 `
    --output D:\projects\shot_01\splats\v001 `
    --frame-start 1001 --frame-end 1100

# Cleanup: cull dark background splats and near-transparent ones
uv run gs-filter `
    --input-dir  D:\projects\shot_01\splats\v001 `
    --output-dir D:\projects\shot_01\splats_clean\v001 `
    --min-brightness 0.15 `
    --min-opacity    0.05
```

Then in 3ds Max, use the `splat_sequence_loader` MAXScript to load the cleaned
sequence onto a `VRayGaussiansGeom` object. See that repo for details.

## Project layout

```
GaussianSplat_videoTools/
    setup.ps1                   Install script
    pyproject.toml              Project definition, dependencies
    .python-version             Pins Python 3.13
    uv.lock                     Locked dependency tree (committed)
    config/
        pipeline.toml           URLs, paths, version pins
    src/gs_videotools/
        __init__.py
        footage_to_splats.py    SHARP per-frame wrapper
        splat_filter.py         Splat cleanup utility
        coord_transforms.py     OpenCV-to-Max axis fix
        version_check.py        Install diagnostic
    docs/
        SETUP.md                Detailed install walkthrough
        TROUBLESHOOTING.md      Known issues + fixes
        PIPELINE_OVERVIEW.md    Architecture diagram
    tests/                      pytest suite
    bin/                        Convenience wrappers
    checkpoints/                Fetched by setup.ps1, NOT in Git
    external/ml-sharp/          Cloned by setup.ps1, NOT in Git
    .venv/                      Built by setup.ps1, NOT in Git
```

## Related repos

- `splat_sequence_loader` (MAXScript) - consumes the .ply sequences this
  pipeline produces. The two share a filename contract: `Frame.####.ply`.

## Standards

- Pure ASCII source files (no em-dashes, en-dashes, curly quotes). PowerShell
  on Windows Server reads scripts as Windows-1252; multi-byte UTF-8 breaks parsing.
- Minimal dependencies. PyPI for everything except SHARP itself; SHARP is
  vendored from GitHub at a pinned commit.
- Deterministic installs via `uv.lock`.
- Class/function-level docstrings; minimal inline noise.
- Console logging on every action for traceable execution on farms.
- Idempotent setup; re-running `setup.ps1` is safe.

## Versioning

This project uses semantic versioning. Tag the repo before each production run:

```powershell
git tag -a v0.2.0 -m "Used for Galderma spot"
git push origin v0.2.0
```

When something breaks 18 months later: `git checkout v0.2.0` rebuilds the
exact environment used at the time.

## License and attribution

Internal MAKE Visual tool. Uses [Apple ML-SHARP](https://github.com/apple/ml-sharp)
under its research license; commercial use of SHARP requires review of
Apple's terms.
