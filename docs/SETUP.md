# Detailed setup walkthrough

For the TL;DR, see the main `README.md`. This document covers the gotchas.

## Prerequisites

### Hardware
- NVIDIA GPU. Tested on:
  - RTX 3050 mobile (4GB VRAM) - works but VRAM-constrained, slow
  - RTX A6000 (48GB VRAM) - target production hardware
- Driver supporting CUDA 12.4 or newer. Check with `nvidia-smi`; "CUDA Version"
  in the top right should be 12.4+.

### Software (manual install, one-time per machine)
- Windows 10 or 11
- Git for Windows
- uv (https://docs.astral.sh/uv/)

To install uv, run this once in PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell after install so the new PATH takes effect.

## First-time setup

```powershell
git clone <TRUENAS_REPO_PATH> C:\ml\GaussianSplat_videoTools
cd C:\ml\GaussianSplat_videoTools
.\setup.ps1
```

The setup script reports `[OK]`, `[WARN]`, or `[FAIL]` for each step.
Any `[FAIL]` aborts the install loudly.

## Verifying the install

After `setup.ps1` completes:

```powershell
uv run gs-version-check
```

This produces a `version_report.txt` in the current directory documenting the
exact Python version, package versions, GPU detected, and so on. Commit this
file alongside production work so you can diff against a future report when
something breaks.

## Common issues

### `cuda_available=False`
PyTorch CUDA wheels failed to install. Re-run setup with `-Force`:
```powershell
.\setup.ps1 -Force
```
If the issue persists, manually verify `nvidia-smi` shows a CUDA version >= 12.4.

### Checkpoint download fails on all sources
- The TrueNAS UNC path is only reachable when on the MAKE network. If working
  remote, the script falls through to the Dropbox mirror and then Apple's CDN.
- Check `config/pipeline.toml` to confirm the Dropbox link is current and uses
  `?dl=1` not `?dl=0`.
- As a last resort, manually download from Apple:
  `wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`
  Then place the file in `checkpoints\sharp_2572gikvuh.pt` and re-run setup.

### `sharp` command not found
- The venv may not be activated. Either activate it:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- Or use `uv run sharp predict ...` which runs in the project venv automatically.

### Python 3.13 not found by uv
- uv installs Python itself, so this should be rare. If it fails, manually
  install Python 3.13 from python.org and re-run setup.

## Updating SHARP to a newer commit

1. Pull the latest in `external/ml-sharp`:
   ```powershell
   cd external\ml-sharp
   git pull
   git log -1 --format="%H"   # copy this hash
   ```
2. Update `config/pipeline.toml`:
   ```toml
   upstream_commit = "<new hash>"
   ```
3. Tag the project with a new version reflecting the SHARP upgrade.

## Air-gapped install

If the target machine cannot reach the internet:

1. On a connected machine, run `setup.ps1` normally. This populates:
   - `.venv/` (the Python venv)
   - `external/ml-sharp/` (SHARP source)
   - `checkpoints/` (the 2GB model file)
2. Copy the entire repo directory (including those folders) to the offline
   machine.
3. On the offline machine, set the environment variable
   `UV_OFFLINE=1` and run `setup.ps1`. uv will only use already-downloaded
   wheels and won't try to fetch anything.

Note: this only works if the offline machine has the same OS and CPU
architecture as the connected one (Windows x64 in both cases is the
default).
