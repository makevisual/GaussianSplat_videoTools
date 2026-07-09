"""
version_check.py
================

Diagnostic utility. Run after install to verify the environment is correct.
Writes a `version_report.txt` next to where it was invoked.

This file is the future-you debugger. When something stops working in 2027,
running this and comparing the report to a known-good report will show you
exactly what changed.

Checks performed
----------------
- Python version matches the pin
- Required packages import and report expected versions
- CUDA is available to PyTorch and a GPU is visible
- SHARP install directory exists and the CLI binary is on PATH
- SHARP checkpoint exists and matches expected size
- Free disk space on the output drive

Exit codes
----------
    0 = all green
    1 = one or more warnings (install is usable but not optimal)
    2 = one or more errors (install is broken)

Standards: pure ASCII, no curly quotes or em-dashes.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# =============================================================================
# Reporting helpers
# =============================================================================

REPORT_LINES: list[str] = []
N_WARNINGS = 0
N_ERRORS   = 0


def report(line: str) -> None:
    print(line)
    REPORT_LINES.append(line)


def header(text: str) -> None:
    report("")
    report(f"=== {text} ===")


def ok(text: str) -> None:
    report(f"  [OK]    {text}")


def warn(text: str) -> None:
    global N_WARNINGS
    N_WARNINGS += 1
    report(f"  [WARN]  {text}")


def fail(text: str) -> None:
    global N_ERRORS
    N_ERRORS += 1
    report(f"  [FAIL]  {text}")


# =============================================================================
# Individual checks
# =============================================================================

def check_python_version() -> None:
    header("Python version")
    v = sys.version_info
    report(f"  Detected: {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) == (3, 13):
        ok("Python 3.13 confirmed.")
    else:
        fail(f"Expected Python 3.13.x, found {v.major}.{v.minor}.{v.micro}. "
             "Pipeline pins to 3.13 because torch wheels lag Python releases.")


def check_platform() -> None:
    header("Platform")
    report(f"  System:  {platform.system()}")
    report(f"  Release: {platform.release()}")
    report(f"  Machine: {platform.machine()}")
    if platform.system() != "Windows":
        warn("Pipeline was developed and tested on Windows. Other OSes may work but are untested.")


def check_packages() -> None:
    header("Python packages")
    expected = {
        "numpy":   None,   # version-agnostic check, just confirm importable
        "plyfile": None,
        "torch":   None,
        "imageio": None,
    }
    for pkg in expected:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            ok(f"{pkg} {ver}")
        except ImportError as e:
            fail(f"{pkg} not importable: {e}")

def check_exr_support() -> None:
    header("EXR support (openexr-numpy)")
    try:
        from openexr_numpy import imread, imwrite
        import numpy as np
        import tempfile
        import os as os_module
        # Round-trip a tiny EXR to prove openexr-numpy is actually functional.
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            tmp = f.name
        try:
            imwrite(tmp, np.zeros((4, 4, 3), dtype=np.float32))
            arr = imread(tmp)
            if arr.shape == (4, 4, 3):
                ok("EXR read/write verified via openexr-numpy.")
            else:
                fail(f"EXR round-trip produced wrong shape: {arr.shape}")
        finally:
            os_module.unlink(tmp)
    except Exception as e:
        fail(f"EXR support not functional: {e}")

def check_cuda() -> None:
    header("CUDA / GPU")
    try:
        import torch
    except ImportError:
        fail("PyTorch not installed; cannot check CUDA.")
        return

    available = torch.cuda.is_available()
    report(f"  torch.cuda.is_available: {available}")
    if not available:
        fail("CUDA not available. SHARP will fall back to CPU and be unusably slow. "
             "Likely cause: CPU-only torch wheel was installed. "
             "Reinstall via setup.ps1 to get the CUDA wheels.")
        return

    n = torch.cuda.device_count()
    report(f"  Device count: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        ok(f"GPU {i}: {name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 8.0:
            warn(f"GPU {i} has < 8GB VRAM. SHARP may OOM on full-resolution input; "
                 "consider running smaller input images.")


def check_sharp_binary() -> None:
    header("SHARP CLI")
    sharp = shutil.which("sharp")
    if sharp is None:
        fail("`sharp` not found on PATH. SHARP install may have failed, or venv not activated.")
        return
    ok(f"sharp binary: {sharp}")
    try:
        result = subprocess.run(["sharp", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            ok("sharp --help executed cleanly.")
        else:
            fail(f"`sharp --help` returned exit code {result.returncode}: {result.stderr[:200]}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        fail(f"Could not invoke `sharp --help`: {e}")


def check_checkpoint() -> None:
    header("SHARP checkpoint")
    # Look in the standard place first.
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / "checkpoints" / "sharp_2572gikvuh.pt",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "sharp_2572gikvuh.pt",
    ]
    found = None
    for p in candidate_paths:
        if p.is_file():
            found = p
            break
    if found is None:
        warn(f"Checkpoint not found in expected locations: {[str(p) for p in candidate_paths]}. "
             "SHARP will auto-download on first use, which requires internet.")
        return

    size_mb = found.stat().st_size / (1024 ** 2)
    ok(f"Checkpoint: {found} ({size_mb:.0f} MB)")
    if size_mb < 1000:
        warn(f"Checkpoint is only {size_mb:.0f} MB; expected ~2GB. May be truncated.")


def check_disk_space() -> None:
    header("Disk space")
    repo_root = Path(__file__).resolve().parents[2]
    total, used, free = shutil.disk_usage(repo_root)
    free_gb = free / (1024 ** 3)
    report(f"  Free on {repo_root.anchor}: {free_gb:.1f} GB")
    if free_gb < 10:
        fail(f"Only {free_gb:.1f} GB free. Pipeline produces large .ply files; "
             "recommend at least 50 GB free for a typical shot.")
    elif free_gb < 50:
        warn(f"Only {free_gb:.1f} GB free. May fill up on a long batch.")
    else:
        ok(f"Adequate free space ({free_gb:.1f} GB).")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    report("GaussianSplat_videoTools install report")
    report(f"Generated: {datetime.now().isoformat()}")
    report(f"Running from: {Path.cwd()}")

    check_python_version()
    check_platform()
    check_packages()
    check_cuda()
    check_sharp_binary()
    check_checkpoint()
    check_disk_space()

    header("Summary")
    if N_ERRORS == 0 and N_WARNINGS == 0:
        report("  ALL CHECKS PASSED.")
        exit_code = 0
    elif N_ERRORS == 0:
        report(f"  {N_WARNINGS} warning(s). Install is usable.")
        exit_code = 1
    else:
        report(f"  {N_ERRORS} error(s), {N_WARNINGS} warning(s). Install is BROKEN.")
        exit_code = 2

    # Write report file alongside the cwd, not the script location.
    out_path = Path.cwd() / "version_report.txt"
    try:
        out_path.write_text("\n".join(REPORT_LINES) + "\n", encoding="ascii")
        report(f"\nReport written to: {out_path}")
    except OSError as e:
        report(f"\nWARNING: could not write report to {out_path}: {e}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
