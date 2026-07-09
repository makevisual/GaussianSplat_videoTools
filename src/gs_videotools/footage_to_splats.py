"""
footage_to_splats.py
====================

Per-frame Gaussian splat generation from an image sequence using Apple's
ML-SHARP single-image 3DGS regressor.

Pipeline overview
-----------------
    image_sequence (input, any of: exr, png, jpg, tif)
        |
        +--> EXR-to-PNG conversion if needed (via imageio)
                |
                +--> SHARP inference (one image -> one .ply)
                        |
                        +--> Coordinate transform (OpenCV -> 3ds Max Z-up)
                                |
                                +--> Numbered .ply output (Frame.0001.ply, ...)

EXR handling
------------
SHARP cannot read EXR directly (its Pillow-based loader lacks OpenEXR
support). We convert to 16-bit PNG in a temp folder before invoking
SHARP. By default the EXR is assumed to be already display-referred
(view-transformed) and no color transform is applied on the way to PNG.
For linear scene-referred EXR sources, use --linear-to-srgb.

Design philosophy
-----------------
- Standard library + numpy + plyfile + imageio. imageio adds EXR read
  support with imageio-freeimage on Windows; still a small dep footprint.
- Class-level documentation; minimal inline comments.
- Console prints at every phase for traceable execution.
- Robustness over performance. Single-process, sequential. Parallelism
  can be added later by submitting multiple invocations to a farm.
- Atomic writes: outputs to *.ply.tmp then rename on success.
- Idempotent: re-running on the same input range will skip frames that
  already exist unless --overwrite is set.

Standards
---------
Pure ASCII. No em-dashes, en-dashes, curly quotes.

Authored: MAKE pipeline, 2026-07
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from plyfile import PlyData, PlyElement


# =============================================================================
# Logging
# =============================================================================

def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


log = logging.getLogger("footage_to_splats")


# =============================================================================
# Coordinate transform
# =============================================================================
#
# SHARP outputs splats in OpenCV camera convention:
#     +X right, +Y down, +Z forward
#
# 3ds Max world convention:
#     +X right, +Y forward, +Z up
#
# Transform:
#     x_max =  x_opencv
#     y_max =  z_opencv
#     z_max = -y_opencv
#
# Positions and rotations (as quaternions) both rotate. SH DC coefficients
# and opacity do not. Higher-order SH bands are technically view-direction
# dependent and would need SH basis rotation for correctness; we leave them
# untouched, accepting small color-shift artifacts in exchange for simpler
# code. This matches how SuperSplat and most 3DGS viewers handle ad-hoc
# rotations.

R_OPENCV_TO_MAX = np.array(
    [
        [1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0],
    ],
    dtype=np.float32,
)


def quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to unit quaternion (w, x, y, z), Shoemake-stable."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2. Arrays shaped (..., 4) in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


@dataclass
class TransformConfig:
    """Coordinate transform stage config."""
    scale: float = 1.0
    rotate_to_max: bool = True
    recenter: bool = False


def transform_ply(in_path: Path, out_path: Path, cfg: TransformConfig) -> None:
    log.debug("Transforming %s -> %s", in_path.name, out_path.name)

    ply = PlyData.read(str(in_path))
    if "vertex" not in [el.name for el in ply.elements]:
        raise ValueError(f"No 'vertex' element in {in_path}; not a 3DGS .ply?")

    v = ply["vertex"]
    n = len(v.data)
    log.debug("  %d Gaussians", n)

    pos = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    if cfg.rotate_to_max:
        pos = pos @ R_OPENCV_TO_MAX.T
    if cfg.recenter:
        pos = pos - pos.mean(axis=0, keepdims=True)
    if cfg.scale != 1.0:
        pos = pos * cfg.scale

    v["x"] = pos[:, 0]
    v["y"] = pos[:, 1]
    v["z"] = pos[:, 2]

    if cfg.rotate_to_max and all(f"rot_{i}" in v.data.dtype.names for i in range(4)):
        q_local = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
        ).astype(np.float32)
        norms = np.linalg.norm(q_local, axis=-1, keepdims=True)
        q_local = q_local / np.maximum(norms, 1e-8)

        q_world = quat_from_rotmat(R_OPENCV_TO_MAX)
        q_world_b = np.broadcast_to(q_world, q_local.shape).copy()
        q_new = quat_mul(q_world_b, q_local)

        v["rot_0"] = q_new[:, 0]
        v["rot_1"] = q_new[:, 1]
        v["rot_2"] = q_new[:, 2]
        v["rot_3"] = q_new[:, 3]

    if cfg.scale != 1.0:
        log_scale = np.log(cfg.scale)
        for k in ("scale_0", "scale_1", "scale_2"):
            if k in v.data.dtype.names:
                v[k] = v[k] + log_scale

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    PlyData([PlyElement.describe(v.data, "vertex")], text=False).write(str(tmp))
    tmp.replace(out_path)


# =============================================================================
# EXR / image conversion
# =============================================================================
#
# SHARP's loader is Pillow-based and does not handle EXR. We pre-convert to
# 16-bit PNG in a temp folder. By default no color transform is applied
# (assumes EXR is already display-referred). --linear-to-srgb applies a
# standard sRGB gamma if raw linear EXR is provided.

EXR_EXTENSIONS      = (".exr",)
NATIVE_EXTENSIONS   = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
INPUT_EXTENSIONS    = EXR_EXTENSIONS + NATIVE_EXTENSIONS

def convert_exr_to_png(exr_path: Path, png_path: Path, linear_to_srgb: bool) -> None:
    """Read an EXR via openexr-numpy, apply optional gamma, write 16-bit PNG.

    Uses the ASWF OpenEXR Python bindings (via openexr-numpy) for EXR reading,
    which has Python 3.13 wheels and does not require a separate binary
    download step. imageio still handles PNG writing.
    """
    try:
        from openexr_numpy import imread as exr_imread
        import imageio.v3 as iio
    except ImportError as e:
        raise RuntimeError(
            "openexr-numpy and imageio are required to read EXR inputs. "
            "Reinstall via setup.ps1.\n"
            f"Underlying error: {e}"
        )

    log.debug("EXR -> PNG: %s", exr_path.name)
    img = exr_imread(str(exr_path))   # float32 HxWx3 by default

    # Handle alpha: drop it. SHARP ignores alpha, and PNGs with alpha can
    # confuse SHARP's input assumption of RGB.
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    # Optional linear-to-sRGB. Standard piecewise sRGB curve.
    if linear_to_srgb:
        img = np.clip(img, 0.0, None)
        a = 0.055
        low = img <= 0.0031308
        high = ~low
        out = np.empty_like(img)
        out[low] = img[low] * 12.92
        out[high] = (1.0 + a) * np.power(img[high], 1.0 / 2.4) - a
        img = out

    # Clip and quantize to 16-bit unsigned.
    img = np.clip(img, 0.0, 1.0)
    img_16 = (img * 65535.0 + 0.5).astype(np.uint16)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(png_path, img_16)   # PNG write via imageio's built-in Pillow path

def ensure_sharp_readable(input_path: Path, temp_dir: Path, linear_to_srgb: bool) -> Path:
    """Return a path SHARP can read. Converts EXR to PNG if needed.

    For non-EXR inputs, returns the input path unchanged (no copy).
    """
    if input_path.suffix.lower() in EXR_EXTENSIONS:
        png_path = temp_dir / (input_path.stem + ".png")
        convert_exr_to_png(input_path, png_path, linear_to_srgb)
        return png_path
    return input_path


# =============================================================================
# SHARP inference
# =============================================================================
#
# SHARP is wrapped as a subprocess call to its `sharp predict` CLI:
#   1. Stable public interface less likely to change than the internal API
#   2. Subprocess isolation: a bad frame doesn't kill the whole batch
#   3. Easier to swap for a different single-image splat regressor later.


@dataclass
class InferenceConfig:
    """SHARP inference config."""
    python_exe: Path
    checkpoint_path: Optional[Path] = None   # None means SHARP uses its own cache lookup
    sharp_module: str = "sharp.predict"
    device: str = "cuda"


def run_sharp_on_image(image_path: Path, out_ply: Path, cfg: InferenceConfig) -> None:
    """Invoke SHARP on a single image, writing one .ply to out_ply."""
    log.debug("Running SHARP: %s -> %s", image_path.name, out_ply.name)

    with tempfile.TemporaryDirectory(prefix="sharp_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        cmd = [
            str(cfg.python_exe),
            "-m", cfg.sharp_module,
            "-i", str(image_path),
            "-o", str(tmpdir_path),
            "--device", cfg.device,
        ]
        if cfg.checkpoint_path is not None:
            cmd += ["-c", str(cfg.checkpoint_path)]

        log.debug("  command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("SHARP failed on %s", image_path.name)
            log.error("stdout: %s", result.stdout)
            log.error("stderr: %s", result.stderr)
            raise RuntimeError(f"SHARP returned exit code {result.returncode}")

        candidates = list(tmpdir_path.glob("*.ply"))
        if not candidates:
            raise RuntimeError(
                f"SHARP produced no .ply in {tmpdir_path}. "
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        if len(candidates) > 1:
            log.warning("SHARP produced %d .plys; using first: %s", len(candidates), candidates[0].name)
        shutil.copy2(candidates[0], out_ply)


# =============================================================================
# Frame iteration
# =============================================================================

def iter_input_frames(
    input_dir: Path,
    frame_start: Optional[int],
    frame_end: Optional[int],
) -> Iterator[tuple[int, Path]]:
    """Yield (frame_number, image_path) pairs in frame order."""
    import re
    pat = re.compile(r"(\d+)(?=\.[^.]+$)")

    frames: list[tuple[int, Path]] = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() not in INPUT_EXTENSIONS:
            continue
        m = pat.search(p.name)
        if not m:
            log.warning("Skipping %s: no frame number in filename", p.name)
            continue
        n = int(m.group(1))
        if frame_start is not None and n < frame_start:
            continue
        if frame_end is not None and n > frame_end:
            continue
        frames.append((n, p))

    if not frames:
        raise RuntimeError(f"No input frames found in {input_dir}")

    log.info("Found %d input frames (range %d to %d)", len(frames), frames[0][0], frames[-1][0])
    yield from frames


# =============================================================================
# Main pipeline
# =============================================================================

def output_path_for_frame(output_dir: Path, frame: int, basename: str = "Frame") -> Path:
    return output_dir / f"{basename}.{frame:04d}.ply"


def process_sequence(
    input_dir: Path,
    output_dir: Path,
    inference: InferenceConfig,
    transform: TransformConfig,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    overwrite: bool = False,
    basename: str = "Frame",
    linear_to_srgb: bool = False,
) -> None:
    """Run the full pipeline over an image sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_skipped = 0
    t_start = time.time()

    # Shared temp dir for EXR conversions; created once per batch,
    # cleaned up on completion (or on exception).
    with tempfile.TemporaryDirectory(prefix="footage_to_splats_") as temp_root:
        temp_dir = Path(temp_root)

        for frame, image_path in iter_input_frames(input_dir, frame_start, frame_end):
            out_path = output_path_for_frame(output_dir, frame, basename)
            if out_path.exists() and not overwrite:
                log.info("Frame %04d: skip (exists). Use --overwrite to regenerate.", frame)
                n_skipped += 1
                continue

            log.info("Frame %04d: %s", frame, image_path.name)
            t0 = time.time()

            # If EXR, convert to PNG in the shared temp dir. Otherwise use path directly.
            sharp_input = ensure_sharp_readable(image_path, temp_dir, linear_to_srgb)

            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_ply:
                sharp_out = Path(tmp_ply.name)

            try:
                run_sharp_on_image(sharp_input, sharp_out, inference)
                transform_ply(sharp_out, out_path, transform)
            finally:
                if sharp_out.exists():
                    sharp_out.unlink()

            n_processed += 1
            log.info("  done in %.1fs -> %s", time.time() - t0, out_path.name)

    elapsed = time.time() - t_start
    log.info(
        "Pipeline complete: %d processed, %d skipped, %.1fs total",
        n_processed, n_skipped, elapsed,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-frame Gaussian splat generation from an image sequence (SHARP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",      required=True, type=Path, help="Input image sequence directory.")
    p.add_argument("--output",     required=True, type=Path, help="Output .ply sequence directory.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="SHARP checkpoint path. If omitted, SHARP uses its default cache lookup.")
    p.add_argument("--python-exe", type=Path, default=None,
                   help="Path to SHARP venv python.exe. Defaults to the current interpreter.")
    p.add_argument("--frame-start", type=int, default=None, help="First frame number (inclusive).")
    p.add_argument("--frame-end",   type=int, default=None, help="Last frame number (inclusive).")
    p.add_argument("--basename", default="Frame", help="Output filename prefix (default: Frame).")
    p.add_argument("--scale", type=float, default=1.0, help="Uniform scale factor on output.")
    p.add_argument("--no-axis-flip", action="store_true",
                   help="Skip OpenCV-to-Max axis flip (use V-Ray loader's flip_axis toggle instead).")
    p.add_argument("--recenter", action="store_true", help="Recenter splat centroid to origin.")
    p.add_argument("--linear-to-srgb", action="store_true",
                   help="Apply sRGB gamma to EXR inputs (use only if EXR is raw linear/scene-referred).")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"), help="Inference device.")
    p.add_argument("--overwrite", action="store_true", help="Reprocess frames that already exist.")
    p.add_argument("--verbose", action="store_true", help="Verbose (DEBUG-level) logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.input.is_dir():
        log.error("Input directory does not exist: %s", args.input)
        return 1
    if args.checkpoint is not None and not args.checkpoint.is_file():
        log.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    python_exe = args.python_exe or Path(sys.executable)

    inference = InferenceConfig(
        python_exe=python_exe,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    transform = TransformConfig(
        scale=args.scale,
        rotate_to_max=not args.no_axis_flip,
        recenter=args.recenter,
    )

    try:
        process_sequence(
            input_dir=args.input,
            output_dir=args.output,
            inference=inference,
            transform=transform,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            overwrite=args.overwrite,
            basename=args.basename,
            linear_to_srgb=args.linear_to_srgb,
        )
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
