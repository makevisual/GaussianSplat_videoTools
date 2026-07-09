"""
splat_filter.py
===============

Procedural .ply cleanup utility for 3D Gaussian Splat files. Filters out
unwanted splats based on:

  - Brightness (perceptual luma via Rec. 709 weights)
  - Opacity (post-sigmoid, i.e. the actual visual opacity of the splat)
  - Position (3D bounding box, inclusive or exclusive)
  - Size (log-space scale outlier culling)
  - Color match (new: cull splats whose color matches a target within
    a tolerance; used to remove greenscreen / bluescreen / any solid-color
    keyed background artifacts)

Pipeline placement
------------------
    SHARP output .ply
        |
        +--> splat_filter.py (this tool)
                |
                +--> cleaned .ply (ready for V-Ray)

Common recipes
--------------
Cull absolute black background (default; works for footage composited
onto black plate before SHARP inference):
    gs-filter --input-dir raw --output-dir clean

Cull green screen (subject shot on green):
    gs-filter --input-dir raw --output-dir clean \\
              --key-color 0.05,0.6,0.05 --key-tolerance 0.20 \\
              --no-min-brightness

Cull custom neutral gray plate:
    gs-filter --input-dir raw --output-dir clean \\
              --key-color 0.5,0.5,0.5 --key-tolerance 0.10

Standards
---------
- Pure ASCII source. No em-dashes, en-dashes, or curly quotes.
- Dependencies: numpy, plyfile (already in the venv). No new deps.
- Reads and writes standard 3DGS .ply files; preserves all per-splat
  attributes that pass the filter.
- Atomic writes: outputs go to .ply.tmp then rename on success.

Color storage in 3DGS .ply
--------------------------
3DGS files store the base color of each splat as the DC (constant) term
of a spherical harmonic expansion, in fields named f_dc_0, f_dc_1, f_dc_2
(one per RGB channel).

To convert SH DC to approximate RGB:
    SH_C0 = 0.28209479177387814
    RGB   = 0.5 + SH_C0 * f_dc

Result is in [0, 1] for normal splats but can drift outside that range
for extreme colors. We clip to [0, 1] before doing color comparisons so
tolerance thresholds are intuitive.

Opacity is stored pre-sigmoid in the 'opacity' field. To get real opacity:
    opacity_real = 1 / (1 + exp(-opacity_raw))
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


log = logging.getLogger("splat_filter")


# =============================================================================
# Math constants
# =============================================================================

SH_C0 = 0.28209479177387814


# =============================================================================
# Filter configuration
# =============================================================================

@dataclass
class FilterConfig:
    """All filter parameters in one place. None means 'skip that filter'."""

    # Brightness (Rec.709 luma) in [0, 1]
    min_brightness: Optional[float] = None
    max_brightness: Optional[float] = None

    # Opacity (post-sigmoid) in [0, 1]
    min_opacity: Optional[float] = None

    # 3D bounding box: (min_x, min_y, min_z, max_x, max_y, max_z) in ply coordinates
    bbox: Optional[tuple[float, float, float, float, float, float]] = None
    bbox_invert: bool = False   # If True, keep splats OUTSIDE the box

    # Log-space scale outlier cull (large blobs)
    max_scale: Optional[float] = None

    # Color key: cull (or optionally keep) splats within `key_tolerance` of
    # `key_color` in RGB space. key_color is (r, g, b) in [0, 1].
    # By convention we CULL matches (removing background); key_invert flips
    # to KEEP-only-matches mode (rare, but useful for isolating a color).
    key_color: Optional[tuple[float, float, float]] = None
    key_tolerance: float = 0.05    # Euclidean distance in RGB [0,1] space
    key_invert: bool = False


# =============================================================================
# Color conversion helpers
# =============================================================================

def sh_dc_to_rgb(f_dc_0: np.ndarray, f_dc_1: np.ndarray, f_dc_2: np.ndarray) -> np.ndarray:
    """Convert 3DGS SH DC coefficients to approximate RGB, clipped to [0, 1]."""
    rgb = np.stack(
        [
            0.5 + SH_C0 * f_dc_0,
            0.5 + SH_C0 * f_dc_1,
            0.5 + SH_C0 * f_dc_2,
        ],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_brightness(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma weights: Y = 0.2126 R + 0.7152 G + 0.0722 B."""
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=rgb.dtype)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


# =============================================================================
# Filtering
# =============================================================================

def build_keep_mask(v: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    """Compute a boolean mask of which splats to keep."""
    n = len(v)
    keep = np.ones(n, dtype=bool)

    # RGB is used by both brightness and color-key checks; compute once if needed.
    need_rgb = (
        cfg.min_brightness is not None
        or cfg.max_brightness is not None
        or cfg.key_color is not None
    )
    rgb = None
    if need_rgb:
        if not all(k in v.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
            log.warning("No f_dc_* fields present; skipping color-based filters.")
        else:
            rgb = sh_dc_to_rgb(v["f_dc_0"], v["f_dc_1"], v["f_dc_2"])

    # --- Brightness
    if rgb is not None and (cfg.min_brightness is not None or cfg.max_brightness is not None):
        brightness = rgb_to_brightness(rgb)
        if cfg.min_brightness is not None:
            mask = brightness >= cfg.min_brightness
            log.debug("  brightness >= %.3f: %d kept of %d", cfg.min_brightness, mask.sum(), n)
            keep &= mask
        if cfg.max_brightness is not None:
            mask = brightness <= cfg.max_brightness
            log.debug("  brightness <= %.3f: %d kept of %d", cfg.max_brightness, mask.sum(), n)
            keep &= mask

    # --- Color key
    # Euclidean distance in RGB space to the target color. This is not a
    # perceptually uniform distance (Lab or Delta-E would be more accurate)
    # but is dramatically simpler to reason about and adjust interactively.
    # For our use case (culling background), simple RGB distance is what
    # SuperSplat's Data Panel uses too, so behavior matches artist expectations.
    if rgb is not None and cfg.key_color is not None:
        target = np.array(cfg.key_color, dtype=rgb.dtype)
        dist = np.linalg.norm(rgb - target, axis=-1)
        within = dist <= cfg.key_tolerance
        if cfg.key_invert:
            # Keep-only-matches mode: retain splats INSIDE the tolerance sphere.
            mask = within
            log.debug("  key-color KEEP (dist <= %.3f): %d kept of %d",
                      cfg.key_tolerance, mask.sum(), n)
        else:
            # Default: cull matches, keep everything else.
            mask = ~within
            log.debug("  key-color CULL (dist <= %.3f): %d kept of %d",
                      cfg.key_tolerance, mask.sum(), n)
        keep &= mask

    # --- Opacity
    if cfg.min_opacity is not None:
        if "opacity" not in v.dtype.names:
            log.warning("No opacity field present; skipping opacity filter.")
        else:
            opacity_real = sigmoid(v["opacity"])
            mask = opacity_real >= cfg.min_opacity
            log.debug("  opacity >= %.3f: %d kept of %d", cfg.min_opacity, mask.sum(), n)
            keep &= mask

    # --- 3D bounding box
    if cfg.bbox is not None:
        x0, y0, z0, x1, y1, z1 = cfg.bbox
        x, y, z = v["x"], v["y"], v["z"]
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
        mask = (~inside) if cfg.bbox_invert else inside
        log.debug("  bbox (invert=%s): %d kept of %d", cfg.bbox_invert, mask.sum(), n)
        keep &= mask

    # --- Max splat scale
    if cfg.max_scale is not None:
        if not all(k in v.dtype.names for k in ("scale_0", "scale_1", "scale_2")):
            log.warning("No scale_* fields present; skipping size filter.")
        else:
            max_axis = np.maximum.reduce([v["scale_0"], v["scale_1"], v["scale_2"]])
            mask = max_axis <= cfg.max_scale
            log.debug("  max(scale_*) <= %.3f: %d kept of %d", cfg.max_scale, mask.sum(), n)
            keep &= mask

    return keep


def filter_ply(in_path: Path, out_path: Path, cfg: FilterConfig) -> tuple[int, int]:
    """Apply filter to a single .ply file. Returns (n_in, n_out)."""
    log.debug("Filtering %s -> %s", in_path.name, out_path.name)

    ply = PlyData.read(str(in_path))
    if "vertex" not in [el.name for el in ply.elements]:
        raise ValueError(f"No 'vertex' element in {in_path}")

    v = ply["vertex"].data
    n_in = len(v)
    keep = build_keep_mask(v, cfg)
    n_out = int(keep.sum())

    v_filtered = v[keep].copy()

    # Atomic write.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v_filtered, "vertex")], text=False).write(str(tmp))
    tmp.replace(out_path)

    return n_in, n_out


# =============================================================================
# Batch processing
# =============================================================================

def process_one(in_path: Path, out_path: Path, cfg: FilterConfig) -> None:
    n_in, n_out = filter_ply(in_path, out_path, cfg)
    pct = (100.0 * n_out / n_in) if n_in > 0 else 0.0
    log.info("%s: %d -> %d splats (%.1f%% kept)", in_path.name, n_in, n_out, pct)


def process_directory(in_dir: Path, out_dir: Path, cfg: FilterConfig, overwrite: bool) -> None:
    plys = sorted(in_dir.glob("*.ply"))
    if not plys:
        raise RuntimeError(f"No .ply files found in {in_dir}")

    log.info("Processing %d files from %s", len(plys), in_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in plys:
        out_path = out_dir / in_path.name
        if out_path.exists() and not overwrite:
            log.info("Skip (exists): %s", in_path.name)
            continue
        process_one(in_path, out_path, cfg)


# =============================================================================
# CLI
# =============================================================================

def parse_rgb_triple(s: str) -> tuple[float, float, float]:
    """Parse 'r,g,b' as three floats in [0, 1]. Raises ArgumentTypeError on bad input."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected 'r,g,b' (three comma-separated floats), got: {s!r}")
    try:
        rgb = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Non-numeric value in RGB triple: {s!r}")
    for name, v in zip("rgb", rgb):
        if not (0.0 <= v <= 1.0):
            raise argparse.ArgumentTypeError(f"RGB component '{name}={v}' out of range [0, 1]")
    return rgb   # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter unwanted splats from 3DGS .ply files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Cull dark background (works for footage on a black plate)
  gs-filter --input-dir raw --output-dir clean \\
            --min-brightness 0.15 --min-opacity 0.05

  # Cull green screen (subject shot on green)
  gs-filter --input-dir raw --output-dir clean \\
            --key-color 0.05,0.6,0.05 --key-tolerance 0.20

  # Cull neutral gray plate
  gs-filter --input-dir raw --output-dir clean \\
            --key-color 0.5,0.5,0.5 --key-tolerance 0.10
""",
    )

    p.add_argument("--input",      type=Path, help="Single input .ply.")
    p.add_argument("--output",     type=Path, help="Single output .ply.")
    p.add_argument("--input-dir",  type=Path, help="Input directory of .ply files.")
    p.add_argument("--output-dir", type=Path, help="Output directory for filtered .plys.")

    p.add_argument("--min-brightness", type=float, help="Keep splats with brightness >= this (0-1).")
    p.add_argument("--max-brightness", type=float, help="Keep splats with brightness <= this (0-1).")
    p.add_argument("--min-opacity",    type=float, help="Keep splats with opacity >= this (0-1, post-sigmoid).")

    p.add_argument("--key-color", type=parse_rgb_triple, metavar="R,G,B",
                   help="RGB triple to key out (e.g. 0.0,0.0,0.0 for black, 0.05,0.6,0.05 for green).")
    p.add_argument("--key-tolerance", type=float, default=0.05,
                   help="Distance in RGB [0,1] space around key color to cull (default: 0.05).")
    p.add_argument("--key-invert", action="store_true",
                   help="Invert key: KEEP matches instead of culling them (isolate a color).")

    p.add_argument("--bbox", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="Bounding box: keep splats inside this 3D box.")
    p.add_argument("--bbox-invert", action="store_true",
                   help="Invert bbox: keep splats OUTSIDE the box instead of inside.")
    p.add_argument("--max-scale", type=float,
                   help="Keep splats whose max-axis log-scale is <= this (cull large blobs).")

    p.add_argument("--overwrite", action="store_true", help="Reprocess files that already exist (dir mode).")
    p.add_argument("--verbose",   action="store_true", help="Verbose logging.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    cfg = FilterConfig(
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_opacity=args.min_opacity,
        key_color=args.key_color,
        key_tolerance=args.key_tolerance,
        key_invert=args.key_invert,
        bbox=tuple(args.bbox) if args.bbox else None,
        bbox_invert=args.bbox_invert,
        max_scale=args.max_scale,
    )

    has_single = args.input is not None and args.output is not None
    has_batch  = args.input_dir is not None and args.output_dir is not None
    if has_single == has_batch:
        log.error("Specify either --input/--output OR --input-dir/--output-dir, not both.")
        return 1

    try:
        if has_single:
            if not args.input.is_file():
                log.error("Input file not found: %s", args.input)
                return 1
            process_one(args.input, args.output, cfg)
        else:
            if not args.input_dir.is_dir():
                log.error("Input directory not found: %s", args.input_dir)
                return 1
            process_directory(args.input_dir, args.output_dir, cfg, args.overwrite)
    except Exception as exc:
        log.exception("Filter failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
