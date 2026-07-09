"""
run_pipeline.py
===============

End-to-end orchestrator for the video-to-splat pipeline. Runs
footage_to_splats (SHARP inference) followed by splat_filter (cleanup)
in a single invocation, using the same Python venv.

This is the "quality of life" wrapper. footage_to_splats.py and
splat_filter.py remain independently usable as CLI tools. This script
just wires them together with sensible defaults for the common case.

Design philosophy
-----------------
Rather than shelling out to the CLI entry points, this script imports
the two modules directly and calls their public functions. Reasons:

  1. Fewer subprocess handoffs -> faster, less overhead per frame
  2. Better error handling -> Python exceptions propagate naturally
  3. Single log stream -> everything appears in one console
  4. Easier to edit -> you can copy this file, tweak the parameters
     in the CONFIG section, and produce a customized runner for a
     specific shot without touching the underlying modules.

Typical use
-----------
Edit the CONFIG section below to point at your inputs, then:
    uv run python run_pipeline.py

Or from the CLI (overrides CONFIG):
    uv run python run_pipeline.py \\
        --input   tests/DemoFootage_002 \\
        --output  tests/DemoFootage_002_splats \\
        --key-color 0.0,0.0,0.0 \\
        --key-tolerance 0.05

Standards
---------
Pure ASCII. No em-dashes, en-dashes, curly quotes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Import our two pipeline modules as libraries.
# When installed via `uv pip install -e .` (as setup.ps1 does), these
# resolve through the gs_videotools package. If running this file
# standalone alongside the modules in the same folder, the imports also
# work via Python's implicit cwd behavior.
from gs_videotools import footage_to_splats as fts
from gs_videotools import splat_filter as sf


log = logging.getLogger("run_pipeline")


# =============================================================================
# CONFIG (edit this block to change defaults)
# =============================================================================
# CLI arguments override anything set here. The intent is that you copy
# this file to a shot-specific version (e.g. run_pipeline_galderma.py)
# and edit these values for that shot's specific needs, then re-run
# without CLI arguments. This gives you a paper trail of what parameters
# were used for each production run.

CONFIG = {
    # -- Input / output paths --
    "input_dir":       Path("tests/DemoFootage_001"),
    "output_dir":      Path("tests/DemoFootage_001_splats"),
    "cleaned_dir":     Path("tests/DemoFootage_001_splats_clean"),

    # -- Frame range (None = process all frames in input_dir) --
    "frame_start":     None,
    "frame_end":       None,

    # -- SHARP inference options --
    "device":          "cuda",         # "cuda" or "cpu" or "mps"
    "linear_to_srgb":  False,          # True only for raw linear EXR sources
    "checkpoint":      None,           # None = SHARP's default cache lookup
    "output_basename": "Frame",        # Frame.0001.ply etc.
    "no_axis_flip":    False,          # True = skip Python-side coord flip (rely on V-Ray toggle)
    "recenter":        False,          # True = translate splat centroid to origin
    "scale":           1.0,

    # -- Splat filter options --
    "filter_enabled":  True,
    # Color key (RGB in [0, 1]). Default is absolute black background.
    "key_color":       (0.0, 0.0, 0.0),
    "key_tolerance":   0.05,   # Euclidean distance in RGB space
    "key_invert":      False,  # True = keep only matches (isolate that color)
    # Additional filters. Set to None to disable.
    "min_brightness":  None,
    "max_brightness":  None,
    "min_opacity":     0.05,
    "bbox":            None,
    "bbox_invert":     False,
    "max_scale":       None,

    # -- Behavior --
    "overwrite":       False,
    "verbose":         False,
}


# =============================================================================
# Stage runners
# =============================================================================

def run_inference_stage(cfg: dict) -> None:
    """Stage 1: SHARP per-frame inference from input images to raw .ply sequence."""
    log.info("=" * 70)
    log.info("STAGE 1: SHARP inference")
    log.info("  input:  %s", cfg["input_dir"])
    log.info("  output: %s", cfg["output_dir"])
    log.info("=" * 70)

    python_exe = Path(sys.executable)

    inference_cfg = fts.InferenceConfig(
        python_exe=python_exe,
        checkpoint_path=cfg["checkpoint"],
        device=cfg["device"],
    )
    transform_cfg = fts.TransformConfig(
        scale=cfg["scale"],
        rotate_to_max=not cfg["no_axis_flip"],
        recenter=cfg["recenter"],
    )

    t0 = time.time()
    fts.process_sequence(
        input_dir=cfg["input_dir"],
        output_dir=cfg["output_dir"],
        inference=inference_cfg,
        transform=transform_cfg,
        frame_start=cfg["frame_start"],
        frame_end=cfg["frame_end"],
        overwrite=cfg["overwrite"],
        basename=cfg["output_basename"],
        linear_to_srgb=cfg["linear_to_srgb"],
    )
    log.info("Stage 1 complete in %.1f sec", time.time() - t0)


def run_filter_stage(cfg: dict) -> None:
    """Stage 2: Cull background/unwanted splats from the raw .ply sequence."""
    log.info("=" * 70)
    log.info("STAGE 2: Splat filter")
    log.info("  input:  %s", cfg["output_dir"])
    log.info("  output: %s", cfg["cleaned_dir"])
    log.info("=" * 70)

    filter_cfg = sf.FilterConfig(
        min_brightness=cfg["min_brightness"],
        max_brightness=cfg["max_brightness"],
        min_opacity=cfg["min_opacity"],
        key_color=cfg["key_color"],
        key_tolerance=cfg["key_tolerance"],
        key_invert=cfg["key_invert"],
        bbox=cfg["bbox"],
        bbox_invert=cfg["bbox_invert"],
        max_scale=cfg["max_scale"],
    )

    t0 = time.time()
    sf.process_directory(
        in_dir=cfg["output_dir"],
        out_dir=cfg["cleaned_dir"],
        cfg=filter_cfg,
        overwrite=cfg["overwrite"],
    )
    log.info("Stage 2 complete in %.1f sec", time.time() - t0)


# =============================================================================
# CLI overrides
# =============================================================================

def parse_args() -> argparse.Namespace:
    """CLI args OVERRIDE values in CONFIG. Anything not passed uses CONFIG default."""
    p = argparse.ArgumentParser(
        description="End-to-end video-to-splat pipeline (SHARP + filter).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",       type=Path, help="Input image sequence directory.")
    p.add_argument("--output",      type=Path, help="Raw .ply output directory.")
    p.add_argument("--cleaned",     type=Path, help="Cleaned .ply output directory.")

    p.add_argument("--frame-start", type=int)
    p.add_argument("--frame-end",   type=int)

    p.add_argument("--device", choices=("cuda", "cpu", "mps"))
    p.add_argument("--linear-to-srgb", action="store_true",
                   help="Apply sRGB gamma to EXR inputs (only for raw linear EXR).")
    p.add_argument("--checkpoint", type=Path)

    p.add_argument("--key-color", type=str, metavar="R,G,B",
                   help="RGB triple to key out (e.g. 0,0,0 for black, 0.05,0.6,0.05 for green).")
    p.add_argument("--key-tolerance", type=float)
    p.add_argument("--key-invert", action="store_true")

    p.add_argument("--min-brightness", type=float)
    p.add_argument("--max-brightness", type=float)
    p.add_argument("--min-opacity",    type=float)
    p.add_argument("--max-scale",      type=float)

    p.add_argument("--skip-filter", action="store_true",
                   help="Run inference only, skip the filter stage.")
    p.add_argument("--filter-only", action="store_true",
                   help="Skip inference; only run filter on existing .ply directory.")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge CLI args into cfg. Only overwrites where CLI actually provided a value."""
    def _set(key: str, value):
        if value is not None:
            cfg[key] = value

    _set("input_dir",    args.input)
    _set("output_dir",   args.output)
    _set("cleaned_dir",  args.cleaned)
    _set("frame_start",  args.frame_start)
    _set("frame_end",    args.frame_end)
    _set("device",       args.device)
    _set("checkpoint",   args.checkpoint)
    _set("key_tolerance", args.key_tolerance)
    _set("min_brightness", args.min_brightness)
    _set("max_brightness", args.max_brightness)
    _set("min_opacity",    args.min_opacity)
    _set("max_scale",      args.max_scale)

    if args.key_color is not None:
        cfg["key_color"] = sf.parse_rgb_triple(args.key_color)
    if args.linear_to_srgb:
        cfg["linear_to_srgb"] = True
    if args.key_invert:
        cfg["key_invert"] = True
    if args.overwrite:
        cfg["overwrite"] = True
    if args.verbose:
        cfg["verbose"] = True

    return cfg


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(dict(CONFIG), args)

    # Configure logging using cfg["verbose"].
    fts.configure_logging(cfg["verbose"])
    sf.configure_logging(cfg["verbose"])

    log.info("Pipeline config summary:")
    for k in ("input_dir", "output_dir", "cleaned_dir", "device", "key_color",
              "key_tolerance", "min_opacity", "overwrite"):
        log.info("  %-15s %s", k, cfg[k])
    log.info("")

    t_start = time.time()
    try:
        if not args.filter_only:
            run_inference_stage(cfg)
        else:
            log.info("(Skipping inference stage; --filter-only)")

        if cfg["filter_enabled"] and not args.skip_filter:
            run_filter_stage(cfg)
        else:
            log.info("(Skipping filter stage)")
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        return 1

    log.info("")
    log.info("Full pipeline complete in %.1f sec", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
