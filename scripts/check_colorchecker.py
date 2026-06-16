"""Diagnostic: colour-checker detection + correction quality for ac53.

Builds/loads the rig from the ac53 config, reports:
  - whether a ColorCorrection is active and the ROI it is using (in shape-corrected coords),
  - the post-correction swatch residual on a real calibration frame (the quality verdict),
  - and saves a view of the shape-corrected baseline's upper-right so the real checker
    corners can be read for an explicit ROI if auto-detection failed.

Run from the repo root with the calibration venv:

    python scripts/check_colorchecker.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

RUN = sys.argv[1] if len(sys.argv) > 1 else "ac53"
COMMON = REPO / "config_seg6" / "common.toml"
RUNCFG = REPO / "config_seg6" / "run_ac" / f"{RUN}.toml"
COLORON = REPO / "config_seg6" / "coloron.toml"
CFGS = [COMMON, RUNCFG] + ([COLORON] if COLORON.exists() else [])
OUT = REPO / "logs"
OUT.mkdir(parents=True, exist_ok=True)

lines: list[str] = []


def emit(msg: str) -> None:
    print(msg, flush=True)
    lines.append(str(msg))


emit("=== COLORCHECK DIAGNOSTIC (ac53) ===")
try:
    from darsia.presets.workflows.rig import Rig
    from darsia.presets.workflows.analysis.analysis_context import (
        prepare_analysis_context,
    )

    ctx = prepare_analysis_context(
        cls=Rig, path=CFGS, all=False, require_color_to_mass=False
    )
    ff = ctx.fluidflower
    ccs = list(getattr(ff, "color_corrections", []) or [])
    emit(f"num color_corrections on rig: {len(ccs)}")

    cc = ccs[0] if ccs else None
    if cc is not None:
        roi = getattr(cc, "roi", None)
        emit(f"color correction active={getattr(cc, 'active', None)}")
        emit(f"ROI it is using (shape-corrected coords):\n{np.asarray(roi).tolist() if roi is not None else None}")

    # Save a view of the shape-corrected baseline upper-right (where the checker lives),
    # so the TRUE corners can be read in the SAME coordinate system the ROI uses.
    try:
        scb = getattr(ff, "shape_corrected_baseline", None) or getattr(ff, "baseline", None)
        arr = np.asarray(scb.img)
        if arr.dtype == np.uint8:
            arr8 = arr
        else:
            a = arr.astype(np.float64)
            if a.max() > 1.5:  # already 0..255-ish
                a = a / 255.0
            arr8 = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        H, W = arr8.shape[:2]
        emit(f"shape_corrected_baseline shape: {arr8.shape}")
        from PIL import Image
        full = Image.fromarray(arr8[..., :3] if arr8.ndim == 3 else arr8)
        ur = full.crop((int(W * 0.60), 0, W, int(H * 0.40)))
        ur.save(OUT / "ac53_scb_upperright.png")
        emit(f"saved shape-corrected upper-right crop -> {OUT / 'ac53_scb_upperright.png'} "
             f"(crop offset x0={int(W*0.60)}, y0=0)")
    except Exception:
        emit("could not save shape-corrected baseline crop:")
        emit(traceback.format_exc())

    # Quality verdict: read one real calibration frame (applies all corrections incl colour)
    # and report the post-correction swatch residual + flag.
    try:
        paths = list(getattr(ctx, "image_paths", []) or [])
        if paths and cc is not None:
            _ = ff.read_image(Path(paths[0]))
            emit(f"read calibration frame: {Path(paths[0]).name}")
            emit(f"  last_residual = {getattr(cc, 'last_residual', None)}")
            emit(f"  last_flagged  = {getattr(cc, 'last_flagged', None)}")
            thr = getattr(cc, "residual_warn_threshold", 0.06)
            res = getattr(cc, "last_residual", float("nan"))
            if res == res and res <= thr:
                emit("VERDICT: correction looks GOOD (low residual).")
            else:
                emit("VERDICT: correction BAD/uncertain (high or NaN residual) -> need explicit ROI.")
        else:
            emit("no image_paths or no color correction; skipped residual check.")
    except Exception:
        emit("residual check failed:")
        emit(traceback.format_exc())

except Exception:
    emit("ERROR while building rig:")
    emit(traceback.format_exc())

(OUT / "colorcheck_result.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n(written to {OUT / 'colorcheck_result.txt'})", flush=True)
