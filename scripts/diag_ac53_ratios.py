"""Per-timepoint detected/injected ratios for ac53 with the CURRENT (colour-corrected)
calibration, to compare against the pre-correction numbers (41.6h=0.80, 48h=2.04).

Builds the ac53 context (colour correction now active) and evaluates each calibration
frame. Reports the objective so we can see whether the loaded calibration is the optimised
one (objective ~= the run's best 0.00206) or the seed (much higher).

    python scripts/diag_ac53_ratios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from auto_calibrate_color_to_mass import build_context  # noqa: E402
from darsia.presets.workflows.rig import Rig  # noqa: E402

ctx = build_context(
    run="ac53",
    config_dir=Path("config_seg6/run_ac"),
    rig_cls=Rig,
    ref_config_path="config_seg6/common.toml",
    use_facies=True,
    per_label_params=True,
)

print(f"\nloaded {len(ctx._loaded)} calibration frames (after any neighbour-substitution/drops)")
print(f"{'t(h)':>7} {'injected':>12} {'detected':>12} {'ratio':>7}")
total = 0.0
rows = []
for img, injected, t_h in ctx._loaded:
    try:
        detected = float(ctx.geometry.integrate(ctx.calibration(img).mass))
    except Exception as exc:  # noqa: BLE001
        print(f"{t_h:7.2f}  eval-error: {exc}")
        continue
    ratio = detected / injected if injected else float("nan")
    total += abs(detected - injected)
    flag = "  <==" if (injected and abs(ratio - 1) > 0.25) else ""
    print(f"{t_h:7.2f} {injected:12.4g} {detected:12.4g} {ratio:7.3f}{flag}")
    rows.append((t_h, injected, detected, ratio))

print(f"\nobjective (sum|det-inj|) = {total:.6g}   "
      f"(run's best was 0.00206 -> if close, this IS the optimised calibration)")

# write a small csv next to the logs for sharing
out = HERE.parent / "logs" / "ac53_ratios_corrected.csv"
out.parent.mkdir(parents=True, exist_ok=True)
import csv
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_h", "injected", "detected", "ratio"])
    w.writerows(rows)
print(f"(written to {out})")
