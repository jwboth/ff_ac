"""Auto-calibration objective against the new DarSIA color-to-mass API.

Re-implements the ff_um ``auto_calibrate_color_to_mass`` objective on top of the
new DarSIA preset workflow. The objective is the classic mass-balance error:

    objective = sum_over_calibration_images | detected_total_mass - injected_mass |

where for a trial set of signal-function values we

  1. build the analysis context (loads experiment, rig/geometry and the
     color-to-mass pipeline) via ``prepare_analysis_context``;
  2. apply the trial values to the per-label signal functions
     (``signal_model.model[1][label]`` PWTransformations, monotonically clipped -
     exactly the attribute path the DarSIA UI and ff_um use);
  3. for each calibration image, run the pipeline, integrate the mass field over
     the geometry, and compare to ``injection_protocol.injected_mass(date)``.

The DarSIA-dependent parts are imported lazily so the pure parameter logic
(``parse_params`` / ``apply_params``) can be unit-tested without DarSIA.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("calibration_objective")

_VALUE_RE = re.compile(r"(?:.*\.)?value(\d+)$", re.IGNORECASE)
_LABEL_VALUE_RE = re.compile(r"(?:signal\.)?label(\d+)\.value(\d+)$", re.IGNORECASE)


# -------------------------------------------------------------------------
# Pure parameter logic (DarSIA-free, unit-testable)
# -------------------------------------------------------------------------
def parse_params(spec: str) -> dict:
    """Parse "value1=0.3;value2=1.1;..." (or "label0.value1=...") into a dict."""
    out: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        k, v = part.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _index_map(params: dict) -> dict:
    """Map params to {value_index: x}. Used for the global (per-all-labels) case."""
    idx: dict[int, float] = {}
    for k, v in params.items():
        m = _VALUE_RE.match(k)
        if m:
            idx[int(m.group(1))] = float(v)
    return idx


def _per_label_map(params: dict) -> dict:
    """Map params to {label: {value_index: x}} for label-specific names."""
    out: dict[int, dict] = {}
    for k, v in params.items():
        m = _LABEL_VALUE_RE.match(k)
        if m:
            out.setdefault(int(m.group(1)), {})[int(m.group(2))] = float(v)
    return out


def _apply_to_func(func, idx_vals: dict, np_module) -> None:
    """Set values[i]=x on a single PWTransformation-like object, monotonically."""
    vals = list(getattr(func, "values"))
    vals = [float(x) for x in vals]
    for i, x in idx_vals.items():
        if 0 <= i < len(vals):
            vals[i] = x
    if np_module is not None and len(vals) > 1:
        vals = list(np_module.maximum.accumulate(np_module.asarray(vals, dtype=float)))
    else:  # monotonic without numpy
        for i in range(1, len(vals)):
            vals[i] = max(vals[i], vals[i - 1])
    func.update(values=vals)


def apply_params(color_to_mass_analysis, params: dict, labels=None,
                 np_module=None) -> int:
    """Apply trial params to the analysis' per-label signal functions.

    Supports both global ``valueI`` names (applied to every label) and
    label-specific ``labelL.valueI`` names. Returns the number of labels updated.
    """
    hetero = color_to_mass_analysis.signal_model.model[1]  # label -> PWTransformation
    per_label = _per_label_map(params)
    global_idx = _index_map({k: v for k, v in params.items()
                             if not _LABEL_VALUE_RE.match(k)})

    if labels is None:
        labels = _hetero_labels(hetero)

    updated = 0
    for lbl in labels:
        idx_vals = dict(global_idx)
        idx_vals.update(per_label.get(lbl, {}))
        if not idx_vals:
            continue
        try:
            func = hetero[lbl]
        except (KeyError, TypeError):
            continue
        _apply_to_func(func, idx_vals, np_module)
        updated += 1
    return updated


def _hetero_labels(hetero) -> list:
    """Best-effort extraction of the label keys from a heterogeneous model."""
    for attr in ("keys", "labels"):
        obj = getattr(hetero, attr, None)
        if callable(obj):
            try:
                return list(obj())
            except Exception:  # noqa: BLE001
                pass
    for attr in ("functions", "model", "_functions", "data"):
        d = getattr(hetero, attr, None)
        if isinstance(d, dict):
            return list(d.keys())
    if isinstance(hetero, dict):
        return list(hetero.keys())
    return []


# -------------------------------------------------------------------------
# DarSIA-backed objective evaluation
# -------------------------------------------------------------------------
def evaluate(config_paths: list, params: dict,
             rig_cls=None, max_images: Optional[int] = None) -> float:
    """Mass-balance objective for one trial. Lower is better.

    Imports DarSIA lazily (requires ``uv sync``).
    """
    import numpy as np
    from darsia.presets.workflows.rig import Rig
    from darsia.presets.workflows.analysis.analysis_context import (
        prepare_analysis_context,
    )

    cls = rig_cls or Rig
    paths = [Path(p) for p in config_paths]
    ctx = prepare_analysis_context(
        cls=cls, path=paths, all=False, require_color_to_mass=True,
    )
    cta = ctx.color_to_mass_analysis
    if cta is None:
        raise RuntimeError("color_to_mass_analysis not initialised; calibrate first.")

    labels = _label_ids_from_ctx(ctx, np)
    n = apply_params(cta, params, labels=labels, np_module=np)
    LOG.info("Applied trial params to %d label(s)", n)

    total_err = 0.0
    count = 0
    for path in ctx.image_paths:
        if max_images is not None and count >= max_images:
            break
        img = ctx.fluidflower.read_image(Path(path))
        result = cta(img)
        detected = float(ctx.fluidflower.geometry.integrate(result.mass))
        injected = float(ctx.experiment.injection_protocol.injected_mass(date=img.date))
        total_err += abs(detected - injected)
        count += 1
    LOG.info("Evaluated %d image(s); objective=%.6g", count, total_err)
    return total_err


def _label_ids_from_ctx(ctx, np) -> list:
    """Prefer the rig labels image for the set of calibrated labels."""
    labels_img = getattr(ctx.fluidflower, "labels", None)
    arr = getattr(labels_img, "img", None)
    if arr is not None:
        return [int(x) for x in np.unique(arr) if x >= 0]
    # fall back to whatever the heterogeneous model exposes
    return _hetero_labels(ctx.color_to_mass_analysis.signal_model.model[1])
