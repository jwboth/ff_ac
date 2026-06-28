"""ff_ac-native auto-calibration objective (mass-balance), with the ff_um queue contract.

The science and orchestration are ff_um's: a per-label Optuna search over the
piecewise signal-model values that minimises the classic mass-balance error

    objective = sum_over_calibration_images | detected_total_mass - injected_mass |

but the *evaluation* is written against ff_ac's current DarSIA preset API
(``prepare_analysis_context`` + ``HeterogeneousColorToMassAnalysis``), which
loads experiment/rig/geometry/color-to-mass and resolves all the calibration
config internally. This module exposes exactly the names the ported
``distributed_auto_calibration_queue.py`` imports.
"""
from __future__ import annotations

import argparse
import random
import copy
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

logger = logging.getLogger(__name__)

PENALTY_VALUE = 1e12

# --- ff_ac signal model: 4 support points (value0 locked at 0, value1..3 free);
#     flash locked at ff_ac defaults (0, 0.75, 0.75, 1.0). Per-label expansion in
#     build_param_space replicates this across the active (non-ignored) labels. ---
PARAM_SPACE_TEMPLATE: List[Dict[str, Any]] = [
    {"name": "signal.label0.value0",
     "attr_path": ["signal_model", "model", 1, 0, "values", 0],
     "bounds": (0.0, 0.0), "type": "float"},
    *[{"name": f"signal.label0.value{i}",
       "attr_path": ["signal_model", "model", 1, 0, "values", i],
       "bounds": (0.0, 2.0), "type": "float"} for i in range(1, 4)],  # signal node bound (validated)
    {"name": "flash.min_value_aq", "attr_path": ["flash", "min_value_aq"],
     "bounds": (0.0, 0.0), "type": "float"},
    {"name": "flash.max_value_aq", "attr_path": ["flash", "max_value_aq"],
     "bounds": (0.75, 0.75), "type": "float"},
    {"name": "flash.min_value_g", "attr_path": ["flash", "min_value_g"],
     "bounds": (0.75, 0.75), "type": "float"},
    {"name": "flash.max_value_g", "attr_path": ["flash", "max_value_g"],
     "bounds": (1.0, 1.0), "type": "float"},  # ff_um value, LOCKED (static flash)
]

_SIGNAL_PARAM_RE = re.compile(r"^signal\.label(?P<label>-?\d+)\.value(?P<idx>\d+)$")
_LABEL_VALUE_RE = re.compile(r"(?:signal\.)?label(\d+)\.value(\d+)$", re.IGNORECASE)
_VALUE_RE = re.compile(r"(?:.*\.)?value(\d+)$", re.IGNORECASE)


# =========================================================================
# Dataclasses (mirror the queue's expected shapes)
# =========================================================================
@dataclass
class Metrics:
    injected_full: float
    total_full: float


@dataclass
class EvalResult:
    objective: float
    feasible: bool
    metrics: Dict[str, Metrics]
    status: str
    params: Dict[str, Any]


@dataclass
class CalibrationContext:
    run: str
    config: Any
    experiment: Any
    fluidflower: Any
    geometry: Any
    calibration: Any                       # HeterogeneousColorToMassAnalysis
    calibration_images: List[Path]
    reference_label: int
    signal_label: Optional[int]
    signal_labels: List[int]
    param_space: List[Dict[str, Any]] = field(default_factory=list)
    enforce_lower: bool = False
    per_label_params: bool = True
    objective_integral: str = "off"
    label_weights: Optional[Dict[int, float]] = None
    calibration_folder: Optional[Path] = None
    # preloaded (corrected image, injected_mass) pairs - read once, reused per trial
    _loaded: List[Tuple[Any, float, float]] = field(default_factory=list)  # (img, injected, t_hours)


# =========================================================================
# Param-name helpers (queue contract)
# =========================================================================
def _parse_signal_name(name: str) -> Optional[Tuple[int, int]]:
    m = _SIGNAL_PARAM_RE.match(name)
    if not m:
        return None
    return int(m.group("label")), int(m.group("idx"))


def _value_entries_by_label(param_space: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    entries_by_label: Dict[int, List[Dict[str, Any]]] = {}
    for entry in param_space:
        parsed = _parse_signal_name(entry.get("name", ""))
        if not parsed:
            continue
        label, idx = parsed
        entries_by_label.setdefault(label, []).append(entry)
    for label, entries in entries_by_label.items():
        entries_by_label[label] = sorted(entries, key=lambda e: _parse_signal_name(e["name"])[1])
    return entries_by_label


def _monotonic_bounds(entries: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    lows = [float(e["bounds"][0]) for e in entries]
    highs = [float(e["bounds"][1]) for e in entries]
    lower: List[float] = []
    for i, low in enumerate(lows):
        lower.append(low if i == 0 else max(low, lower[i - 1]))
    upper: List[float] = [0.0] * len(entries)
    if entries:
        upper[-1] = highs[-1]
        for i in range(len(entries) - 2, -1, -1):
            upper[i] = min(highs[i], upper[i + 1])
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo > hi:
            raise ValueError("Infeasible monotonic bounds for {}: {} > {}".format(entries[i]["name"], lo, hi))
    return lower, upper


def _project_monotonic_values(values: Sequence[float], entries: Sequence[Dict[str, Any]]) -> List[float]:
    if not entries:
        return []
    if len(values) != len(entries):
        raise ValueError("Value count mismatch for monotonic projection.")
    lower, upper = _monotonic_bounds(entries)
    projected: List[float] = []
    prev = None
    for i, raw in enumerate(values):
        min_i = lower[i] if prev is None else max(lower[i], prev)
        max_i = upper[i]
        projected_val = min(max(float(raw), min_i), max_i)
        projected.append(projected_val)
        prev = projected_val
    return projected


def _parse_bool(value) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# =========================================================================
# Param space
# =========================================================================
def _match_bounds(name, override, default_override):
    if name in override:
        return override[name]
    if name in default_override:
        return default_override[name]
    # wildcard: signal.labelN.valueI is also matched by signal.label*.valueI
    # (this is the key form --param-ranges writes, applied across all labels).
    wild = re.sub(r"\.label-?\d+\.", ".label*.", name)
    if wild != name:
        if wild in override:
            return override[wild]
        if wild in default_override:
            return default_override[wild]
    return None


def _rebuild_template_signal(template, n_free_values):
    """Regenerate signal.* as value0 (locked) + value1..value{n_free_values}, preserving the
    template free-value bounds and all non-signal (flash) entries. Lets the param space match
    the signal-model resolution (num_segments+1 points): a 7-point model optimises value1..6."""
    sig = [e for e in template if _parse_signal_name(e["name"])]
    other = [e for e in template if not _parse_signal_name(e["name"])]
    free_bounds = (0.0, 4.0)
    for e in sig:
        p = _parse_signal_name(e["name"])
        if p and p[1] >= 1:
            free_bounds = tuple(e["bounds"]); break
    new_sig = [{"name": "signal.label0.value0",
                "attr_path": ["signal_model", "model", 1, 0, "values", 0],
                "bounds": (0.0, 0.0), "type": "float"}]
    for i in range(1, int(n_free_values) + 1):
        new_sig.append({"name": f"signal.label0.value{i}",
                        "attr_path": ["signal_model", "model", 1, 0, "values", i],
                        "bounds": free_bounds, "type": "float"})
    return new_sig + other


def build_param_space(run, bounds_map, signal_label=None, signal_labels=None,
                      per_label_params=False, use_facies=True, n_free_values=None):
    base = copy.deepcopy(PARAM_SPACE_TEMPLATE)
    if n_free_values is not None:
        base = _rebuild_template_signal(base, int(n_free_values))
    override = (bounds_map or {}).get(run, {})
    default_override = (bounds_map or {}).get("default", {})
    sig = [e for e in base if _parse_signal_name(e["name"])]
    other = [e for e in base if not _parse_signal_name(e["name"])]
    space: List[Dict[str, Any]] = []
    if per_label_params:
        labels = list(signal_labels or ([signal_label] if signal_label is not None else []))
        for label in labels:
            for e in sig:
                ne = copy.deepcopy(e)
                ne["name"] = e["name"].replace("signal.label0", f"signal.label{label}")
                ne["attr_path"] = list(e["attr_path"]); ne["attr_path"][3] = int(label)
                space.append(ne)
        space.extend(copy.deepcopy(other))
    else:
        space = base
        if signal_label is not None:
            for e in space:
                if _parse_signal_name(e["name"]):
                    e["attr_path"] = list(e["attr_path"]); e["attr_path"][3] = int(signal_label)
    for e in space:
        b = _match_bounds(e["name"], override, default_override)
        if b is not None:
            e["bounds"] = tuple(b)
    return space


def sample_params(param_space: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    samples: Dict[str, Any] = {}
    for _, entries in _value_entries_by_label(param_space).items():
        lower, upper = _monotonic_bounds(entries)
        prev = None
        for i, entry in enumerate(entries):
            lo, hi = entry["bounds"]
            if lo == hi:
                val = float(lo)
            else:
                min_i = lower[i] if prev is None else max(lower[i], prev)
                max_i = upper[i]
                val = min_i if max_i <= min_i else (min_i + random.random() * (max_i - min_i))
            samples[entry["name"]] = val
            prev = val
    for entry in param_space:
        if entry["name"] in samples:
            continue
        low, high = entry["bounds"]
        samples[entry["name"]] = (int(round(random.uniform(low, high)))
                                  if entry.get("type", "float") == "int"
                                  else random.uniform(low, high))
    return samples


def suggest_params_trial(trial, param_space: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Cumulative-fraction monotone suggestion. For each label, value_i is built
    as value_{i-1} + f_i * (upper_i - value_{i-1}) with f_i in [0,1] (fixed bounds).
    The mapping f -> values is BIJECTIVE and monotone by construction, so Optuna
    optimises the monotone structure directly (no permutation symmetry / hidden
    sort, no clipping mismatch). Non-signal params (e.g. flash) are suggested
    directly. The returned values are exactly what is evaluated, logged and applied."""
    params: Dict[str, Any] = {}
    for _, entries in _value_entries_by_label(param_space).items():
        lower, upper = _monotonic_bounds(entries)
        prev = None
        for i, entry in enumerate(entries):
            lo, hi = entry["bounds"]
            if lo == hi:
                val = float(lo)
            else:
                min_i = lower[i] if prev is None else max(lower[i], prev)
                max_i = upper[i]
                if max_i <= min_i:
                    val = min_i
                else:
                    f = trial.suggest_float("cum_" + entry["name"], 0.0, 1.0)
                    val = min_i + f * (max_i - min_i)
            params[entry["name"]] = val
            prev = val
    for entry in param_space:
        if entry["name"] in params:
            continue
        low, high = entry["bounds"]
        if entry.get("type", "float") == "int":
            params[entry["name"]] = trial.suggest_int(entry["name"], int(low), int(high))
        else:
            params[entry["name"]] = trial.suggest_float(entry["name"], float(low), float(high))
    return params


# weighting stubs (not used by default; kept for queue contract) --------------
def compute_auto_label_weights(context) -> Dict[int, float]:
    return {}


def apply_label_weight_grouping(weights, context, grouping):
    return weights


def load_bounds_map(path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalise_static_light_mode(mode: str | None) -> str:
    value = (mode or os.environ.get("FFAC_STATIC_LIGHT_CORRECTION") or "off").strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "on": "blue-gain",
        "blue": "blue-gain",
        "rgb": "blue-gain",
        "static-rgb": "blue-gain",
        "intensity": "intensity",
        "luma": "intensity",
        "static-intensity": "intensity",
        "spatial": "blue-spatial",
        "spatial-gain": "blue-spatial",
        "spatial-rgb": "blue-spatial",
        "static-spatial": "blue-spatial",
        "blue-spatial-gain": "blue-spatial",
        "spatial-blue-gain": "blue-spatial",
        "spatial-intensity": "intensity-spatial",
        "intensity-spatial-gain": "intensity-spatial",
    }
    return aliases.get(value, value)


def _smooth_static_gain_field(
    raw_gain: np.ndarray,
    mask: np.ndarray,
    fallback_gain: np.ndarray,
    sigma: float,
    gain_low: float,
    gain_high: float,
) -> np.ndarray:
    """Smooth sparse stable-region gains into a full low-resolution gain field."""

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return np.broadcast_to(fallback_gain.reshape(1, 1, 3), raw_gain.shape).astype(np.float32)

    field = np.empty_like(raw_gain, dtype=np.float32)
    base_weight = mask.astype(np.float32)
    for channel in range(3):
        values = raw_gain[..., channel].astype(np.float32)
        finite = np.isfinite(values)
        weights = base_weight * finite.astype(np.float32)
        if float(np.sum(weights)) <= 0:
            field[..., channel] = float(fallback_gain[channel])
            continue
        numerator = gaussian_filter(np.where(finite, values, 0.0) * weights, sigma=sigma, mode="nearest")
        denominator = gaussian_filter(weights, sigma=sigma, mode="nearest")
        smoothed = np.where(denominator > 1e-6, numerator / denominator, float(fallback_gain[channel]))
        field[..., channel] = np.clip(smoothed, gain_low, gain_high)
    return field


def _upsample_static_gain_field(field: np.ndarray, shape: tuple[int, int], stride: int) -> np.ndarray:
    """Nearest-neighbour upsampling from stride-sampled grid to image shape."""

    expanded = np.repeat(np.repeat(field, stride, axis=0), stride, axis=1)
    pad_y = max(0, shape[0] - expanded.shape[0])
    pad_x = max(0, shape[1] - expanded.shape[1])
    if pad_y or pad_x:
        expanded = np.pad(expanded, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
    return expanded[: shape[0], : shape[1], :]


def _default_calibration_log_root() -> str:
    env_root = os.environ.get("FFAC_CALIBRATION_LOG_ROOT")
    if env_root:
        return env_root
    preferred = Path(r"Z:\Albus\Autokalibrering_log")
    try:
        if Path(preferred.drive + "\\").exists():
            return str(preferred)
    except Exception:
        pass
    return "logs"


def _apply_static_light_correction(
    loaded: List[Tuple[Any, float, float]],
    *,
    mode: str | None,
    run: str,
) -> None:
    """Apply a small per-frame gain from stable FluidFlower image regions."""

    correction = _normalise_static_light_mode(mode)
    if correction == "off" or len(loaded) < 2:
        return
    if correction not in {"blue-gain", "intensity", "blue-spatial", "intensity-spatial"}:
        raise ValueError(
            "Unknown static light correction mode "
            f"{correction!r}; expected off, blue-gain, intensity, blue-spatial, or intensity-spatial."
        )

    stride = max(4, int(os.environ.get("FFAC_STATIC_LIGHT_STRIDE", "16")))
    min_samples = max(1000, int(os.environ.get("FFAC_STATIC_LIGHT_MIN_SAMPLES", "4000")))
    max_samples = max(min_samples, int(os.environ.get("FFAC_STATIC_LIGHT_MAX_SAMPLES", "50000")))
    clip_frac = float(os.environ.get("FFAC_STATIC_LIGHT_GAIN_CLIP", "0.15"))
    gain_low = max(0.01, 1.0 - abs(clip_frac))
    gain_high = 1.0 + abs(clip_frac)
    spatial_sigma = float(os.environ.get("FFAC_STATIC_LIGHT_SPATIAL_SIGMA", "6.0"))

    samples: List[np.ndarray] = []
    for img, _injected, _t_h in loaded:
        arr = np.asarray(img.img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return
        samples.append(arr[::stride, ::stride, :3])
    stack = np.stack(samples, axis=0)

    finite = np.all(np.isfinite(stack), axis=(0, 3))
    mean_rgb = np.nanmean(stack, axis=0)
    std_rgb = np.nanstd(stack, axis=0)
    mean_luma = np.mean(mean_rgb, axis=2)
    std_luma = np.std(stack.mean(axis=3), axis=0)
    std_chroma = np.linalg.norm(std_rgb, axis=2)

    finite_luma = mean_luma[np.isfinite(mean_luma) & finite]
    if finite_luma.size == 0:
        return
    lo, hi = np.nanpercentile(finite_luma, [5.0, 95.0])
    stable_limit = np.nanpercentile(std_luma[finite], 35.0)
    chroma_stable_limit = np.nanpercentile(std_chroma[finite], 40.0)
    blueish = (mean_rgb[..., 2] > mean_rgb[..., 0] + 0.005) & (
        mean_rgb[..., 2] > mean_rgb[..., 1] + 0.002
    )
    base_mask = (
        finite
        & (mean_luma >= lo)
        & (mean_luma <= hi)
        & (std_luma <= stable_limit)
        & (std_chroma <= chroma_stable_limit)
    )
    mask = base_mask & blueish
    if int(np.count_nonzero(mask)) < min_samples:
        mask = base_mask
    if int(np.count_nonzero(mask)) < min_samples:
        logger.warning(
            "[%s] static light correction skipped: only %d stable samples found.",
            run,
            int(np.count_nonzero(mask)),
        )
        return

    ys, xs = np.nonzero(mask)
    if ys.size > max_samples:
        choose = np.linspace(0, ys.size - 1, max_samples, dtype=int)
        ys = ys[choose]
        xs = xs[choose]

    frame_values = stack[:, ys, xs, :]
    med_rgb = np.nanmedian(frame_values, axis=1)
    if not np.all(np.isfinite(med_rgb)):
        logger.warning("[%s] static light correction skipped: non-finite medians.", run)
        return

    reference_mode = (os.environ.get("FFAC_STATIC_LIGHT_REFERENCE") or "median").strip().lower()
    ref_rgb = med_rgb[0] if reference_mode == "first" else np.nanmedian(med_rgb, axis=0)
    ref_rgb = np.maximum(ref_rgb, 1e-6)

    spatial_reference = None
    if correction in {"blue-spatial", "intensity-spatial"}:
        spatial_reference = stack[0] if reference_mode == "first" else np.nanmedian(stack, axis=0)
        spatial_reference = np.maximum(spatial_reference.astype(np.float32), 1e-6)

    gains: List[np.ndarray] = []
    field_ranges: List[Tuple[float, float]] = []
    for idx, (img, _injected, _t_h) in enumerate(loaded):
        cur_rgb = np.maximum(med_rgb[idx], 1e-6)
        if correction == "intensity":
            ref_luma = float(np.mean(ref_rgb))
            cur_luma = float(np.mean(cur_rgb))
            gain = np.array([ref_luma / max(cur_luma, 1e-6)] * 3, dtype=np.float32)
        elif correction == "blue-gain":
            gain = (ref_rgb / cur_rgb).astype(np.float32)
        elif correction == "intensity-spatial":
            assert spatial_reference is not None
            cur_grid = np.maximum(stack[idx].astype(np.float32), 1e-6)
            ref_luma_grid = np.mean(spatial_reference, axis=2)
            cur_luma_grid = np.mean(cur_grid, axis=2)
            raw_scalar = ref_luma_grid / np.maximum(cur_luma_grid, 1e-6)
            raw_gain = np.repeat(raw_scalar[..., None], 3, axis=2)
            ref_luma = float(np.mean(ref_rgb))
            cur_luma = float(np.mean(cur_rgb))
            gain = np.array([ref_luma / max(cur_luma, 1e-6)] * 3, dtype=np.float32)
        else:
            assert spatial_reference is not None
            cur_grid = np.maximum(stack[idx].astype(np.float32), 1e-6)
            raw_gain = spatial_reference / cur_grid
            gain = (ref_rgb / cur_rgb).astype(np.float32)
        gain = np.clip(gain, gain_low, gain_high)
        arr = np.asarray(img.img, dtype=np.float32).copy()
        if correction in {"blue-spatial", "intensity-spatial"}:
            raw_gain = np.clip(raw_gain.astype(np.float32), gain_low, gain_high)
            field_small = _smooth_static_gain_field(
                raw_gain,
                mask,
                gain.astype(np.float32),
                spatial_sigma,
                gain_low,
                gain_high,
            )
            field = _upsample_static_gain_field(field_small, arr.shape[:2], stride)
            arr[..., :3] = np.clip(arr[..., :3] * field, 0.0, 1.0)
            field_ranges.append((float(np.nanmin(field)), float(np.nanmax(field))))
        else:
            arr[..., :3] = np.clip(arr[..., :3] * gain.reshape(1, 1, 3), 0.0, 1.0)
        img.img = arr
        gains.append(gain)

    gain_arr = np.stack(gains, axis=0)
    field_note = ""
    if field_ranges:
        field_note = (
            f" field_min={min(v[0] for v in field_ranges):.4f}"
            f" field_max={max(v[1] for v in field_ranges):.4f}"
            f" spatial_sigma={spatial_sigma:.2f}"
        )
    logger.info(
        "[%s] static light correction mode=%s samples=%d stride=%d "
        "gain_min=%s gain_max=%s%s",
        run,
        correction,
        int(ys.size),
        stride,
        np.round(gain_arr.min(axis=0), 4).tolist(),
        np.round(gain_arr.max(axis=0), 4).tolist(),
        field_note,
    )


def write_history_csv(path: Path, history: Sequence[Dict[str, Any]]) -> None:  # type: ignore
    rows = list(history or [])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("iter,objective\n", encoding="utf-8"); return
    keys = sorted({k for r in rows for k in r.keys()})
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)


# =========================================================================
# Signal-value application (monotone) - mirrors the DarSIA UI / ff_um
# =========================================================================
def _apply_to_func(func, idx_vals, np_module) -> None:
    vals = [float(x) for x in list(getattr(func, "values"))]
    for i, x in idx_vals.items():
        if 0 <= i < len(vals):
            vals[i] = float(x)
    if np_module is not None and len(vals) > 1:
        vals = list(np_module.maximum.accumulate(np_module.asarray(vals, dtype=float)))
    else:
        for i in range(1, len(vals)):
            vals[i] = max(vals[i], vals[i - 1])
    func.update(values=vals)


def apply_params(calibration, params, labels=None, np_module=None) -> int:
    hetero = calibration.signal_model.model[1]
    per_label: Dict[int, Dict[int, float]] = {}
    global_idx: Dict[int, float] = {}
    for k, v in params.items():
        m = _LABEL_VALUE_RE.match(k)
        if m:
            per_label.setdefault(int(m.group(1)), {})[int(m.group(2))] = float(v)
            continue
        mv = _VALUE_RE.match(k)
        if mv and not k.lower().startswith("flash"):
            global_idx[int(mv.group(1))] = float(v)
    if labels is None:
        labels = list(hetero.keys()) if hasattr(hetero, "keys") else []
    updated = 0
    for lbl in labels:
        idx_vals = dict(global_idx); idx_vals.update(per_label.get(lbl, {}))
        if not idx_vals:
            continue
        try:
            func = hetero[lbl]
        except (KeyError, TypeError):
            continue
        _apply_to_func(func, idx_vals, np_module)
        updated += 1
    # Apply flash params (e.g. flash.max_value_g). SimpleFlash.__call__ reads the
    # attributes live, so a direct setattr is enough (and avoids update()'s
    # `x or self.x` bug that drops 0.0).
    flash = getattr(calibration, "flash", None)
    if flash is not None:
        for name, val in params.items():
            if name.startswith("flash."):
                attr = name.split(".", 1)[1]
                if hasattr(flash, attr):
                    setattr(flash, attr, float(val)); updated += 1
        # Physical coupling (opt-in via env FFAC_COUPLE_AQ_GAS): gas onset = aqueous
        # saturation, i.e. min_value_g := max_value_aq. The aq->gas transition is ONE
        # point (water saturates -> free gas begins), so min_value_g is DERIVED, not
        # optimised independently. This forbids a non-physical overlap zone where a
        # pixel is counted as both partially dissolved and partially gas below
        # saturation. Off by default, so the fleet and other runs are unaffected.
        if os.environ.get("FFAC_COUPLE_AQ_GAS", "").strip().lower() in ("1", "true", "yes", "on"):
            try:
                if hasattr(flash, "max_value_aq") and hasattr(flash, "min_value_g"):
                    setattr(flash, "min_value_g", float(getattr(flash, "max_value_aq")))
            except Exception:
                pass
    return updated


# =========================================================================
# Context building + evaluation (ff_ac-native)
# =========================================================================
def build_context(run, config_dir, rig_cls, ref_config_path=None, use_facies=True,
                  bounds_map=None, enforce_lower=False, per_label_params=True,
                  use_label_weights=False, label_weights=None, quality_scale=1.0,
                  quality_dtype=None, objective_integral="off",
                  static_light_correction=None) -> CalibrationContext:
    import numpy as np
    from darsia.presets.workflows.analysis.analysis_context import prepare_analysis_context

    config_dir = Path(config_dir)
    config_path = config_dir / f"{run}.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    common_path = Path(ref_config_path) if ref_config_path else config_dir.parent / "common.toml"
    config_paths = [common_path, config_path] if common_path.exists() else [config_path]

    # Colour-correction toggle. The prep step writes a per-run stamp recording whether the
    # cached rig + embedding + seed were built WITH colour correction (built on corrected vs
    # uncorrected images - the two states are baked into the cache, see load_corrections).
    # build_context auto-follows the stamp so master and workers stay consistent with the
    # cache without any flag threading. Stamp: <config_dir>/.color_state/<run>.txt = on|off.
    _color_state = "off"
    try:
        _stamp = config_dir / ".color_state" / f"{run}.txt"
        if _stamp.exists():
            _color_state = _stamp.read_text(encoding="utf-8").strip().lower()
    except Exception:  # noqa: BLE001
        _color_state = "off"
    if _color_state == "on":
        _overlay = config_dir.parent / "coloron.toml"
        if _overlay.exists():
            config_paths = config_paths + [_overlay]
            logger.info("[%s] colour-correction ON (stamp) -> overlay %s added",
                        run, _overlay.name)
        else:
            logger.warning("[%s] colour stamp=on but %s missing; running WITHOUT colour",
                           run, _overlay)
    else:
        logger.info("[%s] colour-correction OFF (stamp=%s)", run, _color_state)

    ctx_raw = prepare_analysis_context(
        cls=rig_cls, path=config_paths, all=False, require_color_to_mass=True,
    )
    cta = ctx_raw.color_to_mass_analysis
    if cta is None:
        raise RuntimeError(f"[{run}] color_to_mass_analysis not initialised; seed/calibrate first.")
    fluidflower = ctx_raw.fluidflower
    experiment = ctx_raw.experiment
    geometry = getattr(fluidflower, "geometry", None)
    config = getattr(ctx_raw, "config", None)

    # Sanitise the geometry integration weights: the porosity map carries NaN
    # OUTSIDE the sand domain (~26%% of pixels), and geometry.integrate weights by
    # voxel_volume(=area*depth*porosity) with a plain np.sum -> any NaN weight makes
    # the whole integral NaN. Those regions have no pore space, so set them to 0.
    for _attr in ("voxel_volume", "cached_voxel_volume"):
        _v = getattr(geometry, _attr, None)
        if isinstance(_v, np.ndarray):
            setattr(geometry, _attr,
                    np.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0))

    # active labels + reference + calibration folder (resolved ColorPathEmbedding).
    embedding = None
    try:
        embedding = config.calibration.mass.color
    except Exception:
        embedding = None
    if embedding is None:
        # fall back to the analysis mass embedding (same [color.path.*]) used to
        # build the color-to-mass pipeline.
        embedding = config.color.resolve(config.analysis.mass.color)
    ignore = set(getattr(embedding, "ignore_labels", []) or [])
    reference_label = int(getattr(embedding, "reference_label", 0))
    cp_folder = Path(embedding.color_paths_folder)
    calibration_folder = cp_folder.parent.parent / "color_to_mass" / cp_folder.name

    # --- Titration-anchored aqueous transfer (opt-in) ---
    # Replace the cached SimpleFlash's LINEAR c_aq(signal) ramp with the physically-derived
    # BTB/carbonate titration curve (TitrationFlash). Done here (not in the rig) so it needs
    # NO rig rebuild and applies in master + workers alike.
    #
    # TWO equivalent triggers (either turns it on):
    #   1. STAMP (recommended for the distributed queue): a per-run file
    #      <config_dir>/.titration_state/<run>.txt containing "on" (mirrors the colour stamp).
    #      Master + every worker call build_context and auto-follow the stamp, so there is NO
    #      env threading and no risk of mixed workers. Optional recipe on the same/next line:
    #      "on 1.25,0.726,34" (alk_mM,btb_mM,co2sat_mM).
    #   2. ENV (handy for the standalone script in one shell): FFAC_TITRATION_FLASH=on, recipe
    #      override FFAC_TITRATION_RECIPE="alk_mM,btb_mM,co2sat_mM".
    _titr_on = os.environ.get("FFAC_TITRATION_FLASH", "").strip().lower() in ("1", "true", "yes", "on")
    _titr_recipe = os.environ.get("FFAC_TITRATION_RECIPE", "").strip()
    try:
        _tstamp = config_dir / ".titration_state" / f"{run}.txt"
        if _tstamp.exists():
            _toks = _tstamp.read_text(encoding="utf-8").split()
            if _toks and _toks[0].strip().lower() in ("1", "true", "yes", "on"):
                _titr_on = True
                if len(_toks) > 1 and "," in _toks[1]:
                    _titr_recipe = _titr_recipe or _toks[1].strip()
    except Exception:  # noqa: BLE001
        pass
    if _titr_on:
        try:
            from darsia.multiphase.flash import TitrationFlash
            _flash = getattr(cta, "flash", None)
            if _flash is not None and not isinstance(_flash, TitrationFlash):
                _kw = {}
                if _titr_recipe:
                    _alk, _btb, _sat = (float(x) for x in _titr_recipe.split(","))
                    _kw = dict(alkalinity_M=_alk * 1e-3, btb_M=_btb * 1e-3, co2_sat_M=_sat * 1e-3)
                cta.flash = TitrationFlash.from_simple(_flash, **_kw)
                logger.info("[%s] TitrationFlash ACTIVE (aqueous branch = BTB titration curve) %s",
                            run, _kw or "(default recipe 1.25 mM alk, 0.726 mM BTB, 34 mM sat)")
        except Exception as _exc:  # noqa: BLE001
            logger.warning("[%s] TitrationFlash requested but injection failed: %s", run, _exc)

    hetero = cta.signal_model.model[1]
    all_labels = [int(x) for x in (hetero.keys() if hasattr(hetero, "keys") else [])]
    if not all_labels:
        labels_img = getattr(getattr(fluidflower, "labels", None), "img", None)
        if labels_img is not None:
            all_labels = [int(x) for x in np.unique(labels_img) if x >= 0]
    signal_labels = [l for l in all_labels if l not in ignore and l != 0]

    # preload corrected calibration images + (param-independent) injected mass
    loaded: List[Tuple[Any, float, float]] = []
    exp_start = getattr(experiment, "experiment_start", None)

    # Nearest-good-neighbour substitution. read_image applies the rig corrections incl.
    # ColorCorrection, which sets last_flagged when a frame's lighting/checker is too far
    # gone to recover. For CALIBRATION we only need ~13 well-spread points, not these exact
    # frames, so a flagged frame is replaced by the nearest-in-time correctable neighbour
    # (frames are ~5 min apart and the CO2 state barely changes over a few minutes). If no
    # neighbour within the window is usable, the calibration point is dropped. When colour
    # correction is OFF, color_corrections is empty -> _flagged is always False -> this
    # behaves exactly as a plain read (no neighbour I/O, current behaviour unchanged).
    def _read_and_flag(path):
        im = fluidflower.read_image(Path(path))
        flagged = any(
            getattr(_cc, "active", False) and getattr(_cc, "last_flagged", False)
            for _cc in (getattr(fluidflower, "color_corrections", None) or [])
        )
        return im, flagged

    def _select_frame(path, window_frames=6, max_minutes=30.0):
        im, flagged = _read_and_flag(path)
        if not flagged:
            return im
        folder = Path(path).parent
        siblings = sorted(folder.glob("*" + Path(path).suffix))
        names = [s.name for s in siblings]
        try:
            idx = names.index(Path(path).name)
        except ValueError:
            return im  # cannot locate neighbours; keep original (better than dropping)
        base_date = getattr(im, "date", None)
        for d in range(1, window_frames + 1):  # expand outward -> nearest first
            cands = []
            for j in (idx - d, idx + d):
                if not (0 <= j < len(siblings)):
                    continue
                c_im, c_flag = _read_and_flag(siblings[j])
                if c_flag:
                    continue
                c_date = getattr(c_im, "date", None)
                if base_date is not None and c_date is not None:
                    dt_min = abs((c_date - base_date).total_seconds()) / 60.0
                    if dt_min > max_minutes:
                        continue
                else:
                    dt_min = float(d)
                cands.append((dt_min, siblings[j].name, c_im))
            if cands:
                cands.sort(key=lambda t: t[0])
                logger.warning(
                    "[%s] calibration frame %s flagged; substituting nearest good "
                    "neighbour %s (%.0f min away)", run, Path(path).name,
                    cands[0][1], cands[0][0])
                return cands[0][2]
        return None  # no usable neighbour within window -> drop this calibration point

    light_master = os.environ.get("FFAC_MASTER_LIGHT_CONTEXT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if light_master:
        logger.info(
            "[%s] FFAC_MASTER_LIGHT_CONTEXT=on -> skipping calibration image preload",
            run,
        )
    else:
        for p in ctx_raw.image_paths:
            img = _select_frame(p)
            if img is None:
                logger.warning(
                    "[%s] calibration frame %s uncorrectable and no good neighbour within "
                    "window; dropping this calibration point.", run, Path(p).name)
                continue
            # Memory-bandwidth reduction for worker evaluations (the master always builds
            # with quality_dtype=None -> full float64, so the saved/finalised calibration
            # stays full-scale). A float32 cast halves the bytes streamed per array op in
            # the colour->signal->flash->mass pipeline; the integral itself is summed in
            # float64 inside geometry.integrate, so accuracy is preserved.
            if quality_dtype:
                try:
                    img.img = np.asarray(img.img, dtype=np.dtype(quality_dtype))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] quality_dtype=%s cast failed: %s", run, quality_dtype, exc)
            injected = float(experiment.injection_protocol.injected_mass(date=img.date))
            try:
                t_h = (img.date - exp_start).total_seconds() / 3600.0
            except Exception:
                t_h = float(len(loaded))
            loaded.append((img, injected, t_h))
        _apply_static_light_correction(loaded, mode=static_light_correction, run=run)
        logger.info("[%s] preloaded %d calibration image(s); active labels=%s",
                    run, len(loaded), signal_labels)

    # ---- optional spatial downscale (workers only; master uses quality_scale=1.0) ----
    # Ported from ff_um's _apply_quality_to_rig. Use darsia.resize (which updates the
    # Image's num_voxels / coordinate system, unlike a raw cv2 resize of .img) on the
    # SOURCE arrays the pipeline derives from, to a single shared target shape:
    #   - the calibration images,
    #   - the relative-mode colour baseline (color_analysis.base) and facies labels,
    #   - the CO2 mass analysis baseline (co2_mass_analysis.baseline). Its height_map is a
    #     @property of baseline.num_voxels, and temperature/pressure/solubility/density all
    #     derive from it, so re-running setup_density_gaseous_co2() after coarsening the
    #     baseline rebuilds all mass arrays at the coarse resolution. (My earlier mistake
    #     was coarsening solubility/density directly with cv2 on .img, which they then
    #     recomputed at full res from the un-coarsened baseline -> shape mismatch.)
    # geometry.integrate() rescales the voxel volume to the data shape itself, so the
    # integral stays mass-conservative without rebuilding the geometry.
    if quality_scale and float(quality_scale) != 1.0 and loaded:
        import darsia as _darsia
        import cv2 as _cv2
        s = float(quality_scale)
        H0, W0 = loaded[0][0].img.shape[:2]
        target = (max(1, int(round(H0 * s))), max(1, int(round(W0 * s))))

        def _rs(im, nearest=False):
            if im is None or getattr(im, "img", None) is None:
                return im
            return _darsia.resize(
                im, shape=target,
                interpolation="inter_nearest" if nearest else "inter_area",
            )

        # Each coarsening step runs in ISOLATION. Previously everything sat in one outer
        # try, so the first step that raised aborted all SUBSEQUENT steps - but the in-place
        # mutations already applied left a HALF-coarsened rig (some arrays coarse, some full
        # res), which is exactly what produces "operands could not be broadcast together".
        # Isolating each step guarantees every array is attempted independently, and the
        # per-step warning + final shape report make any remaining mismatch visible.
        def _step(name, fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] quality_scale step '%s' FAILED: %s", run, name, exc)

        # 1) Calibration images.
        def _do_images():
            for i, (im, inj, th) in enumerate(loaded):
                loaded[i] = (_rs(im), inj, th)

        # 2) Per-analysis base / labels / mask (color_analysis + signal_model).
        def _do_analyses():
            for _ca_attr in ("color_analysis", "signal_model"):
                _ca = getattr(cta, _ca_attr, None)
                if _ca is None:
                    continue
                if getattr(_ca, "base", None) is not None:
                    _ca.base = _rs(_ca.base)                    # relative-mode colour baseline
                if getattr(_ca, "labels", None) is not None:
                    _ca.labels = _rs(_ca.labels, nearest=True)  # facies labels (integer ids)
                _m = getattr(_ca, "mask", None)
                if isinstance(_m, np.ndarray) and _m.ndim >= 2 and _m.shape[:2] == (H0, W0):
                    _ca.mask = _cv2.resize(_m.astype(np.uint8), (target[1], target[0]),
                                           interpolation=_cv2.INTER_NEAREST).astype(_m.dtype)

        # 3) HeterogeneousModel.masks - the ACTUAL per-facies masking (darsia.Masks built
        #    from full-res labels at setup). Rebuilt from each model's own coarse labels.
        def _do_masks():
            def _rebuild(_model, _seen):
                if _model is None or id(_model) in _seen:
                    return
                _seen.add(id(_model))
                _mk = getattr(_model, "masks", None)
                if isinstance(_mk, _darsia.Masks) and getattr(_mk, "labels", None) is not None:
                    if _mk.labels.img.shape[:2] != target:
                        _model.masks = _darsia.Masks(_rs(_mk.labels, nearest=True))
                for _sub in (getattr(_model, "models", None) or []):
                    _rebuild(_sub, _seen)
                _obj = getattr(_model, "obj", None)
                if isinstance(_obj, dict):
                    for _sub in _obj.values():
                        _rebuild(_sub, _seen)
            for _mattr in ("color_analysis", "signal_model"):
                _mca = getattr(cta, _mattr, None)
                if _mca is not None:
                    _rebuild(getattr(_mca, "model", None), set())

        # 4) CO2 mass analysis: coarsen baseline (height_map / temperature / pressure all
        #    derive from baseline.num_voxels) then rebuild density + solubility. As a
        #    guaranteed fallback, if setup_density did not yield the target shape, resize
        #    the density/solubility maps directly (they vary smoothly with height, so
        #    INTER_AREA is physically faithful).
        # 4a) Best-effort: coarsen the CO2 baseline and let setup_density rebuild the maps
        #     from the coarse height_map. May raise if the baseline is not a plain 2d Image
        #     (darsia.resize asserts space_dim==2) - that is fine, step 4b is the guarantee.
        def _do_mass_baseline():
            cma = getattr(cta, "co2_mass_analysis", None)
            if cma is None:
                return
            if getattr(cma, "baseline", None) is not None:
                cma.baseline = _rs(cma.baseline)
            if hasattr(cma, "setup_density_gaseous_co2"):
                cma.setup_density_gaseous_co2()

        # 4b) GUARANTEED: directly resize the density/solubility maps to the target shape.
        #     Independent of 4a, so even if the baseline resize raised, the mass-side maps
        #     are still coarsened and the multiply density*s_g / solubility*c_aq matches.
        #     They vary smoothly with height, so INTER_AREA is physically faithful.
        def _do_mass_force():
            cma = getattr(cta, "co2_mass_analysis", None)
            if cma is None:
                return
            for _attr in ("density_gaseous_co2", "solubility_co2"):
                _arr = getattr(cma, _attr, None)
                if isinstance(_arr, np.ndarray) and _arr.ndim >= 2 and _arr.shape[:2] != target:
                    setattr(cma, _attr,
                            _cv2.resize(_arr.astype(np.float64), (target[1], target[0]),
                                        interpolation=_cv2.INTER_AREA).astype(_arr.dtype))

        # 3b) Restoration objects (e.g. VolumeAveraging) hold their OWN full-res mask Image
        #     plus derived mean_pore_volume / zero_indices. ConcentrationAnalysis._restore_signal
        #     multiplies the coarse signal by restoration.mask.img -> the (2979,5472) mismatch
        #     the traceback pinpointed. Coarsen the mask, scale the REV (in voxels) with the
        #     resolution, and recompute the derived maps exactly as VolumeAveraging.__init__.
        def _do_restoration():
            for _ca_attr in ("color_analysis", "signal_model"):
                _ca = getattr(cta, _ca_attr, None)
                if _ca is None:
                    continue
                _rest = getattr(_ca, "restoration", None)
                if _rest is None:
                    continue
                for _r in (_rest if isinstance(_rest, (list, tuple)) else [_rest]):
                    _mask = getattr(_r, "mask", None)
                    if _mask is None or getattr(_mask, "img", None) is None:
                        continue
                    if _mask.img.shape[:2] != (H0, W0):
                        continue
                    if isinstance(getattr(_r, "rev_size", None), (int, float)):
                        _r.rev_size = max(1, int(round(_r.rev_size * s)))
                    _r.mask = _rs(_mask)
                    if hasattr(_r, "_heterogeneous_uniform_filter"):
                        _r.mean_pore_volume = _r._heterogeneous_uniform_filter(
                            _r.mask.astype(float).img)
                        _r.zero_indices = np.where(_r.mean_pore_volume < 1e-12)

        _step("images", _do_images)
        _step("analyses_base_labels_mask", _do_analyses)
        _step("restoration", _do_restoration)
        _step("hetero_masks", _do_masks)
        _step("co2_mass_baseline", _do_mass_baseline)
        _step("co2_mass_force", _do_mass_force)

        # 5) Diagnostics via print() (stdout is captured in the worker console, unlike the
        #    module logger). Shows the ACTUAL post-coarsening shapes - any surviving full-res
        #    array is then immediately visible.
        try:
            cma = getattr(cta, "co2_mass_analysis", None)
            _ish = loaded[0][0].img.shape[:2] if loaded else None
            _b = getattr(getattr(cta, "color_analysis", None), "base", None)
            _bsh = _b.img.shape[:2] if _b is not None and getattr(_b, "img", None) is not None else None
            _dsh = getattr(getattr(cma, "density_gaseous_co2", None), "shape", None)
            _ssh = getattr(getattr(cma, "solubility_co2", None), "shape", None)
            _r = getattr(getattr(cta, "color_analysis", None), "restoration", None)
            _r0 = _r[0] if isinstance(_r, (list, tuple)) and _r else _r
            _rsh = getattr(getattr(getattr(_r0, "mask", None), "img", None), "shape", None)
            print(f"[QSCALE {run}] scale={s:.3f} target={target} | img={_ish} base={_bsh} "
                  f"density={_dsh[:2] if _dsh else None} solubility={_ssh[:2] if _ssh else None} "
                  f"rest_mask={_rsh[:2] if _rsh else None}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[QSCALE {run}] shape-report failed: {exc}", flush=True)

    # auto-match the param space to the signal-model resolution (num_segments+1 points)
    n_free_values = None
    try:
        probe = signal_labels[0] if signal_labels else (
            list(hetero.keys())[0] if hasattr(hetero, "keys") and list(hetero.keys()) else None)
        if probe is not None:
            n_free_values = max(1, len(list(hetero[probe].values)) - 1)
    except Exception:
        n_free_values = None
    param_space = build_param_space(run, bounds_map, signal_labels=signal_labels,
                                    per_label_params=per_label_params, use_facies=use_facies,
                                    n_free_values=n_free_values)

    return CalibrationContext(
        run=run, config=config, experiment=experiment, fluidflower=fluidflower,
        geometry=geometry, calibration=cta, calibration_images=list(ctx_raw.image_paths),
        reference_label=reference_label, signal_label=None, signal_labels=signal_labels,
        param_space=param_space, enforce_lower=enforce_lower,
        per_label_params=per_label_params, objective_integral=objective_integral,
        label_weights=label_weights, calibration_folder=calibration_folder, _loaded=loaded,
    )


_EVAL_TB_PRINTED = False


def evaluate_run(context: CalibrationContext, params: Dict[str, Any]) -> EvalResult:
    import numpy as np
    if not context._loaded:
        return EvalResult(
            objective=PENALTY_VALUE,
            feasible=False,
            metrics={},
            status="no-calibration-images",
            params=params,
        )
    apply_params(context.calibration, params, labels=context.signal_labels, np_module=np)
    total_err = 0.0
    feasible = True
    metrics: Dict[str, Metrics] = {}
    samples: List[Tuple[float, float, float]] = []  # (t_hours, injected, detected)
    for i, (img, injected, t_h) in enumerate(context._loaded):
        try:
            detected = float(context.geometry.integrate(context.calibration(img).mass))
        except Exception as exc:  # noqa: BLE001
            # One-shot full traceback to stdout so the EXACT file:line of a shape/broadcast
            # mismatch is visible (the status string only carries the message). Also dumps
            # geometry weight shapes, the remaining un-coarsened suspect under quality_scale.
            global _EVAL_TB_PRINTED
            if not _EVAL_TB_PRINTED:
                _EVAL_TB_PRINTED = True
                import traceback as _tb
                _tbstr = _tb.format_exc()   # capture ORIGINAL traceback BEFORE any re-call
                try:
                    _gv = getattr(context.geometry, "voxel_volume", None)
                    _cv = getattr(context.geometry, "cached_voxel_volume", None)
                    _ish = getattr(getattr(img, "img", None), "shape", None)
                    print(f"[EVALTB {context.run}] {type(exc).__name__}: {exc}\n"
                          f"  img={_ish} geometry={type(context.geometry).__name__} "
                          f"voxel_volume={getattr(_gv,'shape',None)} "
                          f"cached={getattr(_cv,'shape',None)}", flush=True)
                except Exception as _e2:
                    print(f"[EVALTB {context.run}] shape probe failed: {_e2}", flush=True)
                print(_tbstr, flush=True)   # the traceback pinpoints the exact file:line
            return EvalResult(objective=PENALTY_VALUE, feasible=False, metrics={},
                              status=f"eval-error:{exc}", params=params)
        if not (np.isfinite(detected) and np.isfinite(injected)):
            # extreme signal values can make the flash/mass field non-finite;
            # penalise so Optuna avoids this region rather than crashing on NaN.
            return EvalResult(objective=PENALTY_VALUE, feasible=False, metrics=metrics,
                              status="non-finite-mass", params=params)
        total_err += abs(detected - injected)
        if context.enforce_lower and detected > injected:
            feasible = False
        # key by time-since-start in hours so the calibration viewer plots a real
        # time axis (0.17h .. 48h), not image indices.
        metrics[f"{t_h:.3f}h"] = Metrics(injected_full=injected, total_full=detected)
        samples.append((float(t_h), float(injected), float(detected)))
    # --- Mass-conservation (drift) penalty, opt-in via --objective-integral drift[:LAMBDA] ---
    # Physics: after shut-in the cell is closed, so TRUE total mass is constant. But the BTB
    # indicator saturates at ~11 % of CO2 solubility (1.25 mM NaOH alkalinity + 0.73 mM BTB:
    # fully yellow at DIC ~3.9 mM vs ~34 mM saturation), so the dilute fringe of the dissolved
    # plume reads as saturated and detected mass tracks plume AREA, which keeps growing after
    # shut-in. Penalising the temporal total variation of detected mass over the post-injection
    # plateau rewards signal->c_aq maps that put their weight at the END of the colour path
    # (where the fringe contributes ~0), i.e. a FLAT detected-mass curve at the right level
    # rather than a drifting curve that is merely right on average. Metrics stay raw.
    mode = str(getattr(context, "objective_integral", "off") or "off").strip().lower()
    if mode.startswith("drift"):
        lam = 1.0
        if ":" in mode:
            try:
                lam = float(mode.split(":", 1)[1])
            except ValueError:
                lam = 1.0
        pts = sorted(samples)  # chronological
        if pts:
            inj_max = max(p[1] for p in pts)
            plateau = [p for p in pts if p[1] >= inj_max * (1.0 - 1e-9)]
            if len(plateau) >= 2:
                drift = sum(abs(b[2] - a[2]) for a, b in zip(plateau, plateau[1:]))
                total_err += lam * drift
    if not np.isfinite(total_err):
        return EvalResult(objective=PENALTY_VALUE, feasible=False, metrics=metrics,
                          status="non-finite-objective", params=params)
    return EvalResult(objective=float(total_err), feasible=feasible, metrics=metrics,
                      status="ok", params=params)


def save_best_calibration(context: CalibrationContext, best_params, out_folder=None) -> None:
    import numpy as np
    if not best_params:
        logger.warning("[%s] no best params to save", context.run); return
    apply_params(context.calibration, best_params, labels=context.signal_labels, np_module=np)
    folder = Path(context.calibration_folder) / "signal_model"
    folder.mkdir(parents=True, exist_ok=True)
    hetero = context.calibration.signal_model.model[1]
    for label in (hetero.keys() if hasattr(hetero, "keys") else []):
        try:
            hetero[label].save(folder / f"signal_model_{label}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] could not save signal_model_%s: %s", context.run, label, exc)
    logger.info("[%s] saved optimised signal models to %s", context.run, folder)
    if out_folder:
        try:
            Path(out_folder).mkdir(parents=True, exist_ok=True)
            (Path(out_folder) / "best_params.json").write_text(
                json.dumps(best_params, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass


# =========================================================================
# Standalone per-run optimisation (no queue) - for smoke tests / single machine
# =========================================================================
def _diagnose_baseline(context) -> None:
    """Log per-image injected vs detected mass for the seeded values, to locate NaNs."""
    import numpy as np
    apply_params(context.calibration, {}, labels=context.signal_labels, np_module=np)
    for i, (img, injected, t_h) in enumerate(context._loaded):
        det = float("nan"); nan_frac = -1.0
        try:
            mass = context.calibration(img).mass
            arr = np.asarray(getattr(mass, "img", mass), dtype=float)
            nan_frac = float(np.mean(~np.isfinite(arr))) if arr.size else -1.0
            det = float(context.geometry.integrate(mass))
        except Exception as exc:  # noqa: BLE001
            logger.info("[diag %s] img%d ERROR %s", context.run, i, exc); continue
        date = getattr(img, "date", None)
        logger.info("[diag %s] img%d date=%s injected=%s detected=%s mass_nan_frac=%.3f",
                    context.run, i, date, injected, det, nan_frac)


def optimize_per_run(context: CalibrationContext, max_iters: int, logs_dir: Path,
                     warmup_iters: int = 0):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    best = {"objective": float("inf"), "params": None}

    _diagnose_baseline(context)

    # Baseline = current (seeded) signal values (apply nothing). Anchors the search
    # and confirms the seeded calibration yields a finite objective.
    base = evaluate_run(context, {})
    logger.info("[%s] baseline (seeded) objective=%.6g feasible=%s status=%s",
                context.run, base.objective, base.feasible, base.status)
    if base.objective < best["objective"]:
        best["objective"] = base.objective; best["params"] = {}

    def _objective(trial):
        params = suggest_params_trial(trial, context.param_space)
        res = evaluate_run(context, params)
        obj = res.objective if (res.objective == res.objective) else PENALTY_VALUE  # NaN guard
        if obj < best["objective"]:
            best["objective"] = obj; best["params"] = params
        return obj

    study.optimize(_objective, n_trials=max(1, max_iters))
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_history_csv(logs_dir / f"auto_calibration_{context.run}.csv",
                      [{"iter": i, "objective": t.value} for i, t in enumerate(study.trials)])
    logger.info("[%s] best objective=%.6g over %d trials", context.run,
                best["objective"], len(study.trials))
    return best["params"], best["objective"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--mode", choices=["per-run"], default="per-run")
    p.add_argument("--max-iters", type=int, default=40)
    p.add_argument("--config-dir", type=str, default="config/run_ac")
    p.add_argument("--logs-dir", type=str, default=_default_calibration_log_root())
    p.add_argument("--ref-config", type=str, default=None)
    p.add_argument("--use-facies", action="store_true")
    p.add_argument("--per-label", action="store_true")
    p.add_argument("--use-last-best", action="store_true")
    p.add_argument("--enforce-lower", action="store_true")
    p.add_argument("--warmup-iters", type=int, default=0)
    p.add_argument("--warmup-levels", default=None)
    p.add_argument("--run-mode", default="serial")
    p.add_argument("--max-in-flight-per-run", type=int, default=0)
    p.add_argument("--objective-integral", default="off")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    from darsia.presets.workflows.rig import Rig
    config_dir = Path(args.config_dir); logs_dir = Path(args.logs_dir)
    for run in args.runs:
        ctx = build_context(run=run, config_dir=config_dir, rig_cls=Rig,
                            ref_config_path=args.ref_config,
                            use_facies=args.use_facies, per_label_params=args.per_label,
                            enforce_lower=args.enforce_lower,
                            objective_integral=args.objective_integral)
        params, obj = optimize_per_run(ctx, args.max_iters, logs_dir, args.warmup_iters)
        if params:
            save_best_calibration(ctx, params)
        print(f"DONE {run}: best objective={obj:.6g}")


if __name__ == "__main__":
    main()
