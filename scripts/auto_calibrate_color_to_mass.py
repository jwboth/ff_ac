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

try:
    from scripts.ffac_titration_flash import TitrationFlash
except ModuleNotFoundError:
    from ffac_titration_flash import TitrationFlash

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
_SHARED_SIGNAL_PARAM_RE = re.compile(r"^signal\.shared\.value(?P<idx>\d+)$")
_SIGNAL_GAIN_RE = re.compile(r"^signal\.label(?P<label>-?\d+)\.gain$")
_VALUE_RE = re.compile(r"(?:.*\.)?value(\d+)$", re.IGNORECASE)
_SHARED_SIGNAL_LABEL = -1
_RESIDUAL_GAS_SCORE_CLIP = 2.0


# =========================================================================
# Dataclasses (mirror the queue's expected shapes)
# =========================================================================
@dataclass
class Metrics:
    injected_full: float
    total_full: float
    gaseous_full: Optional[float] = None
    aqueous_full: Optional[float] = None


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
    signal_parameterization: str = "per-label"
    objective_integral: str = "off"
    phase_separation: str = "shared-signal"
    label_weights: Optional[Dict[int, float]] = None
    calibration_folder: Optional[Path] = None
    # preloaded (corrected image, injected_mass) pairs - read once, reused per trial
    _loaded: List[Tuple[Any, float, float]] = field(default_factory=list)  # (img, injected, t_hours)
    # Quantized, parameter-independent optical residual used by residual-gas.
    _gas_scores: List[np.ndarray] = field(default_factory=list)
    # Color-path projection is independent of all Optuna parameters and can be
    # computed once per frame instead of once per trial.
    _prepared_colors: List[Any] = field(default_factory=list)
    _evaluation_backend: str = "legacy"
    _cuda_evaluator: Any = None
    _opencl_evaluator: Any = None


# =========================================================================
# Param-name helpers (queue contract)
# =========================================================================
def _parse_signal_name(name: str) -> Optional[Tuple[int, int]]:
    m = _SIGNAL_PARAM_RE.match(name)
    if m:
        return int(m.group("label")), int(m.group("idx"))
    shared = _SHARED_SIGNAL_PARAM_RE.match(name)
    if shared:
        return _SHARED_SIGNAL_LABEL, int(shared.group("idx"))
    return None


def _normalise_signal_parameterization(mode: str | None) -> str:
    value = (
        mode
        or os.environ.get("FFAC_SIGNAL_PARAMETERIZATION")
        or "per-label"
    ).strip().lower().replace("_", "-")
    aliases = {
        "default": "per-label",
        "perlabel": "per-label",
        "shared": "shared-shape",
        "reduced": "shared-shape",
        "shared-gain": "shared-shape",
        "shared-shape-gain": "shared-shape",
    }
    value = aliases.get(value, value)
    if value not in {"per-label", "shared-shape"}:
        raise ValueError(
            f"Unknown signal parameterization {mode!r}; expected per-label or shared-shape."
        )
    return value


def _normalise_phase_separation(mode: str | None) -> str:
    value = (
        mode
        or os.environ.get("FFAC_PHASE_SEPARATION")
        or "shared-signal"
    ).strip().lower().replace("_", "-")
    aliases = {
        "off": "shared-signal",
        "none": "shared-signal",
        "default": "shared-signal",
        "shared": "shared-signal",
        "residual": "residual-gas",
        "optical-residual": "residual-gas",
        "separate-gas": "residual-gas",
    }
    value = aliases.get(value, value)
    if value not in {"shared-signal", "residual-gas"}:
        raise ValueError(
            f"Unknown phase separation {mode!r}; expected shared-signal or residual-gas."
        )
    return value


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


def _rebuild_shared_shape_signal(
    template: Sequence[Dict[str, Any]],
    n_free_values: int,
    signal_labels: Sequence[int],
) -> List[Dict[str, Any]]:
    """Build an identifiable shared curve shape with one amplitude per facies.

    The shared shape starts at zero and ends at one. Intermediate values are
    monotone fractions in [0, 1]. Multiplying by each facies amplitude therefore
    keeps every effective signal value in the established [0, 2] range without
    a free shape/amplitude scale degeneracy.
    """

    other = [e for e in template if not _parse_signal_name(e["name"])]
    count = max(1, int(n_free_values))
    shared: List[Dict[str, Any]] = []
    for idx in range(count + 1):
        if idx == 0:
            bounds = (0.0, 0.0)
        elif idx == count:
            bounds = (1.0, 1.0)
        else:
            bounds = (0.0, 1.0)
        shared.append(
            {
                "name": f"signal.shared.value{idx}",
                "attr_path": ["signal_model", "shared_shape", "values", idx],
                "bounds": bounds,
                "type": "float",
            }
        )

    gains = [
        {
            "name": f"signal.label{int(label)}.gain",
            "attr_path": ["signal_model", "model", 1, int(label), "gain"],
            "bounds": (0.0, 2.0),
            "type": "float",
        }
        for label in sorted({int(label) for label in signal_labels})
    ]
    return [*shared, *gains, *copy.deepcopy(other)]


def build_param_space(run, bounds_map, signal_label=None, signal_labels=None,
                      per_label_params=False, use_facies=True, n_free_values=None,
                      signal_parameterization="per-label"):
    parameterization = _normalise_signal_parameterization(signal_parameterization)
    base = copy.deepcopy(PARAM_SPACE_TEMPLATE)
    if n_free_values is not None:
        base = _rebuild_template_signal(base, int(n_free_values))
    override = (bounds_map or {}).get(run, {})
    default_override = (bounds_map or {}).get("default", {})
    space: List[Dict[str, Any]] = []
    if parameterization == "shared-shape":
        labels = list(signal_labels or ([signal_label] if signal_label is not None else []))
        free_values = int(n_free_values) if n_free_values is not None else max(
            (_parse_signal_name(e["name"])[1] for e in base if _parse_signal_name(e["name"])),
            default=1,
        )
        space = _rebuild_shared_shape_signal(base, free_values, labels)
    elif per_label_params:
        sig = [e for e in base if _parse_signal_name(e["name"])]
        other = [e for e in base if not _parse_signal_name(e["name"])]
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


def sample_params(
    param_space: Sequence[Dict[str, Any]],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    random_source = rng or random
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
                val = min_i if max_i <= min_i else (
                    min_i + random_source.random() * (max_i - min_i)
                )
            samples[entry["name"]] = val
            prev = val
    for entry in param_space:
        if entry["name"] in samples:
            continue
        low, high = entry["bounds"]
        samples[entry["name"]] = (int(round(random_source.uniform(low, high)))
                                  if entry.get("type", "float") == "int"
                                  else random_source.uniform(low, high))
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


_ANCHOR_BASELINE_MEDIAN_CACHE: Dict[Tuple[str, Tuple[int, ...], int], np.ndarray] = {}


def _porous_baseline_median(
    rgb: np.ndarray,
    labels: np.ndarray,
    active_labels: Sequence[int],
    *,
    stride: int = 8,
) -> np.ndarray:
    """Robust RGB median in active porous facies, sampled for low memory use."""

    arr = np.asarray(rgb)
    facies = np.asarray(labels)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected RGB baseline, got shape {arr.shape}")
    if facies.shape[:2] != arr.shape[:2]:
        raise ValueError(
            f"Baseline/facies shape mismatch: {arr.shape[:2]} vs {facies.shape[:2]}"
        )
    step = max(1, int(stride))
    sampled = np.asarray(arr[::step, ::step, :3], dtype=np.float32)
    sampled_labels = facies[::step, ::step]
    mask = np.isin(sampled_labels, np.asarray(active_labels, dtype=int))
    mask &= np.all(np.isfinite(sampled), axis=2)
    if int(np.count_nonzero(mask)) < 1000:
        raise ValueError(
            f"Only {int(np.count_nonzero(mask))} active baseline samples found"
        )
    return np.asarray(np.median(sampled[mask], axis=0), dtype=np.float32)


def _compute_anchor_photometric_gain(
    target_median: np.ndarray,
    anchor_median: np.ndarray,
    *,
    gain_low: float = 0.5,
    gain_high: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    target = np.maximum(np.asarray(target_median, dtype=np.float32), 1e-6)
    anchor = np.maximum(np.asarray(anchor_median, dtype=np.float32), 1e-6)
    raw = anchor / target
    return raw, np.clip(raw, float(gain_low), float(gain_high)).astype(np.float32)


def _apply_anchor_photometric_gain(image: Any, gain: np.ndarray) -> None:
    gain_arr = np.asarray(gain, dtype=np.float32).reshape(1, 1, 3)
    if np.allclose(gain_arr, 1.0, rtol=0.0, atol=1e-7):
        return
    source = np.asarray(image.img)
    arr = np.asarray(source, dtype=np.float32).copy()
    arr[..., :3] = np.clip(arr[..., :3] * gain_arr, 0.0, 1.0)
    image.img = arr.astype(source.dtype, copy=False)


def _setup_anchor_photometric_gain(
    *,
    run: str,
    config: Any,
    fluidflower: Any,
    active_labels: Sequence[int],
    master_light_context: bool,
) -> Optional[np.ndarray]:
    """Derive one global RGB gain from target and anchor pre-injection baselines."""

    anchor_run = os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_RUN", "").strip().lower()
    if anchor_run in {"", "0", "false", "no", "none", "off"}:
        return None
    if master_light_context:
        logger.info(
            "[%s] photometric anchor=%s deferred to workers (light master)",
            run,
            anchor_run,
        )
        return None

    try:
        results_dir = Path(getattr(getattr(config, "data", None), "results"))
        results_root = results_dir.parent
        target_rig = results_dir / "setup" / "rig"
        anchor_rig = results_root / anchor_run / "setup" / "rig"
        target_labels_path = target_rig / "facies.npz"
        anchor_labels_path = anchor_rig / "facies.npz"
        anchor_baseline_path = anchor_rig / "shape_corrected_baseline.npz"
        for required in (target_labels_path, anchor_labels_path, anchor_baseline_path):
            if not required.exists():
                raise FileNotFoundError(required)

        stride = max(1, int(os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_STRIDE", "8")))
        labels_tuple = tuple(sorted(int(label) for label in active_labels))
        target_labels = np.load(target_labels_path, allow_pickle=True)["array"]
        target_rgb = np.asarray(
            getattr(fluidflower, "shape_corrected_baseline").img,
            dtype=np.float32,
        )
        target_median = _porous_baseline_median(
            target_rgb,
            target_labels,
            labels_tuple,
            stride=stride,
        )

        cache_key = (str(anchor_baseline_path).lower(), labels_tuple, stride)
        anchor_median = _ANCHOR_BASELINE_MEDIAN_CACHE.get(cache_key)
        if anchor_median is None:
            anchor_rgb = _registration_baseline_array(anchor_baseline_path)
            anchor_labels = np.load(anchor_labels_path, allow_pickle=True)["array"]
            anchor_median = _porous_baseline_median(
                anchor_rgb,
                anchor_labels,
                labels_tuple,
                stride=stride,
            )
            _ANCHOR_BASELINE_MEDIAN_CACHE[cache_key] = anchor_median

        gain_low = float(os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_GAIN_MIN", "0.5"))
        gain_high = float(os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_GAIN_MAX", "2.0"))
        if not 0.0 < gain_low <= gain_high:
            raise ValueError(
                f"Invalid photometric anchor gain limits [{gain_low}, {gain_high}]"
            )
        raw_gain, gain = _compute_anchor_photometric_gain(
            target_median,
            anchor_median,
            gain_low=gain_low,
            gain_high=gain_high,
        )
        logger.info(
            "[%s] photometric anchor ACTIVE target=%s anchor=%s mode=baseline-diagonal "
            "target_rgb=%s anchor_rgb=%s raw_gain=%s gain=%s",
            run,
            run,
            anchor_run,
            np.round(target_median, 5).tolist(),
            np.round(anchor_median, 5).tolist(),
            np.round(raw_gain, 5).tolist(),
            np.round(gain, 5).tolist(),
        )
        return gain
    except Exception as exc:  # noqa: BLE001
        strict = os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_STRICT", "").strip().lower()
        if strict in {"1", "true", "yes", "on"}:
            raise
        logger.warning("[%s] photometric anchor requested but skipped: %s", run, exc)
        return None


def _template_registration_target() -> str:
    value = os.environ.get("FFAC_TEMPLATE_REGISTRATION", "").strip().lower()
    if value in {"", "0", "false", "no", "none", "off"}:
        return ""
    return value


def _registration_baseline_array(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "array" in data:
        arr = data["array"]
    elif "img" in data:
        arr = data["img"]
    else:
        raise KeyError(f"{path} does not contain an image array")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3]


def _registration_gray(arr: np.ndarray, *, max_col: int, scale: float) -> np.ndarray:
    import cv2

    cropped = arr[:, : min(max_col, arr.shape[1]), :3]
    cropped = np.clip(cropped, 0.0, 1.0)
    gray = cv2.cvtColor((cropped * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(gray)


def _estimate_template_affine(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    mode: str,
    scale: float,
    max_col: int,
    max_features: int,
    keep_matches: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Estimate source->template registration from baseline images.

    The default mode is a partial affine transform, i.e. translation + rotation +
    uniform scale. This is intentionally less flexible than a homography so the
    mass integral is not changed by arbitrary perspective/shear deformation.
    """

    import cv2
    import math

    src_gray = _registration_gray(src, max_col=max_col, scale=scale)
    dst_gray = _registration_gray(dst, max_col=max_col, scale=scale)

    detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=8)
    kp_src, des_src = detector.detectAndCompute(src_gray, None)
    kp_dst, des_dst = detector.detectAndCompute(dst_gray, None)
    if des_src is None or des_dst is None or len(kp_src) < 10 or len(kp_dst) < 10:
        raise RuntimeError("too few registration features")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_src, des_dst), key=lambda m: m.distance)
    matches = matches[: max(10, keep_matches)]
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    mode = mode.strip().lower().replace("-", "_")
    if mode in {"similarity", "partial", "partial_affine", "affine"}:
        mat, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.995,
        )
        if mat is None:
            raise RuntimeError("partial affine registration failed")
        mat = mat.astype(np.float32)
        mat[:, 2] /= float(scale)
        a, b, tx = [float(x) for x in mat[0]]
        c, d, ty = [float(x) for x in mat[1]]
        reg_scale = math.sqrt(a * a + c * c)
        angle = math.degrees(math.atan2(c, a))
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        return mat, {
            "matches": float(len(matches)),
            "inliers": float(inlier_count),
            "dx_px": tx,
            "dy_px": ty,
            "scale": reg_scale,
            "angle_deg": angle,
        }
    if mode == "translation":
        # Use partial affine for robust matching, then keep only the translation.
        mat, stats = _estimate_template_affine(
            src,
            dst,
            mode="partial_affine",
            scale=scale,
            max_col=max_col,
            max_features=max_features,
            keep_matches=keep_matches,
        )
        out = np.array([[1.0, 0.0, mat[0, 2]], [0.0, 1.0, mat[1, 2]]], dtype=np.float32)
        stats["scale"] = 1.0
        stats["angle_deg"] = 0.0
        return out, stats
    if mode == "homography":
        hom, inliers = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        if hom is None:
            raise RuntimeError("homography registration failed")
        s = float(scale)
        to_small = np.diag([s, s, 1.0])
        to_full = np.diag([1.0 / s, 1.0 / s, 1.0])
        hom_full = (to_full @ hom @ to_small).astype(np.float32)
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        return hom_full, {
            "matches": float(len(matches)),
            "inliers": float(inlier_count),
            "dx_px": float(hom_full[0, 2]),
            "dy_px": float(hom_full[1, 2]),
            "scale": float(np.sqrt(abs(np.linalg.det(hom_full[:2, :2])))),
            "angle_deg": float("nan"),
        }
    raise ValueError(
        f"Unknown FFAC_TEMPLATE_REGISTRATION_MODE={mode!r}; expected partial_affine, translation, or homography."
    )


def _warp_registration_image(img: Any, transform: np.ndarray, *, mode: str) -> Any:
    import cv2

    arr = np.asarray(img.img)
    height, width = arr.shape[:2]
    if mode.strip().lower().replace("-", "_") == "homography":
        warped = cv2.warpPerspective(
            arr,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    else:
        warped = cv2.warpAffine(
            arr,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    img.img = warped.astype(arr.dtype, copy=False)
    return img


def _setup_template_registration(
    *,
    run: str,
    config: Any,
    fluidflower: Any,
    calibration: Any,
) -> Tuple[Optional[np.ndarray], str]:
    template_run = _template_registration_target()
    if not template_run or template_run == run.lower():
        return None, "off"

    mode = os.environ.get("FFAC_TEMPLATE_REGISTRATION_MODE", "partial_affine")
    scale = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_SCALE", "0.25"))
    max_col = int(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MAX_COL", "4800"))
    max_features = int(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MAX_FEATURES", "12000"))
    keep_matches = int(os.environ.get("FFAC_TEMPLATE_REGISTRATION_KEEP_MATCHES", "800"))
    min_inlier_frac = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MIN_INLIER_FRAC", "0.50"))
    max_abs_shift = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MAX_SHIFT_PX", "500"))
    min_scale = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MIN_SCALE", "0.90"))
    max_scale = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MAX_SCALE", "1.10"))

    try:
        results_dir = Path(getattr(getattr(config, "data", None), "results"))
        template_dir = results_dir.parent / template_run
        template_path = template_dir / "setup" / "rig" / "shape_corrected_baseline.npz"
        if not template_path.exists():
            raise FileNotFoundError(template_path)
        src = np.asarray(getattr(fluidflower, "shape_corrected_baseline").img, dtype=np.float32)[..., :3]
        dst = _registration_baseline_array(template_path)
        transform, stats = _estimate_template_affine(
            src,
            dst,
            mode=mode,
            scale=scale,
            max_col=max_col,
            max_features=max_features,
            keep_matches=keep_matches,
        )
        inlier_frac = stats["inliers"] / max(stats["matches"], 1.0)
        if inlier_frac < min_inlier_frac:
            raise RuntimeError(f"low inlier fraction {inlier_frac:.3f}")
        if abs(stats["dx_px"]) > max_abs_shift or abs(stats["dy_px"]) > max_abs_shift:
            raise RuntimeError(f"large shift dx={stats['dx_px']:.1f}, dy={stats['dy_px']:.1f}")
        if mode.strip().lower().replace("-", "_") != "homography":
            if not (min_scale <= stats["scale"] <= max_scale):
                raise RuntimeError(f"scale {stats['scale']:.5f} outside [{min_scale}, {max_scale}]")

        color_base = getattr(getattr(calibration, "color_analysis", None), "base", None)
        if color_base is not None:
            _warp_registration_image(color_base, transform, mode=mode)

        logger.info(
            "[%s] template registration ACTIVE target=%s mode=%s "
            "matches=%d inliers=%d frac=%.3f dx=%.1f dy=%.1f scale=%.5f angle=%.3f",
            run,
            template_run,
            mode,
            int(stats["matches"]),
            int(stats["inliers"]),
            inlier_frac,
            stats["dx_px"],
            stats["dy_px"],
            stats["scale"],
            stats["angle_deg"],
        )
        return transform, mode
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("FFAC_TEMPLATE_REGISTRATION_STRICT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise
        logger.warning("[%s] template registration requested but skipped: %s", run, exc)
        return None, "off"


def _path_arc_length(relative_colors: Sequence[np.ndarray]) -> float:
    colors = np.asarray(relative_colors, dtype=np.float64)
    if colors.ndim != 2 or len(colors) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(colors, axis=0), axis=1)))


def _resample_relative_path(color_path: Any, count: int) -> np.ndarray:
    colors = np.asarray(color_path.relative_colors, dtype=np.float64)
    source = np.asarray(color_path.relative_distances, dtype=np.float64)
    target = np.linspace(0.0, 1.0, int(count))
    return np.column_stack(
        [np.interp(target, source, colors[:, channel]) for channel in range(3)]
    )


def _regularize_color_paths(
    *,
    run: str,
    config: Any,
    calibration: Any,
    calibration_folder: Path,
    labels: Sequence[int],
    anchor_run: str | None = None,
    anchor_weight: float | None = None,
    anchor_strict: bool | None = None,
) -> None:
    """Blend noisy per-run path shapes toward a shared, independently observed path.

    The current run keeps its own baseline and path amplitude. Only the normalized
    RGB trajectory is shrunk toward the anchor, so colour correction remains
    run-specific while implausible path tortuosity is regularized.
    """

    anchor_run = (
        anchor_run
        if anchor_run is not None
        else os.environ.get("FFAC_COLOR_PATH_ANCHOR", "")
    ).strip().lower()
    if not anchor_run or anchor_run in {"off", "none"}:
        return
    weight = float(
        anchor_weight
        if anchor_weight is not None
        else os.environ.get("FFAC_COLOR_PATH_ANCHOR_WEIGHT", "0.75")
    )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(
            f"FFAC_COLOR_PATH_ANCHOR_WEIGHT must be within [0, 1], got {weight}"
        )
    if anchor_run == run.lower() or weight == 0.0:
        logger.info(
            "[%s] color-path anchor is the current run; no regularization needed.",
            run,
        )
        return

    try:
        import darsia

        results_dir = Path(getattr(getattr(config, "data", None), "results"))
        relative_folder = calibration_folder.relative_to(results_dir)
        anchor_folder = (
            results_dir.parent
            / anchor_run
            / relative_folder
            / "color_path_interpretation"
        )
        changed: list[str] = []
        for label in labels:
            target_interpolation = calibration.color_path_interpretation[int(label)]
            anchor_path = (
                anchor_folder / f"color_path_interpretation_{int(label)}"
            )
            anchor_interpolation = darsia.ColorPathInterpolation.load(anchor_path)
            target_path = target_interpolation.color_path
            count = len(target_path.relative_colors)
            target_colors = _resample_relative_path(target_path, count)
            anchor_colors = _resample_relative_path(
                anchor_interpolation.color_path,
                count,
            )
            target_arc = _path_arc_length(target_colors)
            anchor_arc = _path_arc_length(anchor_colors)
            if target_arc <= 1e-12 or anchor_arc <= 1e-12:
                raise ValueError(
                    f"label {label}: zero-length target/anchor color path"
                )
            anchor_colors *= target_arc / anchor_arc
            blended = (1.0 - weight) * target_colors + weight * anchor_colors
            blended[0] = 0.0
            regularized_path = darsia.ColorPath(
                base_color=np.asarray(target_path.base_color, dtype=np.float64),
                relative_colors=[row for row in blended],
                mode=target_path.mode,
                name=f"{target_path.name}|anchor={anchor_run}|weight={weight:g}",
            )
            target_interpolation.color_path = regularized_path
            # HeterogeneousModel shallow-copies interpolation objects at setup.
            # Keep the executable copy synchronized with the public mapping.
            calibration.color_analysis.model[0][int(label)].color_path = (
                regularized_path
            )
            changed.append(
                f"{label}:arc={target_arc:.4g},anchor_scale={target_arc / anchor_arc:.4g}"
            )
        logger.info(
            "[%s] color-path regularization ACTIVE anchor=%s weight=%.3f [%s]",
            run,
            anchor_run,
            weight,
            "; ".join(changed),
        )
    except Exception as exc:  # noqa: BLE001
        strict = (
            bool(anchor_strict)
            if anchor_strict is not None
            else os.environ.get(
                "FFAC_COLOR_PATH_ANCHOR_STRICT", ""
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if strict:
            raise
        logger.warning(
            "[%s] color-path regularization requested but skipped: %s",
            run,
            exc,
        )


def _optical_residual_gas_score(
    calibration: Any,
    image: Any,
    labels: Sequence[int],
) -> np.ndarray:
    """Return a quantized gas observable independent of color-path position.

    Aqueous indicator changes should stay close to the calibrated RGB path.
    Gas bubbles and gas-water interfaces add reflection/scattering and local
    high-frequency structure, producing an off-path optical residual.
    """

    import cv2

    observed = np.asarray(image.img, dtype=np.float32)[..., :3]
    base = getattr(getattr(calibration, "color_analysis", None), "base", None)
    if base is not None:
        observed = observed - np.asarray(base.img, dtype=np.float32)[..., :3]
    label_image = np.asarray(calibration.labels.img)
    if label_image.shape != observed.shape[:2]:
        raise ValueError(
            "Residual-gas labels/image shape mismatch: "
            f"{label_image.shape} != {observed.shape[:2]}"
        )

    gray = np.mean(observed, axis=2, dtype=np.float32)
    sigma = max(
        0.1,
        float(os.environ.get("FFAC_RESIDUAL_GAS_TEXTURE_SIGMA", "2.0")),
    )
    texture_weight = max(
        0.0,
        float(os.environ.get("FFAC_RESIDUAL_GAS_TEXTURE_WEIGHT", "0.5")),
    )
    texture = np.abs(
        gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    )
    score = np.zeros(label_image.shape, dtype=np.float32)
    flat_observed = observed.reshape(-1, 3)
    flat_texture = texture.reshape(-1)
    flat_score = score.reshape(-1)
    chunk_size = max(
        10_000,
        int(os.environ.get("FFAC_RESIDUAL_GAS_CHUNK_PIXELS", "250000")),
    )

    for label in labels:
        interpolation = calibration.color_analysis.model[0][int(label)]
        color_path = interpolation.color_path
        positions = np.flatnonzero(label_image.reshape(-1) == int(label))
        if not positions.size:
            continue
        path_scale = max(
            1e-6,
            float(
                np.sum(
                    np.linalg.norm(
                        np.diff(
                            np.asarray(
                                color_path.relative_colors,
                                dtype=np.float32,
                            ),
                            axis=0,
                        ),
                        ord=1,
                        axis=1,
                    )
                )
            ),
        )
        for start in range(0, positions.size, chunk_size):
            index = positions[start : start + chunk_size]
            colors = flat_observed[index]
            parameters = color_path.fit(
                colors=colors,
                color_mode=interpolation.color_mode,
                mode="equidistant",
            )
            projected = color_path.interpret(
                parameters,
                color_mode=interpolation.color_mode,
                mode="equidistant",
            )
            off_path = np.linalg.norm(colors - projected, ord=1, axis=1)
            flat_score[index] = (
                off_path + texture_weight * flat_texture[index]
            ) / path_scale

    return np.rint(
        np.clip(score / _RESIDUAL_GAS_SCORE_CLIP, 0.0, 1.0) * 255.0
    ).astype(np.uint8)


def _precompute_residual_gas_scores(
    calibration: Any,
    loaded: Sequence[Tuple[Any, float, float]],
    labels: Sequence[int],
    *,
    run: str,
) -> List[np.ndarray]:
    scores: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    active = np.isin(np.asarray(calibration.labels.img), np.asarray(labels))
    active_flat = active.reshape(-1)
    for image, _injected, _time_h in loaded:
        quantized = _optical_residual_gas_score(calibration, image, labels)
        scores.append(quantized)
        values = quantized.reshape(-1)[active_flat][::200]
        if values.size:
            samples.append(values)
    if samples:
        sampled = (
            np.concatenate(samples).astype(np.float32)
            * (_RESIDUAL_GAS_SCORE_CLIP / 255.0)
        )
        percentiles = np.percentile(sampled, [50.0, 90.0, 99.0, 99.9])
        logger.info(
            "[%s] residual-gas optical score ACTIVE; p50/p90/p99/p99.9=%s",
            run,
            np.round(percentiles, 4).tolist(),
        )
    return scores


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
    gains: Dict[int, float] = {}
    global_idx: Dict[int, float] = {}
    for k, v in params.items():
        parsed = _parse_signal_name(k)
        if parsed:
            label, idx = parsed
            per_label.setdefault(label, {})[idx] = float(v)
            continue
        gain = _SIGNAL_GAIN_RE.match(k)
        if gain:
            gains[int(gain.group("label"))] = float(v)
            continue
        mv = _VALUE_RE.match(k)
        if mv and not k.lower().startswith("flash"):
            global_idx[int(mv.group(1))] = float(v)
    if labels is None:
        labels = list(hetero.keys()) if hasattr(hetero, "keys") else []
    shared_shape = per_label.get(_SHARED_SIGNAL_LABEL, {})
    updated = 0
    for lbl in labels:
        idx_vals = dict(global_idx)
        if shared_shape:
            amplitude = gains.get(int(lbl), 1.0)
            idx_vals.update(
                {idx: float(fraction) * amplitude for idx, fraction in shared_shape.items()}
            )
        idx_vals.update(per_label.get(int(lbl), {}))
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
                  static_light_correction=None, signal_parameterization=None,
                  phase_separation=None,
                  color_path_anchor=None,
                  color_path_anchor_weight=None,
                  color_path_anchor_strict=None,
                  evaluation_times_hours=None,
                  evaluation_time_tolerance_seconds=600.0) -> CalibrationContext:
    import numpy as np
    signal_parameterization = _normalise_signal_parameterization(signal_parameterization)
    phase_separation = _normalise_phase_separation(phase_separation)
    from darsia.presets.workflows.analysis.analysis_context import (
        prepare_analysis_context,
        select_image_paths,
    )

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
    _regularize_color_paths(
        run=run,
        config=config,
        calibration=cta,
        calibration_folder=calibration_folder,
        labels=signal_labels,
        anchor_run=color_path_anchor,
        anchor_weight=color_path_anchor_weight,
        anchor_strict=color_path_anchor_strict,
    )

    light_master = os.environ.get("FFAC_MASTER_LIGHT_CONTEXT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    anchor_photometric_gain = _setup_anchor_photometric_gain(
        run=run,
        config=config,
        fluidflower=fluidflower,
        active_labels=signal_labels,
        master_light_context=light_master,
    )
    if anchor_photometric_gain is not None:
        color_base = getattr(getattr(cta, "color_analysis", None), "base", None)
        if color_base is not None:
            _apply_anchor_photometric_gain(color_base, anchor_photometric_gain)
    if light_master:
        template_registration, template_registration_mode = None, "off"
    else:
        template_registration, template_registration_mode = _setup_template_registration(
            run=run,
            config=config,
            fluidflower=fluidflower,
            calibration=cta,
        )

    # Calibration image selection belongs to [calibration.mass], independently
    # of the image schedule used by subsequent analyses. Historically this used
    # ctx_raw.image_paths ([analysis].data), which made additions to the declared
    # mass-calibration schedule ineffective.
    selected_image_paths = list(ctx_raw.image_paths)
    calibration_mass_config = getattr(
        getattr(config, "calibration", None),
        "mass",
        None,
    )
    if calibration_mass_config is not None:
        mass_data = getattr(calibration_mass_config, "data", None)
        if hasattr(mass_data, "get_times_with_uncertainty"):
            selected_image_paths = list(
                experiment.find_images_for_paths(
                    paths=list(getattr(mass_data, "image_paths", []) or [])
                )
            )
            for requested_time, tolerance_hours in (
                mass_data.get_times_with_uncertainty()
            ):
                path = experiment.find_images_for_times(
                    float(requested_time),
                    tol=float(tolerance_hours) * 3600.0,
                )
                if path is None:
                    logger.warning(
                        "[%s] no calibration image within %.1f min of %.3f h; "
                        "dropping this unavailable point",
                        run,
                        float(tolerance_hours) * 60.0,
                        float(requested_time),
                    )
                    continue
                path = Path(path)
                if path in selected_image_paths:
                    logger.warning(
                        "[%s] calibration time %.3f h resolves to duplicate %s; "
                        "keeping the image only once",
                        run,
                        float(requested_time),
                        path.name,
                    )
                    continue
                selected_image_paths.append(path)
            selected_image_paths.sort(key=experiment.get_datetime)
        else:
            selected_image_paths = select_image_paths(
                config,
                experiment,
                all=False,
                sub_config=calibration_mass_config,
                data_registry=config.data.registry,
            )
    if not selected_image_paths:
        raise ValueError(f"[{run}] no mass-calibration images were resolved")
    training_image_paths = list(selected_image_paths)
    if evaluation_times_hours is not None:
        selected_image_paths = []
        for requested_time in evaluation_times_hours:
            path = experiment.find_images_for_times(
                float(requested_time),
                tol=float(evaluation_time_tolerance_seconds),
            )
            if path is None:
                raise ValueError(
                    f"[{run}] no image within {evaluation_time_tolerance_seconds:g} s "
                    f"of holdout time {float(requested_time):g} h"
                )
            selected_image_paths.append(Path(path))
        if len(set(selected_image_paths)) != len(selected_image_paths):
            raise ValueError(f"[{run}] holdout times resolved to duplicate image paths")
        training_paths = {
            str(Path(path)).lower() for path in training_image_paths
        }
        overlap = [
            path
            for path in selected_image_paths
            if str(Path(path)).lower() in training_paths
        ]
        if overlap:
            raise ValueError(
                f"[{run}] holdout selection overlaps calibration frame(s): "
                + ", ".join(path.name for path in overlap)
            )

    # preload corrected calibration/holdout images + param-independent injected mass
    loaded: List[Tuple[Any, float, float]] = []
    exp_start = getattr(experiment, "experiment_start", None)

    # Nearest-good-neighbour substitution. read_image applies the rig corrections incl.
    # ColorCorrection, which sets last_flagged when a frame's lighting/checker is too far
    # gone to recover. For calibration we need sparse, well-spread points, not these exact
    # frames, so a flagged frame is replaced by the nearest-in-time correctable neighbour
    # (frames are ~5 min apart and the CO2 state barely changes over a few minutes). If no
    # neighbour within the window is usable, the calibration point is dropped. When colour
    # correction is OFF, color_corrections is empty -> _flagged is always False -> this
    # behaves exactly as a plain read (no neighbour I/O, current behaviour unchanged).
    def _read_and_flag(path):
        im = fluidflower.read_image(Path(path))
        if template_registration is not None:
            im = _warp_registration_image(
                im,
                template_registration,
                mode=template_registration_mode,
            )
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

    if light_master:
        logger.info(
            "[%s] FFAC_MASTER_LIGHT_CONTEXT=on -> skipping calibration image preload",
            run,
        )
    else:
        for p in selected_image_paths:
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
        if anchor_photometric_gain is not None:
            for image, _injected, _t_h in loaded:
                _apply_anchor_photometric_gain(image, anchor_photometric_gain)
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
                                    n_free_values=n_free_values,
                                    signal_parameterization=signal_parameterization)
    if phase_separation == "residual-gas":
        override = (bounds_map or {}).get(run, {})
        default_override = (bounds_map or {}).get("default", {})
        for entry in (
            {
                "name": "gas.residual_onset",
                "attr_path": ["phase_separation", "residual_onset"],
                "bounds": (0.01, 0.80),
                "type": "float",
            },
            {
                "name": "gas.residual_width",
                "attr_path": ["phase_separation", "residual_width"],
                "bounds": (0.02, 1.50),
                "type": "float",
            },
        ):
            bounds = _match_bounds(entry["name"], override, default_override)
            if bounds is not None:
                entry["bounds"] = tuple(bounds)
            param_space.append(entry)

    gas_scores: List[np.ndarray] = []
    if phase_separation == "residual-gas" and loaded:
        gas_scores = _precompute_residual_gas_scores(
            cta,
            loaded,
            signal_labels,
            run=run,
        )

    return CalibrationContext(
        run=run, config=config, experiment=experiment, fluidflower=fluidflower,
        geometry=geometry, calibration=cta, calibration_images=selected_image_paths,
        reference_label=reference_label, signal_label=None, signal_labels=signal_labels,
        param_space=param_space, enforce_lower=enforce_lower,
        per_label_params=per_label_params,
        signal_parameterization=signal_parameterization,
        objective_integral=objective_integral,
        phase_separation=phase_separation,
        label_weights=label_weights, calibration_folder=calibration_folder, _loaded=loaded,
        _gas_scores=gas_scores,
    )


_EVAL_TB_PRINTED = False


def _normalise_evaluation_backend(mode: str | None) -> str:
    value = (mode or "prepared").strip().lower().replace("_", "-")
    aliases = {
        "cpu": "prepared",
        "fast": "prepared",
        "numpy": "prepared",
        "off": "legacy",
        "ocl": "opencl",
    }
    value = aliases.get(value, value)
    if value not in {"legacy", "prepared", "cuda", "opencl"}:
        raise ValueError(
            f"Unknown evaluation backend {mode!r}; expected legacy, prepared, "
            "cuda, or opencl."
        )
    return value


class _CudaIntegratedEvaluator:
    """Evaluate integrated masses while keeping prepared frame data in GPU memory."""

    def __init__(self, context: CalibrationContext) -> None:
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CUDA evaluation requires the optional 'cuda' dependency"
            ) from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CUDA evaluation requested, but no CUDA device is available")
        if not context._prepared_colors:
            raise RuntimeError("CUDA evaluation requires prepared color frames")

        calibration = context.calibration
        flash = calibration.flash
        if getattr(flash, "restoration", None) is not None:
            raise RuntimeError("CUDA evaluation does not support flash restoration")

        shape = tuple(np.asarray(context._prepared_colors[0].img).shape[:2])
        context.geometry._prepare_cached_voxel_volume(list(shape))
        voxel_volume = np.asarray(context.geometry.cached_voxel_volume)
        density = np.asarray(calibration.co2_mass_analysis.density_gaseous_co2)
        solubility = np.asarray(calibration.co2_mass_analysis.solubility_co2)
        try:
            voxel_volume = np.broadcast_to(voxel_volume, shape)
            density = np.broadcast_to(density, shape)
            solubility = np.broadcast_to(solubility, shape)
        except ValueError as exc:
            raise RuntimeError(
                f"CUDA static-array shape mismatch for {context.run}: "
                f"shape={shape}, voxel={voxel_volume.shape}, density={density.shape}, "
                f"solubility={solubility.shape}"
            ) from exc

        labels = np.asarray(calibration.labels.img)
        if labels.shape != shape:
            raise RuntimeError(
                f"CUDA labels/frame shape mismatch for {context.run}: "
                f"{labels.shape} != {shape}"
            )

        self.cp = cp
        self.context = context
        self.flash = flash
        self.phase_separation = _normalise_phase_separation(
            getattr(context, "phase_separation", "shared-signal")
        )
        self.arrival_scale = max(
            1e-6,
            float(os.environ.get("FFAC_RESIDUAL_GAS_ARRIVAL_SCALE", "0.02")),
        )
        self.label_data: dict[int, dict[str, Any]] = {}
        weighted_g = voxel_volume * density
        weighted_aq = voxel_volume * solubility
        gas_constraint = None
        aqueous_constraint = None
        expert_adapter = getattr(calibration, "expert_knowledge_adapter", None)
        if expert_adapter is not None:
            reference_frame = context._prepared_colors[0]
            gas_constraint = expert_adapter.mask_for(
                reference_frame,
                "saturation_g",
            )
            aqueous_constraint = expert_adapter.mask_for(
                reference_frame,
                "concentration_aq",
            )
        signal_models = calibration.signal_model.model[1]
        for label in context.signal_labels:
            mask = labels == int(label)
            if not np.any(mask):
                continue
            model = signal_models[int(label)]
            supports = np.asarray(model.supports, dtype=np.float64)
            self.label_data[int(label)] = {
                "model": model,
                "supports": cp.asarray(supports),
                "weight_g": cp.asarray(np.ascontiguousarray(weighted_g[mask])),
                "weight_aq": cp.asarray(np.ascontiguousarray(weighted_aq[mask])),
                "gas_constraint": (
                    cp.asarray(
                        np.ascontiguousarray(
                            np.asarray(gas_constraint, dtype=bool)[mask]
                        )
                    )
                    if gas_constraint is not None
                    else None
                ),
                "aqueous_constraint": (
                    cp.asarray(
                        np.ascontiguousarray(
                            np.asarray(aqueous_constraint, dtype=bool)[mask]
                        )
                    )
                    if aqueous_constraint is not None
                    else None
                ),
                "signals": [
                    cp.asarray(
                        np.ascontiguousarray(
                            np.asarray(frame.img, dtype=np.float64)[mask]
                        )
                    )
                    for frame in context._prepared_colors
                ],
            }

        self.gas_scores: list[dict[int, Any]] = []
        if self.phase_separation == "residual-gas":
            if len(context._gas_scores) != len(context._prepared_colors):
                raise RuntimeError(
                    f"CUDA residual score count mismatch for {context.run}: "
                    f"{len(context._gas_scores)} != {len(context._prepared_colors)}"
                )
            for score in context._gas_scores:
                score_array = np.asarray(score)
                self.gas_scores.append(
                    {
                        label: cp.asarray(
                            np.ascontiguousarray(score_array[labels == label])
                        )
                        for label in self.label_data
                    }
                )

        self.lut_y = (
            cp.asarray(np.asarray(flash._lut_y, dtype=np.float64))
            if hasattr(flash, "_lut_y")
            else None
        )
        self.lut_caq = (
            cp.asarray(np.asarray(flash._lut_caq, dtype=np.float64))
            if hasattr(flash, "_lut_caq")
            else None
        )
        self.frame_count = len(context._prepared_colors)
        cp.cuda.get_current_stream().synchronize()
        free_bytes, total_bytes = cp.cuda.Device().mem_info
        logger.info(
            "[%s] CUDA evaluator ready on %s; frames=%d labels=%s "
            "vram_used_mb=%.0f vram_free_mb=%.0f",
            context.run,
            cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            self.frame_count,
            sorted(self.label_data),
            (total_bytes - free_bytes) / (1024 * 1024),
            free_bytes / (1024 * 1024),
        )

    def evaluate(self, params: Dict[str, Any]) -> list[tuple[float, float, float]]:
        cp = self.cp
        values_by_label = {
            label: cp.asarray(
                np.asarray(data["model"].values, dtype=np.float64)
            )
            for label, data in self.label_data.items()
        }
        min_aq = float(self.flash.min_value_aq)
        max_aq = float(self.flash.max_value_aq)
        min_g = float(self.flash.min_value_g)
        max_g = float(self.flash.max_value_g)
        aq_denom = (max_aq - min_aq) or 1.0
        gas_denom = (max_g - min_g) or 1.0
        residual_onset = float(params.get("gas.residual_onset", 0.20))
        residual_width = max(
            1e-6,
            float(params.get("gas.residual_width", 0.40)),
        )

        totals: list[Any] = []
        gaseous: list[Any] = []
        aqueous: list[Any] = []
        for frame_index in range(self.frame_count):
            frame_g = cp.asarray(0.0, dtype=cp.float64)
            frame_aq = cp.asarray(0.0, dtype=cp.float64)
            for label, data in self.label_data.items():
                signal = cp.interp(
                    data["signals"][frame_index],
                    data["supports"],
                    values_by_label[label],
                )
                y_norm = cp.clip((signal - min_aq) / aq_denom, 0.0, 1.0)
                if self.lut_y is not None and self.lut_caq is not None:
                    concentration_aq = cp.interp(
                        y_norm,
                        self.lut_y,
                        self.lut_caq,
                    )
                else:
                    concentration_aq = y_norm
                if data["aqueous_constraint"] is not None:
                    concentration_aq *= data["aqueous_constraint"]

                if self.phase_separation == "residual-gas":
                    concentration_aq = concentration_aq.astype(cp.float32)
                    score = (
                        self.gas_scores[frame_index][label].astype(cp.float32)
                        * cp.float32(_RESIDUAL_GAS_SCORE_CLIP / 255.0)
                    )
                    saturation_g = cp.clip(
                        (score - cp.float32(residual_onset))
                        / cp.float32(residual_width),
                        0.0,
                        1.0,
                    )
                    saturation_g *= cp.clip(
                        concentration_aq / cp.float32(self.arrival_scale),
                        0.0,
                        1.0,
                    )
                else:
                    saturation_g = (
                        cp.clip(signal, min_g, max_g) - min_g
                    ) / gas_denom
                    if data["gas_constraint"] is not None:
                        saturation_g *= data["gas_constraint"]

                frame_g += cp.sum(data["weight_g"] * saturation_g)
                frame_aq += cp.sum(
                    data["weight_aq"]
                    * concentration_aq
                    * cp.clip(1.0 - saturation_g, 0.0, None)
                )
            gaseous.append(frame_g)
            aqueous.append(frame_aq)
            totals.append(frame_g + frame_aq)

        values = cp.asnumpy(cp.stack([*totals, *gaseous, *aqueous]))
        count = self.frame_count
        return [
            (
                float(values[index]),
                float(values[count + index]),
                float(values[2 * count + index]),
            )
            for index in range(count)
        ]

    def close(self) -> None:
        self.label_data.clear()
        self.gas_scores.clear()
        self.lut_y = None
        self.lut_caq = None
        self.cp.get_default_memory_pool().free_all_blocks()


def prepare_evaluation_context(
    context: CalibrationContext,
    *,
    backend: str = "prepared",
    release_images: bool = True,
) -> CalibrationContext:
    """Prepare parameter-independent frame data for repeated trial evaluation."""

    import gc
    import time

    requested = _normalise_evaluation_backend(backend)
    if requested == "legacy":
        context._evaluation_backend = "legacy"
        return context
    if not context._prepared_colors:
        started = time.perf_counter()
        prepared: list[Any] = []
        for image, _injected, _time_h in context._loaded:
            if image is None:
                raise RuntimeError(
                    f"[{context.run}] raw calibration images were released before preparation"
                )
            prepared.append(context.calibration.call_color_interpretation(image))

        context._prepared_colors = prepared
        context._evaluation_backend = "prepared"
        if release_images:
            context._loaded = [
                (None, injected, time_h)
                for _image, injected, time_h in context._loaded
            ]
            gc.collect()
        elapsed = time.perf_counter() - started
        logger.info(
            "[%s] prepared %d parameter-independent color frame(s) in %.2fs; "
            "released_raw=%s",
            context.run,
            len(prepared),
            elapsed,
            release_images,
        )

    if requested == "cuda" and context._cuda_evaluator is None:
        try:
            context._cuda_evaluator = _CudaIntegratedEvaluator(context)
        except Exception:
            try:
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            raise
        context._prepared_colors.clear()
        gc.collect()
    if requested == "opencl" and context._opencl_evaluator is None:
        from opencl_integrated_evaluator import OpenCLIntegratedEvaluator

        try:
            context._opencl_evaluator = OpenCLIntegratedEvaluator(
                context,
                residual_score_clip=_RESIDUAL_GAS_SCORE_CLIP,
            )
        except Exception:
            gc.collect()
            raise
        context._prepared_colors.clear()
        gc.collect()
    context._evaluation_backend = requested
    return context


def _mass_result_for_evaluation(
    context: CalibrationContext,
    image: Any,
    image_index: int,
    params: Dict[str, Any],
) -> Any:
    if image_index < len(getattr(context, "_prepared_colors", [])):
        color_interpretation = context._prepared_colors[image_index]
        pH = context.calibration.call_pH_analysis(color_interpretation)
        result = context.calibration.call_flash_and_mass_analysis(pH)
    else:
        result = context.calibration(image)
    mode = _normalise_phase_separation(
        getattr(context, "phase_separation", "shared-signal")
    )
    if mode != "residual-gas":
        return result
    if image_index >= len(context._gas_scores):
        raise IndexError(
            f"Missing residual-gas score {image_index} for {context.run}"
        )

    score = (
        np.asarray(context._gas_scores[image_index], dtype=np.float32)
        * (_RESIDUAL_GAS_SCORE_CLIP / 255.0)
    )
    onset = float(params.get("gas.residual_onset", 0.20))
    width = max(1e-6, float(params.get("gas.residual_width", 0.40)))
    saturation_g = np.clip((score - onset) / width, 0.0, 1.0)

    # Gas can only occur where the indicator records CO2 arrival. The soft gate
    # avoids classifying fixed optical defects as gas while retaining interfaces.
    concentration_aq = np.asarray(result.concentration_aq.img, dtype=np.float32)
    arrival_scale = max(
        1e-6,
        float(os.environ.get("FFAC_RESIDUAL_GAS_ARRIVAL_SCALE", "0.02")),
    )
    saturation_g *= np.clip(concentration_aq / arrival_scale, 0.0, 1.0)

    mass_analysis = context.calibration.co2_mass_analysis
    density = np.asarray(mass_analysis.density_gaseous_co2)
    solubility = np.asarray(mass_analysis.solubility_co2)
    mass_g = density * saturation_g
    mass_aq = solubility * concentration_aq * np.clip(
        1.0 - saturation_g,
        0.0,
        1.0,
    )
    result.saturation_g.img = saturation_g
    result.mass_g.img = mass_g
    result.mass_aq.img = mass_aq
    result.mass.img = mass_g + mass_aq
    return result


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
    integrated_masses: Optional[List[Tuple[float, float, float]]] = None
    integrated_evaluator = getattr(context, "_cuda_evaluator", None)
    if integrated_evaluator is None:
        integrated_evaluator = getattr(context, "_opencl_evaluator", None)
    if integrated_evaluator is not None:
        try:
            integrated_masses = integrated_evaluator.evaluate(params)
        except Exception as exc:  # noqa: BLE001
            backend = getattr(context, "_evaluation_backend", "integrated")
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics={},
                status=f"eval-error:{backend}:{exc}",
                params=params,
            )
    for i, (img, injected, t_h) in enumerate(context._loaded):
        try:
            if integrated_masses is not None:
                detected, detected_g, detected_aq = integrated_masses[i]
            else:
                mass_result = _mass_result_for_evaluation(context, img, i, params)
                detected = float(context.geometry.integrate(mass_result.mass))
                detected_g = (
                    float(context.geometry.integrate(mass_result.mass_g))
                    if getattr(mass_result, "mass_g", None) is not None
                    else None
                )
                detected_aq = (
                    float(context.geometry.integrate(mass_result.mass_aq))
                    if getattr(mass_result, "mass_aq", None) is not None
                    else None
                )
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
        metrics[f"{t_h:.3f}h"] = Metrics(
            injected_full=injected,
            total_full=detected,
            gaseous_full=detected_g,
            aqueous_full=detected_aq,
        )
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
    if mode in {"window-balanced", "windows", "balanced-windows"}:
        windows: dict[str, list[float]] = {
            "I1": [],
            "I2": [],
            "early": [],
            "late": [],
        }
        for time_h, injected, detected in samples:
            if time_h <= 0.92:
                key = "I1"
            elif time_h < 4.0:
                key = "I2"
            # Requested 4.1/8 h frames can resolve a few minutes to either side.
            # These tolerant cutoffs preserve their intended operational window.
            elif time_h <= 8.25:
                key = "early"
            else:
                key = "late"
            windows[key].append(abs(detected - injected))
        if any(not values for values in windows.values()):
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="missing-objective-window",
                params=params,
            )
        # Each operational window contributes one mean absolute error, regardless
        # of how many calibration frames happen to lie inside that window.
        total_err = sum(float(np.mean(values)) for values in windows.values())
    elif mode.startswith("drift"):
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
    if best_params is None:
        raise ValueError(f"[{context.run}] no best params to save")
    if context.calibration_folder is None:
        raise ValueError(f"[{context.run}] calibration folder is not configured")

    apply_params(context.calibration, best_params, labels=context.signal_labels, np_module=np)
    folder = Path(context.calibration_folder) / "signal_model"
    folder.mkdir(parents=True, exist_ok=True)
    hetero = context.calibration.signal_model.model[1]
    failures: List[str] = []
    for label in (hetero.keys() if hasattr(hetero, "keys") else []):
        try:
            hetero[label].save(folder / f"signal_model_{label}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"signal_model_{label}: {exc}")
    logger.info("[%s] saved optimised signal models to %s", context.run, folder)

    flash = getattr(context.calibration, "flash", None)
    if flash is None or not hasattr(flash, "save"):
        failures.append("flash: calibration has no saveable flash model")
    else:
        try:
            flash_path = Path(context.calibration_folder) / "flash" / "flash"
            flash.save(flash_path)
            logger.info(
                "[%s] saved optimised %s to %s",
                context.run,
                type(flash).__name__,
                flash_path.with_suffix(".json"),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"flash: {exc}")

    if failures:
        raise RuntimeError(
            f"[{context.run}] calibration save failed: " + "; ".join(failures)
        )

    if out_folder:
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        (Path(out_folder) / "best_params.json").write_text(
            json.dumps(best_params, indent=2, default=str), encoding="utf-8"
        )


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
            mass = _mass_result_for_evaluation(context, img, i, {}).mass
            arr = np.asarray(getattr(mass, "img", mass), dtype=float)
            nan_frac = float(np.mean(~np.isfinite(arr))) if arr.size else -1.0
            det = float(context.geometry.integrate(mass))
        except Exception as exc:  # noqa: BLE001
            logger.info("[diag %s] img%d ERROR %s", context.run, i, exc); continue
        source = (
            context._prepared_colors[i]
            if i < len(getattr(context, "_prepared_colors", []))
            else img
        )
        date = getattr(source, "date", None)
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
