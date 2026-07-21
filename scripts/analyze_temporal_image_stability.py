r"""Screen within-run camera geometry and lighting stability.

The calibration pipeline estimates template registration once from the run
baseline and reuses that transform for every image. This script checks the
assumption behind that choice by following raw images throughout each run.

It reports

* partial-affine registration of every sampled frame back to the first frame,
* inter-sample camera jumps,
* raw RGB/luminance changes in global and border regions (a screen that is
  intentionally treated as image-content, not pure illumination),
* optional post-correction ColorChecker residuals, and
* optional residual light drift over automatically selected static pixels, and
* whether an observation is one of the mass-calibration frames.

Examples (run from the repository root):

    .\.venv\Scripts\python.exe scripts\analyze_temporal_image_stability.py `
        --runs ac20 ac24 ac40 ac44 ac52 ac60 --step 10 `
        --out Z:\Albus\QA\temporal_stability

    # Slower pass through DarSIA, including per-frame ColorChecker diagnostics.
    .\.venv\Scripts\python.exe scripts\analyze_temporal_image_stability.py `
        --runs ac20 ac24 ac40 ac44 ac52 ac60 --step 20 --corrected `
        --out Z:\Albus\QA\temporal_stability_corrected

The script is diagnostic only and does not modify calibration or rig files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

import cv2
import matplotlib
import numpy as np
from PIL import ExifTags, Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
CALIBRATION_COLOR = "#15803d"
GEOMETRY_COLOR = "#1d4ed8"
JUMP_COLOR = "#dc2626"
LIGHT_COLOR = "#b45309"


@dataclass(frozen=True)
class Frame:
    date: datetime
    name: str
    path: Path
    is_calibration: bool = False


@dataclass
class Features:
    keypoints: Sequence[Any]
    descriptors: np.ndarray | None


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def _parse_duration_seconds(value: str) -> float:
    parts = [float(part) for part in value.strip().split(":")]
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS duration, got {value!r}")
    return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]


def _imaging_pairs(cfg: dict[str, Any], repo_root: Path) -> list[tuple[Path, Path]]:
    protocols = cfg.get("protocols", {})
    imaging = protocols.get("imaging", {})
    folders = cfg.get("data", {}).get("folders", [])
    pairs: list[tuple[Path, Path]] = []
    if isinstance(imaging, dict):
        for folder in folders:
            csv_rel = imaging.get(folder)
            if csv_rel:
                pairs.append((Path(folder), repo_root / csv_rel))
    elif isinstance(imaging, str) and folders:
        pairs.append((Path(folders[0]), repo_root / imaging))
    return pairs


def _ordered_frames(cfg: dict[str, Any], repo_root: Path) -> list[Frame]:
    frames: list[Frame] = []
    for folder, csv_path in _imaging_pairs(cfg, repo_root):
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                name = row.get("path") or row.get("Path")
                date_text = row.get("datetime") or row.get("Datetime")
                if not name or not date_text:
                    continue
                frames.append(
                    Frame(
                        date=_parse_datetime(date_text),
                        name=name,
                        path=folder / name,
                    )
                )
    frames.sort(key=lambda frame: (frame.date, frame.name))
    return frames


def _experiment_start(cfg: dict[str, Any], repo_root: Path, frames: Sequence[Frame]) -> datetime:
    injection = cfg.get("protocols", {}).get("injection")
    if injection:
        path = repo_root / injection
        if path.is_file():
            starts: list[datetime] = []
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("start"):
                        starts.append(_parse_datetime(row["start"]))
            if starts:
                return min(starts)
    if not frames:
        raise ValueError("Cannot infer experiment start without frames")
    return frames[0].date


def _calibration_targets(cfg: dict[str, Any]) -> list[tuple[float, float]]:
    names = cfg.get("calibration", {}).get("mass", {}).get("data", [])
    intervals = cfg.get("data", {}).get("interval", {})
    targets: list[tuple[float, float]] = []
    for name in names:
        spec = intervals.get(name, {})
        if not spec:
            continue
        start = _parse_duration_seconds(str(spec["start"]))
        end = _parse_duration_seconds(str(spec["end"]))
        count = int(spec.get("num", 1))
        tolerance = _parse_duration_seconds(str(spec.get("tol", "00:00:00")))
        for seconds in np.linspace(start, end, max(1, count)):
            targets.append((float(seconds), tolerance))
    return targets


def _mark_calibration_frames(
    frames: Sequence[Frame],
    experiment_start: datetime,
    targets: Sequence[tuple[float, float]],
) -> tuple[list[Frame], set[int]]:
    elapsed = np.asarray(
        [(frame.date - experiment_start).total_seconds() for frame in frames],
        dtype=float,
    )
    selected: set[int] = set()
    for target, tolerance in targets:
        if elapsed.size == 0:
            break
        index = int(np.argmin(np.abs(elapsed - target)))
        if abs(float(elapsed[index]) - target) <= tolerance:
            selected.add(index)
    marked = [
        Frame(frame.date, frame.name, frame.path, index in selected)
        for index, frame in enumerate(frames)
    ]
    return marked, selected


def _sample_indices(
    total: int,
    *,
    step: int,
    max_frames: int,
    required: Iterable[int],
) -> list[int]:
    if total <= 0:
        return []
    regular = list(range(0, total, max(1, step)))
    if regular[-1] != total - 1:
        regular.append(total - 1)
    if max_frames > 0 and len(regular) > max_frames:
        positions = np.linspace(0, len(regular) - 1, max_frames, dtype=int)
        regular = [regular[int(position)] for position in positions]
    return sorted(set(regular).union(int(index) for index in required))


def _load_rgb(path: Path, max_dim: int) -> tuple[np.ndarray, float]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resize_scale = min(1.0, float(max_dim) / max(width, height))
        if resize_scale < 1.0:
            rgb = rgb.resize(
                (
                    max(1, int(round(width * resize_scale))),
                    max(1, int(round(height * resize_scale))),
                ),
                Image.Resampling.BILINEAR,
            )
        return np.asarray(rgb, dtype=np.uint8), resize_scale


def _rgb_from_corrected(image: Any, max_dim: int) -> np.ndarray:
    arr = np.asarray(image.img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = np.asarray(arr[..., :3], dtype=np.float32)
    if float(np.nanmax(arr)) <= 1.5:
        arr = arr * 255.0
    arr8 = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    height, width = arr8.shape[:2]
    scale = min(1.0, float(max_dim) / max(width, height))
    if scale < 1.0:
        arr8 = cv2.resize(arr8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return arr8


def _region_slices(shape: tuple[int, int], fraction: float) -> dict[str, tuple[slice, slice]]:
    height, width = shape
    band_y = max(1, int(round(height * fraction)))
    band_x = max(1, int(round(width * fraction)))
    return {
        "full": (slice(0, height), slice(0, width)),
        "top": (slice(0, band_y), slice(0, width)),
        "bottom": (slice(height - band_y, height), slice(0, width)),
        "left": (slice(0, height), slice(0, band_x)),
        "right": (slice(0, height), slice(width - band_x, width)),
    }


def _light_metrics(rgb: np.ndarray, fraction: float) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, region in _region_slices(rgb.shape[:2], fraction).items():
        pixels = np.asarray(rgb[region], dtype=np.float32).reshape(-1, 3) / 255.0
        median = np.nanmedian(pixels, axis=0)
        metrics[f"{name}_r"] = float(median[0])
        metrics[f"{name}_g"] = float(median[1])
        metrics[f"{name}_b"] = float(median[2])
        metrics[f"{name}_luma"] = float(
            0.2126 * median[0] + 0.7152 * median[1] + 0.0722 * median[2]
        )
    return metrics


def _safe_float(value: Any) -> float:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def _exif_metrics(path: Path) -> dict[str, float]:
    wanted = {
        "ExposureTime": "exposure_s",
        "FNumber": "f_number",
        "ISOSpeedRatings": "iso",
        "PhotographicSensitivity": "iso",
        "ExposureBiasValue": "exposure_bias_ev",
        "FocalLength": "focal_length_mm",
        "WhiteBalance": "white_balance",
    }
    result = {name: float("nan") for name in set(wanted.values())}
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                output = wanted.get(tag)
                if output:
                    result[output] = _safe_float(value)
    except Exception:
        pass
    return result


def _registration_gray(rgb: np.ndarray, max_col_fraction: float) -> np.ndarray:
    width = max(1, int(round(rgb.shape[1] * max_col_fraction)))
    gray = cv2.cvtColor(rgb[:, :width, :3], cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(gray)


def _features(gray: np.ndarray, max_features: int) -> Features:
    detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=8)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return Features(keypoints=keypoints or [], descriptors=descriptors)


def _empty_registration() -> dict[str, float]:
    return {
        "dx_px": float("nan"),
        "dy_px": float("nan"),
        "displacement_px": float("nan"),
        "scale": float("nan"),
        "angle_deg": float("nan"),
        "matches": 0.0,
        "inliers": 0.0,
        "inlier_fraction": float("nan"),
        "reprojection_median_px": float("nan"),
    }


def _estimate_registration(
    source: Features,
    destination: Features,
    *,
    resize_scale: float,
    keep_matches: int,
    ransac_threshold_small_px: float,
) -> dict[str, float]:
    if source.descriptors is None or destination.descriptors is None:
        return _empty_registration()
    if len(source.keypoints) < 10 or len(destination.keypoints) < 10:
        return _empty_registration()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(
        matcher.match(source.descriptors, destination.descriptors),
        key=lambda match: match.distance,
    )[: max(10, keep_matches)]
    if len(matches) < 10:
        return _empty_registration()
    src = np.float32([source.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([destination.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_small_px),
        maxIters=5000,
        confidence=0.995,
    )
    if matrix is None:
        return _empty_registration()
    inlier_mask = (
        np.asarray(inliers, dtype=bool).reshape(-1)
        if inliers is not None
        else np.ones(len(matches), dtype=bool)
    )
    a, _b, tx = (float(value) for value in matrix[0])
    c, _d, ty = (float(value) for value in matrix[1])
    scale = math.sqrt(a * a + c * c)
    angle = math.degrees(math.atan2(c, a))
    projected = cv2.transform(src, matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)
    inlier_errors = errors[inlier_mask] if np.any(inlier_mask) else errors
    full_factor = 1.0 / max(resize_scale, 1e-9)
    dx = tx * full_factor
    dy = ty * full_factor
    return {
        "dx_px": dx,
        "dy_px": dy,
        "displacement_px": math.hypot(dx, dy),
        "scale": scale,
        "angle_deg": angle,
        "matches": float(len(matches)),
        "inliers": float(np.count_nonzero(inlier_mask)),
        "inlier_fraction": float(np.mean(inlier_mask)),
        "reprojection_median_px": float(np.median(inlier_errors) * full_factor),
    }


def _identity_registration() -> dict[str, float]:
    return {
        "dx_px": 0.0,
        "dy_px": 0.0,
        "displacement_px": 0.0,
        "scale": 1.0,
        "angle_deg": 0.0,
        "matches": float("nan"),
        "inliers": float("nan"),
        "inlier_fraction": 1.0,
        "reprojection_median_px": 0.0,
    }


def _normalise_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _build_corrected_reader(
    run: str,
    *,
    config_dir: Path,
    common: Path,
) -> tuple[Any, set[str]]:
    from darsia.presets.workflows.analysis.analysis_context import prepare_analysis_context
    from darsia.presets.workflows.rig import Rig

    run_config = config_dir / f"{run}.toml"
    paths = [common, run_config]
    stamp = config_dir / ".color_state" / f"{run}.txt"
    color_on = stamp.is_file() and stamp.read_text(encoding="utf-8").strip().lower() == "on"
    overlay = config_dir.parent / "coloron.toml"
    if color_on and overlay.is_file():
        paths.append(overlay)
    context = prepare_analysis_context(
        cls=Rig,
        path=paths,
        all=False,
        require_color_to_mass=False,
    )
    calibration_paths = {
        _normalise_path(Path(path)) for path in (getattr(context, "image_paths", []) or [])
    }
    return context.fluidflower, calibration_paths


def _read_shape_corrected(fluidflower: Any, path: Path) -> Any:
    """Read one image through the rig geometry, stopping before color correction."""

    import darsia

    date = fluidflower.experiment.get_datetime(path)
    return darsia.imread(
        path,
        transformations=getattr(fluidflower, "shape_corrections", []) or [],
        date=date,
        reference_date=fluidflower.reference_date,
        name=path.name,
    )


def _pre_colorchecker_metrics(fluidflower: Any, shape_corrected: Any) -> dict[str, float]:
    """Measure checker demand before correction, using the correction's own ROI."""

    import darsia

    residuals: list[float] = []
    neutral_gains: list[np.ndarray] = []
    for correction in getattr(fluidflower, "color_corrections", []) or []:
        if not getattr(correction, "active", False):
            continue
        try:
            roi = correction._restrict_to_roi(np.asarray(shape_corrected.img))
            measured = np.asarray(darsia.CustomColorChecker(image=roi).swatches_rgb, dtype=float)
            reference = np.asarray(correction.colorchecker.swatches_rgb, dtype=float)
            residuals.append(float(np.mean(np.abs(measured - reference))))
            measured_neutral = np.median(measured[-1].reshape(-1, 3), axis=0)
            reference_neutral = np.median(reference[-1].reshape(-1, 3), axis=0)
            neutral_gains.append(reference_neutral / np.maximum(measured_neutral, 1e-9))
        except Exception:
            continue
    if not residuals:
        return {
            "colorchecker_pre_residual": float("nan"),
            "colorchecker_neutral_gain_r": float("nan"),
            "colorchecker_neutral_gain_g": float("nan"),
            "colorchecker_neutral_gain_b": float("nan"),
        }
    gain = np.median(np.stack(neutral_gains), axis=0)
    return {
        "colorchecker_pre_residual": max(residuals),
        "colorchecker_neutral_gain_r": float(gain[0]),
        "colorchecker_neutral_gain_g": float(gain[1]),
        "colorchecker_neutral_gain_b": float(gain[2]),
    }


def _apply_color_corrections(fluidflower: Any, shape_corrected: Any) -> Any:
    corrected = shape_corrected.copy()
    for correction in getattr(fluidflower, "color_corrections", []) or []:
        correction(corrected, overwrite=True)
    return corrected


def _colorchecker_metrics(fluidflower: Any) -> dict[str, Any]:
    residuals: list[float] = []
    flagged = False
    thresholds: list[float] = []
    active = 0
    for correction in getattr(fluidflower, "color_corrections", []) or []:
        if not getattr(correction, "active", False):
            continue
        active += 1
        residual = _safe_float(getattr(correction, "last_residual", float("nan")))
        if math.isfinite(residual):
            residuals.append(residual)
        thresholds.append(
            _safe_float(getattr(correction, "residual_warn_threshold", float("nan")))
        )
        flagged = flagged or bool(getattr(correction, "last_flagged", False))
    return {
        "colorchecker_active": active,
        "colorchecker_residual": max(residuals) if residuals else float("nan"),
        "colorchecker_threshold": max(
            (value for value in thresholds if math.isfinite(value)),
            default=float("nan"),
        ),
        "colorchecker_flagged": int(flagged),
    }


def _stable_pixel_mask(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Find low-variance, blue/static pixels after normal ColorChecker correction."""

    if not frames:
        return np.zeros((0, 0), dtype=bool)
    stack = np.stack(frames).astype(np.float32) / 255.0
    mean_rgb = np.mean(stack, axis=0)
    std_rgb = np.std(stack, axis=0)
    luma = 0.2126 * stack[..., 0] + 0.7152 * stack[..., 1] + 0.0722 * stack[..., 2]
    mean_luma = np.mean(luma, axis=0)
    std_luma = np.std(luma, axis=0)
    std_chroma = np.linalg.norm(std_rgb, axis=2)
    finite = np.all(np.isfinite(mean_rgb), axis=2)
    values = mean_luma[finite]
    if values.size == 0:
        return np.zeros(mean_luma.shape, dtype=bool)
    lo, hi = np.percentile(values, [5.0, 95.0])
    luma_limit = np.percentile(std_luma[finite], 35.0)
    chroma_limit = np.percentile(std_chroma[finite], 40.0)
    base = (
        finite
        & (mean_luma >= lo)
        & (mean_luma <= hi)
        & (std_luma <= luma_limit)
        & (std_chroma <= chroma_limit)
    )
    blueish = (mean_rgb[..., 2] > mean_rgb[..., 0] + 0.005) & (
        mean_rgb[..., 2] > mean_rgb[..., 1] + 0.002
    )
    mask = base & blueish
    minimum = max(500, int(0.002 * mask.size))
    return mask if int(np.count_nonzero(mask)) >= minimum else base


def _fit_spatial_gain(
    reference: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
    *,
    grid_rows: int = 6,
    grid_cols: int = 10,
) -> tuple[float, float, float, float]:
    """Fit a gain plane over static pixels; gradients are edge-to-edge percentages."""

    ref = np.asarray(reference, dtype=np.float32) / 255.0
    cur = np.asarray(current, dtype=np.float32) / 255.0
    ref_luma = 0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
    cur_luma = 0.2126 * cur[..., 0] + 0.7152 * cur[..., 1] + 0.0722 * cur[..., 2]
    height, width = mask.shape
    design: list[list[float]] = []
    gains: list[float] = []
    for row in range(grid_rows):
        y0 = int(round(row * height / grid_rows))
        y1 = int(round((row + 1) * height / grid_rows))
        for col in range(grid_cols):
            x0 = int(round(col * width / grid_cols))
            x1 = int(round((col + 1) * width / grid_cols))
            tile_mask = mask[y0:y1, x0:x1]
            if int(np.count_nonzero(tile_mask)) < 30:
                continue
            ref_value = float(np.median(ref_luma[y0:y1, x0:x1][tile_mask]))
            cur_value = float(np.median(cur_luma[y0:y1, x0:x1][tile_mask]))
            if ref_value <= 1e-6 or cur_value <= 1e-6:
                continue
            x = (col + 0.5) / grid_cols - 0.5
            y = (row + 0.5) / grid_rows - 0.5
            design.append([1.0, x, y])
            gains.append(ref_value / cur_value)
    if len(gains) < 6:
        return (float("nan"),) * 4
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(gains, dtype=float)
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    residual = values - matrix @ coefficients
    return (
        100.0 * (float(coefficients[0]) - 1.0),
        100.0 * float(coefficients[1]),
        100.0 * float(coefficients[2]),
        100.0 * float(np.sqrt(np.mean(residual**2))),
    )


def _add_stable_light_metrics(
    rows: list[dict[str, Any]],
    corrected_frames: Sequence[np.ndarray],
) -> np.ndarray | None:
    if not rows or len(rows) != len(corrected_frames):
        return None
    mask = _stable_pixel_mask(corrected_frames)
    sample_count = int(np.count_nonzero(mask))
    if sample_count == 0:
        return None
    reference = np.asarray(corrected_frames[0])
    ref_pixels = reference[mask].astype(np.float32) / 255.0
    ref_rgb = np.median(ref_pixels, axis=0)
    ref_luma = float(0.2126 * ref_rgb[0] + 0.7152 * ref_rgb[1] + 0.0722 * ref_rgb[2])
    for row, frame in zip(rows, corrected_frames):
        pixels = np.asarray(frame)[mask].astype(np.float32) / 255.0
        rgb = np.median(pixels, axis=0)
        luma = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        global_gain, gradient_x, gradient_y, residual = _fit_spatial_gain(
            reference, frame, mask
        )
        row.update(
            {
                "stable_pixel_count": sample_count,
                "corrected_stable_r": float(rgb[0]),
                "corrected_stable_g": float(rgb[1]),
                "corrected_stable_b": float(rgb[2]),
                "corrected_stable_luma_change_pct": 100.0
                * (luma / max(ref_luma, 1e-9) - 1.0),
                "corrected_stable_rgb_distance": float(np.linalg.norm(rgb - ref_rgb)),
                "corrected_spatial_gain_change_pct": global_gain,
                "corrected_spatial_gradient_x_pct": gradient_x,
                "corrected_spatial_gradient_y_pct": gradient_y,
                "corrected_spatial_gain_residual_pct": residual,
            }
        )
    return mask


def _save_stable_mask(
    path: Path,
    reference: np.ndarray,
    mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(reference, dtype=np.uint8).copy()
    overlay = rgb.copy()
    overlay[mask] = np.array([22, 163, 74], dtype=np.uint8)
    combined = np.concatenate(
        [
            rgb,
            np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2),
            cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0.0),
        ],
        axis=1,
    )
    Image.fromarray(combined).save(path)


def _add_relative_light_metrics(rows: list[dict[str, Any]], prefix: str) -> None:
    if not rows or f"{prefix}_full_luma" not in rows[0]:
        return
    reference = rows[0]
    eps = 1e-9
    for row in rows:
        full_ratio = row[f"{prefix}_full_luma"] / max(reference[f"{prefix}_full_luma"], eps)
        top_ratio = row[f"{prefix}_top_luma"] / max(reference[f"{prefix}_top_luma"], eps)
        bottom_ratio = row[f"{prefix}_bottom_luma"] / max(
            reference[f"{prefix}_bottom_luma"], eps
        )
        left_ratio = row[f"{prefix}_left_luma"] / max(reference[f"{prefix}_left_luma"], eps)
        right_ratio = row[f"{prefix}_right_luma"] / max(reference[f"{prefix}_right_luma"], eps)
        row[f"{prefix}_luma_change_pct"] = 100.0 * (full_ratio - 1.0)
        row[f"{prefix}_vertical_gradient_pct"] = 100.0 * (top_ratio / max(bottom_ratio, eps) - 1.0)
        row[f"{prefix}_horizontal_gradient_pct"] = 100.0 * (left_ratio / max(right_ratio, eps) - 1.0)
        rgb_now = np.asarray(
            [row[f"{prefix}_full_{channel}"] for channel in "rgb"], dtype=float
        )
        rgb_ref = np.asarray(
            [reference[f"{prefix}_full_{channel}"] for channel in "rgb"], dtype=float
        )
        row[f"{prefix}_rgb_distance"] = float(np.linalg.norm(rgb_now - rgb_ref))


def _finite_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values = [_safe_float(row.get(key)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def _max_abs(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    return max((abs(value) for value in values), default=float("nan"))


def _summarise_run(run: str, rows: list[dict[str, Any]], total_frames: int) -> dict[str, Any]:
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    summary: dict[str, Any] = {
        "run": run,
        "total_frames": total_frames,
        "analysed_frames": len(rows),
        "calibration_frames_analysed": sum(int(row["is_calibration"]) for row in rows),
        "start_datetime": first.get("datetime", ""),
        "end_datetime": last.get("datetime", ""),
        "end_elapsed_h": last.get("elapsed_h", float("nan")),
        "start_file": first.get("file", ""),
        "end_file": last.get("file", ""),
        "end_dx_px": last.get("ref_dx_px", float("nan")),
        "end_dy_px": last.get("ref_dy_px", float("nan")),
        "end_displacement_px": last.get("ref_displacement_px", float("nan")),
        "end_scale_change_pct": 100.0
        * (_safe_float(last.get("ref_scale")) - 1.0),
        "end_angle_deg": last.get("ref_angle_deg", float("nan")),
        "end_inlier_fraction": last.get("ref_inlier_fraction", float("nan")),
        "end_reprojection_median_px": last.get(
            "ref_reprojection_median_px", float("nan")
        ),
        "registration_failures": sum(
            not math.isfinite(_safe_float(row.get("ref_displacement_px"))) for row in rows
        ),
        "max_ref_displacement_px": max(
            _finite_values(rows, "ref_displacement_px"), default=float("nan")
        ),
        "max_step_jump_px": max(_finite_values(rows, "step_displacement_px"), default=float("nan")),
        "max_abs_scale_change_pct": 100.0
        * max(
            (abs(value - 1.0) for value in _finite_values(rows, "ref_scale")),
            default=float("nan"),
        ),
        "max_abs_angle_deg": _max_abs(rows, "ref_angle_deg"),
        "max_abs_raw_luma_change_pct": _max_abs(rows, "raw_luma_change_pct"),
        "max_abs_raw_horizontal_gradient_pct": _max_abs(
            rows, "raw_horizontal_gradient_pct"
        ),
        "max_abs_raw_vertical_gradient_pct": _max_abs(rows, "raw_vertical_gradient_pct"),
    }
    if rows and "corrected_luma_change_pct" in rows[0]:
        summary.update(
            {
                "max_abs_corrected_luma_change_pct": _max_abs(
                    rows, "corrected_luma_change_pct"
                ),
                "max_abs_corrected_horizontal_gradient_pct": _max_abs(
                    rows, "corrected_horizontal_gradient_pct"
                ),
                "max_abs_corrected_vertical_gradient_pct": _max_abs(
                    rows, "corrected_vertical_gradient_pct"
                ),
                "max_colorchecker_residual": max(
                    _finite_values(rows, "colorchecker_residual"), default=float("nan")
                ),
                "colorchecker_flagged_frames": sum(
                    int(row.get("colorchecker_flagged", 0)) for row in rows
                ),
                "max_colorchecker_pre_residual": max(
                    _finite_values(rows, "colorchecker_pre_residual"),
                    default=float("nan"),
                ),
            }
        )
    if rows and "corrected_stable_luma_change_pct" in rows[0]:
        summary.update(
            {
                "stable_pixel_count": int(rows[0].get("stable_pixel_count", 0)),
                "max_abs_corrected_stable_luma_change_pct": _max_abs(
                    rows, "corrected_stable_luma_change_pct"
                ),
                "max_abs_corrected_spatial_gradient_x_pct": _max_abs(
                    rows, "corrected_spatial_gradient_x_pct"
                ),
                "max_abs_corrected_spatial_gradient_y_pct": _max_abs(
                    rows, "corrected_spatial_gradient_y_pct"
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_run(run: str, rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    times = np.asarray([float(row["elapsed_h"]) for row in rows])
    calibration = np.asarray([bool(row["is_calibration"]) for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(times, [row["ref_displacement_px"] for row in rows], color=GEOMETRY_COLOR, label="to first frame")
    axes[0].plot(times, [row["step_displacement_px"] for row in rows], color=JUMP_COLOR, alpha=0.75, label="inter-sample jump")
    axes[0].set_ylabel("translation (px)")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(times, [100.0 * (row["ref_scale"] - 1.0) for row in rows], color="#6d28d9", label="scale")
    axes[1].plot(times, [row["ref_angle_deg"] for row in rows], color="#0f766e", label="rotation")
    axes[1].set_ylabel("scale (%) / angle (deg)")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].plot(times, [row["raw_luma_change_pct"] for row in rows], color=LIGHT_COLOR, alpha=0.55, label="raw image-content luma")
    axes[2].plot(times, [row["raw_horizontal_gradient_pct"] for row in rows], color="#7c3aed", alpha=0.75, label="left/right gradient")
    axes[2].plot(times, [row["raw_vertical_gradient_pct"] for row in rows], color="#0284c7", alpha=0.75, label="top/bottom gradient")
    if "corrected_luma_change_pct" in rows[0]:
        axes[2].plot(times, [row["corrected_luma_change_pct"] for row in rows], color="#166534", alpha=0.55, label="post-ColorChecker full image")
    if "corrected_stable_luma_change_pct" in rows[0]:
        axes[2].plot(times, [row["corrected_stable_luma_change_pct"] for row in rows], color="#166534", lw=1.8, label="post-ColorChecker static pixels")
    axes[2].set_ylabel("change (%)")
    axes[2].legend(loc="upper left", fontsize=8, ncol=2)

    if "colorchecker_residual" in rows[0]:
        axes[3].plot(times, [row["colorchecker_residual"] for row in rows], color="#be123c", label="ColorChecker residual")
        axes[3].plot(times, [row["colorchecker_pre_residual"] for row in rows], color="#7c2d12", alpha=0.65, label="pre-correction demand")
        thresholds = _finite_values(rows, "colorchecker_threshold")
        if thresholds:
            axes[3].axhline(float(np.median(thresholds)), color="#64748b", ls="--", lw=1, label="flag threshold")
        axes[3].set_ylabel("swatch residual")
        axes[3].legend(loc="upper left", fontsize=8)
    else:
        axes[3].plot(times, [row["ref_inlier_fraction"] for row in rows], color="#334155")
        axes[3].set_ylabel("registration\ninlier fraction")

    if np.any(calibration):
        for axis in axes:
            axis.scatter(
                times[calibration],
                np.interp(times[calibration], times, np.asarray(axis.lines[0].get_ydata(), dtype=float)),
                marker="|",
                s=90,
                color=CALIBRATION_COLOR,
                zorder=5,
            )
    axes[-1].set_xlabel("hours from injection start")
    axes[0].set_title(f"{run.upper()}: within-run geometry and light stability")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_run(run: str, args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config_dir / f"{run}.toml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = _deep_merge(_read_toml(args.common), _read_toml(config_path))
    frames = _ordered_frames(cfg, args.repo_root)
    if not frames:
        raise RuntimeError(f"{run}: no frames resolved from imaging protocols")
    experiment_start = _experiment_start(cfg, args.repo_root, frames)
    frames, calibration_indices = _mark_calibration_frames(
        frames,
        experiment_start,
        _calibration_targets(cfg),
    )
    indices = _sample_indices(
        len(frames),
        step=args.step,
        max_frames=args.max_frames,
        required=calibration_indices,
    )
    if args.start_end_only:
        indices = [0] if len(frames) == 1 else [0, len(frames) - 1]
    if args.checkpoint_hours:
        elapsed_hours = np.asarray(
            [(frame.date - experiment_start).total_seconds() / 3600.0 for frame in frames]
        )
        checkpoint_indices = {0, len(frames) - 1}
        for hour in args.checkpoint_hours:
            if elapsed_hours[0] <= hour <= elapsed_hours[-1]:
                checkpoint_indices.add(int(np.argmin(np.abs(elapsed_hours - hour))))
        indices = sorted(checkpoint_indices)

    fluidflower = None
    exact_calibration_paths: set[str] = set()
    if args.corrected:
        fluidflower, exact_calibration_paths = _build_corrected_reader(
            run,
            config_dir=args.config_dir,
            common=args.common,
        )

    rows: list[dict[str, Any]] = []
    corrected_frames: list[np.ndarray] = []
    reference_features: Features | None = None
    previous_features: Features | None = None
    previous_scale = 1.0
    reference_width = None
    max_col_fraction = 1.0

    for position, index in enumerate(indices, start=1):
        frame = frames[index]
        if not frame.path.is_file():
            print(f"  {run}: missing {frame.path}", flush=True)
            continue
        raw_rgb, resize_scale = _load_rgb(frame.path, args.max_dim)
        if reference_width is None:
            reference_width = raw_rgb.shape[1]
            max_col_small = min(
                raw_rgb.shape[1],
                int(round(args.registration_max_col * resize_scale)),
            )
            max_col_fraction = max_col_small / max(1, raw_rgb.shape[1])
        gray = _registration_gray(raw_rgb, max_col_fraction)
        current_features = _features(gray, args.max_features)
        if reference_features is None:
            reference_features = current_features
            direct = _identity_registration()
            step_registration = _identity_registration()
        else:
            direct = _estimate_registration(
                current_features,
                reference_features,
                resize_scale=resize_scale,
                keep_matches=args.keep_matches,
                ransac_threshold_small_px=args.ransac_threshold_small_px,
            )
            assert previous_features is not None
            step_registration = _estimate_registration(
                current_features,
                previous_features,
                resize_scale=min(resize_scale, previous_scale),
                keep_matches=args.keep_matches,
                ransac_threshold_small_px=args.ransac_threshold_small_px,
            )

        raw_light = _light_metrics(raw_rgb, args.border_fraction)
        row: dict[str, Any] = {
            "run": run,
            "datetime": frame.date.isoformat(sep=" "),
            "elapsed_h": (frame.date - experiment_start).total_seconds() / 3600.0,
            "frame_index": index,
            "file": frame.name,
            "path": str(frame.path),
            "is_calibration": int(frame.is_calibration),
        }
        row.update({f"ref_{key}": value for key, value in direct.items()})
        row.update({f"step_{key}": value for key, value in step_registration.items()})
        row.update({f"raw_{key}": value for key, value in raw_light.items()})
        row.update(_exif_metrics(frame.path))

        if fluidflower is not None:
            shape_corrected = _read_shape_corrected(fluidflower, frame.path)
            row.update(_pre_colorchecker_metrics(fluidflower, shape_corrected))
            corrected = _apply_color_corrections(fluidflower, shape_corrected)
            corrected_rgb = _rgb_from_corrected(corrected, args.max_dim)
            corrected_frames.append(corrected_rgb)
            corrected_light = _light_metrics(corrected_rgb, args.border_fraction)
            row.update(
                {f"corrected_{key}": value for key, value in corrected_light.items()}
            )
            row.update(_colorchecker_metrics(fluidflower))
            if exact_calibration_paths:
                row["is_calibration"] = int(
                    _normalise_path(frame.path) in exact_calibration_paths
                )

        rows.append(row)
        previous_features = current_features
        previous_scale = resize_scale
        if position == 1 or position % 20 == 0 or position == len(indices):
            print(f"  {run}: {position}/{len(indices)} sampled frames", flush=True)

    stable_mask = None
    _add_relative_light_metrics(rows, "raw")
    if rows and "corrected_full_luma" in rows[0]:
        _add_relative_light_metrics(rows, "corrected")
        stable_mask = _add_stable_light_metrics(rows, corrected_frames)

    run_out = args.out / run
    _write_csv(run_out / f"stability_{run}.csv", rows)
    if stable_mask is not None and corrected_frames:
        _save_stable_mask(
            run_out / f"stable_mask_{run}.png",
            corrected_frames[0],
            stable_mask,
        )
    if not args.no_plots:
        _plot_run(run, rows, run_out / f"stability_{run}.png")
    summary = _summarise_run(run, rows, len(frames))
    (run_out / f"summary_{run}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return summary


def _run_sort_key(path: Path) -> int:
    match = re.fullmatch(r"ac(\d+)", path.stem.lower())
    return int(match.group(1)) if match else 10**9


def _selected_runs(args: argparse.Namespace) -> list[str]:
    if args.all:
        paths = sorted(args.config_dir.glob("ac*.toml"), key=_run_sort_key)
        return [path.stem.lower() for path in paths if re.fullmatch(r"ac\d+", path.stem.lower())]
    if not args.runs:
        raise SystemExit("Provide --runs ... or --all")
    return [run.lower() for run in args.runs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--config-dir", type=Path, default=REPO / "config_seg6" / "run_ac")
    parser.add_argument("--common", type=Path, default=REPO / "config_seg6" / "common.toml")
    parser.add_argument("--out", type=Path, default=REPO / "logs" / "temporal_stability")
    parser.add_argument("--step", type=int, default=10, help="analyse every Nth protocol frame")
    parser.add_argument(
        "--start-end-only",
        action="store_true",
        help="analyse only the first and last protocol frame in each run",
    )
    parser.add_argument(
        "--checkpoint-hours",
        nargs="*",
        type=float,
        default=[],
        help="analyse first/last plus frames nearest these hours from injection start",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=160,
        help="cap regularly sampled frames per run; calibration frames are always added (0 disables)",
    )
    parser.add_argument("--max-dim", type=int, default=700)
    parser.add_argument("--registration-max-col", type=int, default=4800)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--keep-matches", type=int, default=800)
    parser.add_argument("--ransac-threshold-small-px", type=float, default=3.0)
    parser.add_argument("--border-fraction", type=float, default=0.12)
    parser.add_argument(
        "--corrected",
        action="store_true",
        help="also run sampled frames through DarSIA and record ColorChecker diagnostics",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.config_dir = args.config_dir.resolve()
    args.common = args.common.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    runs = _selected_runs(args)
    for run in runs:
        print(f"\n[{run}] temporal stability", flush=True)
        try:
            summary = process_run(run, args)
            summaries.append(summary)
            print(
                f"  {run}: max displacement={summary['max_ref_displacement_px']:.2f}px, "
                f"max raw luma change={summary['max_abs_raw_luma_change_pct']:.2f}%",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"run": run, "error": repr(exc)})
            print(f"  {run}: FAILED: {exc!r}", flush=True)

    if summaries:
        _write_csv(args.out / "stability_summary.csv", summaries)
    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "runs": runs,
            "step": args.step,
            "start_end_only": args.start_end_only,
            "checkpoint_hours": args.checkpoint_hours,
            "max_frames": args.max_frames,
            "max_dim": args.max_dim,
            "corrected": args.corrected,
        },
        "summaries": summaries,
        "failures": failures,
    }
    (args.out / "stability_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(
        f"\nCompleted {len(summaries)}/{len(runs)} run(s); output -> {args.out}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
