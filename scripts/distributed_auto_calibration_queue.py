"""Distributed auto-calibration using a file-based queue.

This script provides a master/worker workflow similar to distributed_analysis_queue.py,
but specialized for auto_calibrate_color_to_mass. It does not modify any existing
calibration scripts and only writes new queue/task/result files.

Usage (example):
  python scripts/distributed_auto_calibration_queue.py master ^
    --queue "\\Server\\AutoCalibQueue" ^
    --runs run6 run7 ^
    --max-iters 350 ^
    --warmup-iters 100 ^
    --warmup-levels 1.0,0.75,0.5 ^
    --run-mode serial ^
    --max-in-flight-per-run 30 ^
    --use-facies --per-label --use-last-best

  python scripts/distributed_auto_calibration_queue.py watchdog ^
    --queue "\\Server\\AutoCalibQueue" ^
    --workers 10
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import json
import math
import os
import random
import socket
import sys
import threading
import time
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import csv
import optuna
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

# Allow running from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
memmap_root = os.getenv("DARSIA_MEMMAP_ROOT")
if memmap_root:
    memmap_src = Path(memmap_root) / "src"
    if memmap_src.exists() and str(memmap_src) not in sys.path:
        sys.path.insert(0, str(memmap_src))

from auto_calibrate_color_to_mass import (  # noqa: E402
    PENALTY_VALUE,
    build_context,
    compute_auto_label_weights,
    apply_label_weight_grouping,
    evaluate_run,
    load_bounds_map,
    sample_params,
    save_best_calibration,
    suggest_params_trial,
    write_history_csv,
    _parse_signal_name,  # type: ignore
    _value_entries_by_label,  # type: ignore
)
try:  # Optional helper for seeding
    from auto_calib_init_loader import load_best_params_from_csv  # type: ignore
except Exception:  # pragma: no cover
    load_best_params_from_csv = None  # type: ignore


QUEUE_SUBDIRS = ["pending", "in_progress", "results", "done", "failed", "heartbeats", "worker_logs"]
PENDING_PICK_WINDOW = 32
MAX_CLAIM_ATTEMPTS = 5
CLAIM_JITTER_MAX_SECONDS = 0.5
READ_JITTER_MAX_SECONDS = 0.2


def _now() -> float:
    return time.time()


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-host"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def _ensure_queue_dirs(queue: Path) -> Dict[str, Path]:
    dirs = {}
    queue.mkdir(parents=True, exist_ok=True)
    for name in QUEUE_SUBDIRS:
        path = queue / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def _clear_queue(queue: Path) -> None:
    """Wipe the queue's task subfolders so a fresh master starts on an empty queue.
    Removes leftover pending/in_progress/results/done/failed/heartbeats from a previous,
    uncleared run (the usual source of 'orphan result' noise and stale tasks). Leaves
    the queue root and any non-task files (e.g. _commands watchdog registrations) intact.
    """
    import shutil
    n = 0
    for name in QUEUE_SUBDIRS:
        d = queue / name
        if not d.exists():
            continue
        for child in d.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
                n += 1
            except OSError:
                pass
    print(f"[master] --clear-queue: wiped {n} leftover item(s) from {queue}")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_write_json(
    path: Path, payload: dict, attempts: int = 3, delay: float = 0.05
) -> bool:
    for attempt in range(max(1, attempts)):
        try:
            _atomic_write_json(path, payload)
            return True
        except OSError:
            time.sleep(delay * (attempt + 1))
    try:
        tmp = path.with_suffix(
            path.suffix + f".{os.getpid()}.{int(_now() * 1000)}.tmp"
        )
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _safe_unlink(path: Path, attempts: int = 3, delay: float = 0.05) -> bool:
    for attempt in range(max(1, attempts)):
        try:
            path.unlink(missing_ok=True)
            return True
        except (FileNotFoundError, PermissionError, OSError):
            time.sleep(delay * (attempt + 1))
    return False


def _safe_exists(path: Path, attempts: int = 3, delay: float = 0.05) -> bool:
    for attempt in range(max(1, attempts)):
        try:
            return path.exists()
        except (PermissionError, OSError):
            if attempt + 1 >= max(1, attempts):
                return False
            time.sleep(delay * (attempt + 1))
    return False


def _load_json_retry(
    path: Path,
    attempts: int = 3,
    delay: float = 0.05,
    jitter: float = 0.0,
) -> Optional[dict]:
    for attempt in range(max(1, attempts)):
        try:
            return _load_json(path)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError, OSError):
            sleep_for = delay * (attempt + 1)
            if jitter:
                sleep_for += random.uniform(0.0, jitter)
            time.sleep(sleep_for)
    return None


def _pending_task_info(path: Path) -> Optional[Tuple[str, int, str]]:
    payload = _load_json_retry(path)
    if not payload:
        return None
    run = payload.get("run")
    if not run:
        return None
    phase = str(payload.get("phase") or "")
    try:
        seq = int(payload.get("seq", 0))
    except Exception:
        seq = 0
    return str(run), seq, phase


def _select_pending_task(
    dirs: Dict[str, Path],
    worker_id: str,
    preferred_run: Optional[str] = None,
    allow_sanity: bool = True,
) -> Optional[Path]:
    pending = list(dirs["pending"].glob("*.json"))
    if not pending:
        return None

    run_to_candidates: Dict[str, List[Tuple[int, Path, str]]] = {}
    for path in pending:
        info = _pending_task_info(path)
        if not info:
            continue
        run, seq, phase = info
        if not allow_sanity and phase == "sanity":
            continue
        run_to_candidates.setdefault(run, []).append((seq, path, phase))

    if not run_to_candidates:
        return None

    in_progress_counts: Dict[str, int] = {run: 0 for run in run_to_candidates}
    for path in dirs["in_progress"].glob("*.json"):
        payload = _load_json_retry(path)
        run = payload.get("run") if payload else None
        if not run:
            continue
        if run in in_progress_counts:
            in_progress_counts[run] = in_progress_counts.get(run, 0) + 1

    min_in_progress = min(in_progress_counts.values()) if in_progress_counts else 0

    for run, items in run_to_candidates.items():
        items.sort(key=lambda item: item[0])
        if len(items) > PENDING_PICK_WINDOW:
            run_to_candidates[run] = items[:PENDING_PICK_WINDOW]

    if preferred_run:
        preferred = run_to_candidates.get(preferred_run)
        if preferred and in_progress_counts.get(preferred_run, 0) <= min_in_progress:
            idx = abs(hash(worker_id)) % len(preferred)
            return preferred[idx][1]

    runs_sorted = sorted(
        run_to_candidates.keys(),
        key=lambda r: (in_progress_counts.get(r, 0), r),
    )
    if runs_sorted:
        offset = abs(hash(worker_id)) % len(runs_sorted)
        runs_sorted = runs_sorted[offset:] + runs_sorted[:offset]
    chosen_run = runs_sorted[0]
    candidates = run_to_candidates[chosen_run]
    idx = abs(hash(worker_id)) % len(candidates)
    return candidates[idx][1]


def _task_run_from_name(name: str) -> Optional[str]:
    base = name.split("__", 1)[0]
    parts = base.split("_", 1)
    if not parts:
        return None
    return parts[0] or None


def _parse_warmup_levels(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    levels = [float(x) for x in raw.split(",") if x.strip()]
    return levels or None


def _parse_param_ranges(raw: Optional[str]) -> Dict[int, Tuple[float, float]]:
    ranges: Dict[int, Tuple[float, float]] = {}
    if not raw:
        return ranges
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip().lower()
        if key.startswith("value"):
            key = key[5:]
        try:
            idx = int(key)
        except ValueError:
            continue
        bounds = [x.strip() for x in val.split(",") if x.strip()]
        if len(bounds) != 2:
            continue
        try:
            low = float(bounds[0])
            high = float(bounds[1])
        except ValueError:
            continue
        ranges[idx] = (low, high)
    return ranges


def _parse_label_weights(raw: Optional[str]) -> Dict[int, float]:
    if not raw:
        return {}
    weights: Dict[int, float] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        try:
            label = int(key.strip())
            weight = float(val.strip())
        except ValueError:
            continue
        weights[label] = weight
    return weights


def _build_quality_spec(scale: Optional[float], dtype: Optional[str]) -> Optional[Dict[str, Any]]:
    if scale is None:
        scale = 1.0
    try:
        scale_val = float(scale)
    except Exception:
        scale_val = 1.0
    dtype_val = dtype or None
    if scale_val == 1.0 and not dtype_val:
        return None
    return {"scale": scale_val, "dtype": dtype_val}


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


def _parse_param_levels(raw: Optional[str]) -> Tuple[Optional[int], Dict[int, int]]:
    if not raw:
        return None, {}
    raw = raw.strip()
    if "=" not in raw and ";" not in raw:
        try:
            return int(raw), {}
        except ValueError:
            return None, {}
    levels: Dict[int, int] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip().lower()
        if key.startswith("value"):
            key = key[5:]
        try:
            idx = int(key)
            count = int(val.strip())
        except ValueError:
            continue
        levels[idx] = count
    return None, levels


def _linspace(low: float, high: float, count: int) -> List[float]:
    if count <= 1:
        return [float(low)]
    step = (high - low) / float(count - 1)
    return [float(low + step * i) for i in range(count)]


def _apply_param_ranges(
    bounds_map: Dict[str, Dict[str, Tuple[float, float]]],
    runs: Optional[Sequence[str]],
    param_ranges: Dict[int, Tuple[float, float]],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    if not param_ranges:
        return bounds_map
    target_runs = list(runs) if runs else ["default"]
    for run in target_runs:
        override = bounds_map.setdefault(run, {})
        for idx, bounds in param_ranges.items():
            override[f"signal.label*.value{idx}"] = bounds
    return bounds_map


def _parse_bool(value: Optional[str]) -> bool:
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


def _parse_limit_text(text: str) -> Optional[int]:
    for token in text.replace(",", " ").split():
        try:
            return int(token)
        except ValueError:
            continue
    return None


def _read_limit_file(control_dir: Optional[str], basename: str) -> Optional[int]:
    if not control_dir:
        return None
    base = Path(control_dir)
    if not base.exists():
        return None
    for ext in (".txt", ".csv"):
        path = base / f"{basename}{ext}"
        if path.exists():
            try:
                return _parse_limit_text(path.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def _read_worker_limit(control_dir: Optional[str], host: str) -> Optional[int]:
    limit = _read_limit_file(control_dir, host)
    if limit is None:
        limit = _read_limit_file(control_dir, "default")
    return limit


def _read_cache_control(control_dir: Optional[str], host: str) -> Optional[int]:
    if not control_dir:
        return None
    base = Path(control_dir)
    for name in (f"{host}.image_cache.txt", "image_cache_size.txt"):
        path = base / name
        if not path.exists():
            continue
        try:
            return _parse_limit_text(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _configure_memmap_env(mode: Optional[str], cache_dir: Optional[str]) -> None:
    if mode:
        os.environ["DARSIA_MEMMAP_MODE"] = str(mode)
        if str(mode).strip().lower() not in ("off", "0", "false", "no"):
            os.environ.setdefault("DARSIA_MEMMAP_WRITE", "1")
    if cache_dir:
        os.environ["DARSIA_MEMMAP_DIR"] = str(cache_dir)


def _memmap_stats(run_name: Optional[str]) -> Dict[str, Any]:
    try:
        from darsia.utils import memmap_cache  # type: ignore
    except Exception:
        return {}
    try:
        return memmap_cache.stats(run_hint=run_name)
    except Exception:
        return {}


def _watchdog_state_path(control_dir: Optional[str], host: str) -> Optional[Path]:
    if not control_dir:
        return None
    return Path(control_dir) / f"{host}.watchdog.json"


def _write_watchdog_state(
    control_dir: Optional[str],
    host: str,
    desired_workers: int,
    workers_running: int,
    thread_limit: Optional[int],
    cache_size: Optional[int],
) -> None:
    path = _watchdog_state_path(control_dir, host)
    if path is None:
        return
    payload = {
        "timestamp": _now(),
        "desired_workers": desired_workers,
        "workers_running": workers_running,
        "thread_limit": thread_limit,
        "cache_size": cache_size,
    }
    _safe_write_json(path, payload)


def _read_watchdog_state(control_dir: Optional[str], host: str) -> Dict[str, Any]:
    path = _watchdog_state_path(control_dir, host)
    if path is None or not path.exists():
        return {}
    payload = _load_json_retry(path)
    return payload or {}


def _sanity_lock_path(control_dir: Optional[str], host: str) -> Optional[Path]:
    if not control_dir:
        return None
    return Path(control_dir) / f"{host}.sanity.lock.json"


def _sanity_lock_active(control_dir: Optional[str], host: str, stale_seconds: float) -> bool:
    path = _sanity_lock_path(control_dir, host)
    if path is None or not path.exists():
        return False
    payload = _load_json_retry(path)
    timestamp = None
    if isinstance(payload, dict):
        timestamp = payload.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        try:
            timestamp = path.stat().st_mtime
        except OSError:
            return False
    if (_now() - float(timestamp)) > stale_seconds:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def _acquire_sanity_lock(
    control_dir: Optional[str],
    host: str,
    worker_id: str,
    stale_seconds: float,
) -> bool:
    path = _sanity_lock_path(control_dir, host)
    if path is None:
        return True
    if path.exists():
        payload = _load_json_retry(path)
        timestamp = None
        if isinstance(payload, dict):
            timestamp = payload.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            try:
                timestamp = path.stat().st_mtime
            except OSError:
                timestamp = None
        if timestamp is not None and (_now() - float(timestamp)) <= stale_seconds:
            if isinstance(payload, dict) and payload.get("worker_id") == worker_id:
                return True
            return False
        try:
            path.unlink()
        except OSError:
            pass
    payload = {"timestamp": _now(), "worker_id": worker_id}
    _safe_write_json(path, payload, attempts=3, delay=0.1)
    current = _load_json_retry(path)
    return isinstance(current, dict) and current.get("worker_id") == worker_id


def _release_sanity_lock(control_dir: Optional[str], host: str, worker_id: str) -> None:
    path = _sanity_lock_path(control_dir, host)
    if path is None or not path.exists():
        return
    payload = _load_json_retry(path)
    if isinstance(payload, dict) and payload.get("worker_id") not in (None, worker_id):
        return
    try:
        path.unlink()
    except OSError:
        pass


def _worker_state_path(log_dir: Path, worker_id: str) -> Path:
    return log_dir / f"{worker_id}.state.json"


def _read_worker_state(log_dir: Path, worker_id: str) -> Dict[str, Any]:
    path = _worker_state_path(log_dir, worker_id)
    if not path.exists():
        return {}
    payload = _load_json_retry(path)
    return payload or {}


def _write_worker_state(log_dir: Path, worker_id: str, payload: Dict[str, Any]) -> None:
    path = _worker_state_path(log_dir, worker_id)
    _atomic_write_json(path, payload)


def _read_inflight_limit(control_dir: Optional[str]) -> Optional[int]:
    if not control_dir:
        return None
    base = Path(control_dir)
    if not base.exists():
        return None
    for name in ("max_in_flight_per_run", "max_in_flight"):
        for ext in (".txt", ".csv"):
            path = base / f"{name}{ext}"
            if not path.exists():
                continue
            try:
                return _parse_limit_text(path.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def _set_thread_env(limit: Optional[int]) -> None:
    if not limit:
        return
    value = str(int(limit))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENCV_NUM_THREADS",
    ):
        os.environ[name] = value


def _parse_worker_index(worker_id: str) -> Optional[int]:
    if "_" not in worker_id:
        return None
    suffix = worker_id.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _summarize_array(value: Any) -> Dict[str, Any]:
    if np is None:
        return {"note": "numpy-unavailable"}
    arr = np.array(value, dtype=float)
    if arr.size == 0:
        return {"empty": True, "shape": list(arr.shape)}
    return {
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "mean": float(np.nanmean(arr)),
        "max": float(np.nanmax(arr)),
    }


def _summarize_debug(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _summarize_debug(v) for k, v in value.items()}
    if np is not None and isinstance(value, np.ndarray):
        return _summarize_array(value)
    if hasattr(value, "shape"):
        try:
            return _summarize_array(value)
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple)):
        if np is not None:
            try:
                arr = np.array(value, dtype=float)
                if arr.size > 64 or arr.ndim > 1:
                    return _summarize_array(arr)
            except Exception:
                return {"len": len(value)}
        return list(value)
    return value


def _generate_warmup_params(
    context,
    warmup_iters: int,
    warmup_levels: Optional[Sequence[float]],
    warmup_levels_by_idx: Optional[Dict[int, int]],
    warmup_levels_default: Optional[int],
    warmup_high: Optional[float],
    warmup_mode: str,
) -> List[Dict[str, Any]]:
    warmups: List[Dict[str, Any]] = []

    val_entries_by_label = _value_entries_by_label(context.param_space)
    template_label = None
    if val_entries_by_label:
        if context.signal_label in val_entries_by_label:
            template_label = context.signal_label
        else:
            template_label = sorted(val_entries_by_label.keys())[0]

    template_entries: List[Dict[str, Any]] = []
    if template_label is not None:
        template_entries = [
            e
            for e in val_entries_by_label.get(template_label, [])
            if (_parse_signal_name(e["name"]) or (None, 0))[1] != 0
        ]

    bounds_by_idx: Dict[int, Tuple[float, float]] = {}
    high_by_idx: Dict[int, float] = {}
    for entry in template_entries:
        parsed = _parse_signal_name(entry["name"])
        if not parsed:
            continue
        _, idx = parsed
        low, high = entry["bounds"]
        if warmup_high is not None:
            high = min(high, warmup_high)
        bounds_by_idx[idx] = (low, high)
        high_by_idx[idx] = high

    high_lookup: Dict[str, float] = {}
    for entries in val_entries_by_label.values():
        for entry in entries:
            parsed = _parse_signal_name(entry["name"])
            if not parsed:
                continue
            _, idx = parsed
            if idx == 0:
                continue
            hi = entry["bounds"][1]
            if warmup_high is not None:
                hi = min(hi, warmup_high)
            high_lookup[entry["name"]] = hi
    warmup_levels_by_idx = warmup_levels_by_idx or {}
    use_grid_levels = bool(warmup_levels_by_idx or warmup_levels_default is not None)
    default_levels = [1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.3, 0.1]
    if use_grid_levels:
        default_count = warmup_levels_default
        if default_count is None:
            default_count = len(warmup_levels) if warmup_levels else len(default_levels)
    else:
        explicit_levels = list(warmup_levels) if warmup_levels else default_levels

    def _levels_for_idx(idx: int, low: float, high: float) -> List[float]:
        if use_grid_levels:
            count = warmup_levels_by_idx.get(idx, default_count)
            return _linspace(low, high, count)
        return list(explicit_levels)

    # All high within bounds
    all_high = {}
    for entry in context.param_space:
        if entry["name"] in high_lookup:
            all_high[entry["name"]] = high_lookup[entry["name"]]
        else:
            all_high[entry["name"]] = entry["bounds"][1]
    warmups.append(all_high)

    # All low within bounds
    warmups.append({e["name"]: e["bounds"][0] for e in context.param_space})

    # Mid values
    mid = {}
    for entry in context.param_space:
        if entry["name"] in high_lookup:
            mid[entry["name"]] = 0.5 * (entry["bounds"][0] + high_lookup[entry["name"]])
        else:
            mid[entry["name"]] = 0.5 * (entry["bounds"][0] + entry["bounds"][1])
    warmups.append(mid)

    def _apply_profile_values(profile: Dict[str, Any], idx_values: Dict[int, float]) -> None:
        for entries in val_entries_by_label.values():
            for entry in entries:
                parsed = _parse_signal_name(entry["name"])
                if not parsed:
                    continue
                _, idx = parsed
                if idx not in idx_values:
                    continue
                low, high = entry["bounds"]
                if warmup_high is not None:
                    high = min(high, warmup_high)
                profile[entry["name"]] = max(low, min(high, idx_values[idx]))

    def _apply_label_values(profile: Dict[str, Any], label: int, idx_values: Dict[int, float]) -> None:
        for entry in val_entries_by_label.get(label, []):
            parsed = _parse_signal_name(entry["name"])
            if not parsed:
                continue
            _, idx = parsed
            if idx not in idx_values:
                continue
            low, high = entry["bounds"]
            if warmup_high is not None:
                high = min(high, warmup_high)
            profile[entry["name"]] = max(low, min(high, idx_values[idx]))

    if getattr(context, "per_label_params", False) and val_entries_by_label:
        labels = sorted(val_entries_by_label.keys())
        for label in labels:
            label_entries = [
                e
                for e in val_entries_by_label.get(label, [])
                if (_parse_signal_name(e["name"]) or (None, 0))[1] != 0
            ]
            if not label_entries:
                continue
            for idx_pos, entry in enumerate(label_entries):
                parsed = _parse_signal_name(entry["name"])
                if not parsed:
                    continue
                _, idx_pos_val = parsed
                low, high = entry["bounds"]
                if warmup_high is not None:
                    high = min(high, warmup_high)
                levels = _levels_for_idx(idx_pos_val, low, high)
                for lvl in levels:
                    profile = dict(all_high)
                    idx_values: Dict[int, float] = {}
                    for j, e in enumerate(label_entries):
                        parsed = _parse_signal_name(e["name"])
                        if not parsed:
                            continue
                        _, idx = parsed
                        low_j, high_j = e["bounds"]
                        if warmup_high is not None:
                            high_j = min(high_j, warmup_high)
                        if warmup_mode == "prefix":
                            if j <= idx_pos:
                                idx_values[idx] = max(low_j, min(high_j, lvl))
                            else:
                                idx_values[idx] = high_j
                        elif warmup_mode == "suffix":
                            if j < idx_pos:
                                idx_values[idx] = low_j
                            else:
                                idx_values[idx] = max(low_j, min(high_j, lvl))
                        else:
                            if j == idx_pos:
                                idx_values[idx] = max(low_j, min(high_j, lvl))
                    _apply_label_values(profile, label, idx_values)
                    warmups.append(profile)
    elif template_entries:
        if warmup_mode == "prefix":
            for idx_pos, entry in enumerate(template_entries):
                parsed = _parse_signal_name(entry["name"])
                if not parsed:
                    continue
                _, idx_pos_val = parsed
                low, high = bounds_by_idx.get(idx_pos_val, entry["bounds"])
                levels = _levels_for_idx(idx_pos_val, low, high)
                for lvl in levels:
                    profile = dict(all_high)
                    idx_values: Dict[int, float] = {}
                    for j, e in enumerate(template_entries):
                        parsed = _parse_signal_name(e["name"])
                        if not parsed:
                            continue
                        _, idx = parsed
                        low_j, high_j = bounds_by_idx.get(idx, e["bounds"])
                        if j <= idx_pos:
                            idx_values[idx] = max(low_j, min(high_j, lvl))
                        else:
                            idx_values[idx] = high_j
                    _apply_profile_values(profile, idx_values)
                    warmups.append(profile)
        elif warmup_mode == "suffix":
            for idx_pos, entry in enumerate(template_entries):
                parsed = _parse_signal_name(entry["name"])
                if not parsed:
                    continue
                _, idx_pos_val = parsed
                low, high = bounds_by_idx.get(idx_pos_val, entry["bounds"])
                levels = _levels_for_idx(idx_pos_val, low, high)
                for lvl in levels:
                    profile = dict(all_high)
                    idx_values: Dict[int, float] = {}
                    for j, e in enumerate(template_entries):
                        parsed = _parse_signal_name(e["name"])
                        if not parsed:
                            continue
                        _, idx = parsed
                        low_j, high_j = bounds_by_idx.get(idx, e["bounds"])
                        if j < idx_pos:
                            idx_values[idx] = low_j
                        else:
                            idx_values[idx] = max(low_j, min(high_j, lvl))
                    _apply_profile_values(profile, idx_values)
                    warmups.append(profile)
        else:
            for entry in template_entries:
                parsed = _parse_signal_name(entry["name"])
                if not parsed:
                    continue
                _, idx = parsed
                low, high = bounds_by_idx.get(idx, entry["bounds"])
                levels = _levels_for_idx(idx, low, high)
                for lvl in levels:
                    profile = dict(all_high)
                    _apply_profile_values(profile, {idx: max(low, min(high, lvl))})
                    warmups.append(profile)

    seen = set()
    unique: List[Dict[str, Any]] = []
    for _ in range(max(0, warmup_iters)):
        warmups.append(sample_params(context.param_space))
    for wp in warmups:
        key = tuple(sorted(wp.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(wp)
    return unique


def _append_temp_csv(path: Path, row: Dict[str, Any], header_written: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iter",
        "objective",
        "feasible",
        "status",
        "params",
        "metrics",
        "best",
        "debug_pre",
        "debug_scale",
        "debug_params",
        "settings",
        "sanity",
    ]
    mode = "a"
    # Retry on transient Windows file locks (PermissionError [Errno 13] from antivirus,
    # cloud sync, or a concurrent reader such as the stats UI) so a momentary lock never
    # crashes the whole master. If it stays locked, log and skip the row (the result is
    # still tracked in the in-memory history and written on the next successful append).
    last_exc = None
    for _attempt in range(12):
        try:
            with path.open(mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerow(row)
            return header_written
        except (PermissionError, OSError) as exc:
            last_exc = exc
            time.sleep(0.3)
    print(f"[master] WARNING: could not write {path.name} after retries ({last_exc}); row skipped")
    return header_written


@dataclass
class TaskInfo:
    task_id: str
    run: str
    phase: str
    seq: int
    params: Dict[str, Any]
    trial: Optional[optuna.trial.Trial] = None
    attempt: int = 0
    sent_at: float = field(default_factory=_now)


@dataclass
class RunState:
    run: str
    context: Any
    label_weights: Dict[int, float]
    study: optuna.Study
    warmup_params: List[Dict[str, Any]]
    max_iters: int
    distributions: Dict[str, optuna.distributions.BaseDistribution]
    init_params: Optional[Dict[str, Any]] = None
    init_dispatched: bool = False
    init_done: bool = False
    warmup_cursor: int = 0
    warmup_done: int = 0
    optuna_done: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    header_written: bool = False
    sanity_header_written: bool = False
    best_eval: Optional[Any] = None
    best_params: Optional[Dict[str, Any]] = None
    best_seq: Optional[int] = None
    best_row: Optional[Dict[str, Any]] = None
    last_sanity_completed: Optional[int] = None
    sanity_pending: bool = False
    best_feasible: Optional[Any] = None
    best_feasible_params: Optional[Dict[str, Any]] = None
    baseline_eval: Optional[Any] = None
    next_iter: int = 0
    phase: str = "warmup"


def _release_run_context(state: RunState) -> None:
    if state.context is None:
        return
    state.context = None
    state.warmup_params = []
    gc.collect()


def _make_task_id(run: str, phase: str, seq: int) -> str:
    stamp = int(_now() * 1000)
    rand = random.randint(1000, 9999)
    return f"{run}_{phase}_{seq}_{stamp}_{rand}"


def _task_payload(
    task_id: str,
    run: str,
    phase: str,
    params: Dict[str, Any],
    seq: int,
    config_dir: Path,
    use_facies: bool,
    per_label_params: bool,
    use_label_weights: bool,
    label_weights: Optional[Dict[int, float]],
    enforce_lower: bool,
    objective_integral: str,
    bounds_file: Optional[Path],
    trial_number: Optional[int] = None,
    attempt: int = 0,
    quality: Optional[Dict[str, Any]] = None,
    sanity: Optional[Dict[str, Any]] = None,
) -> dict:
    payload = {
        "task_id": task_id,
        "run": run,
        "phase": phase,
        "seq": seq,
        "params": params,
        "trial_number": trial_number,
        "config_dir": str(config_dir),
        "use_facies": bool(use_facies),
        "per_label_params": bool(per_label_params),
        "use_label_weights": bool(use_label_weights),
        "label_weights": label_weights or {},
        "enforce_lower": bool(enforce_lower),
        "objective_integral": objective_integral,
        "bounds_file": str(bounds_file) if bounds_file else None,
        "attempt": attempt,
        "created_at": _now(),
    }
    if quality:
        payload["quality"] = quality
    if sanity:
        payload["sanity"] = sanity
    return payload


def _pick_run_order(
    runs: Sequence[str],
    run_mode: str,
    current: Optional[str],
    run_states: Optional[Mapping[str, RunState]] = None,
    max_active_runs: int = 0,
) -> List[str]:
    if run_mode == "serial":
        return [current] if current else []
    if max_active_runs > 0 and run_states is not None:
        active = [
            run for run in runs
            if run in run_states and getattr(run_states[run], "phase", None) != "done"
        ]
        for run in runs:
            if len(active) >= max_active_runs:
                break
            state = run_states.get(run)
            if state is not None:
                if getattr(state, "phase", None) == "done":
                    continue
                if run not in active:
                    active.append(run)
                continue
            active.append(run)
        return active
    return list(runs)


def _choose_best(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not history:
        return None
    feasible = [row for row in history if bool(row.get("feasible"))]
    pool = feasible or history
    return min(pool, key=lambda r: float(r.get("objective", PENALTY_VALUE)))


def _build_distributions(
    param_space: Sequence[Mapping[str, Any]],
) -> Dict[str, optuna.distributions.BaseDistribution]:
    distributions: Dict[str, optuna.distributions.BaseDistribution] = {}
    for entry in param_space:
        name = entry["name"]
        low, high = entry["bounds"]
        if entry.get("type", "float") == "int":
            distributions[name] = optuna.distributions.IntDistribution(int(low), int(high))
        else:
            distributions[name] = optuna.distributions.FloatDistribution(float(low), float(high))
    return distributions


def _finalize_run(
    run_state: RunState,
    logs_dir: Path,
    out_folder: Path,
    *,
    save_calibration: bool = True,
) -> None:
    history = sorted(run_state.history, key=lambda r: int(r.get("iter", 0)))
    for row in history:
        row["best"] = False
    chosen = _choose_best(history)
    if chosen:
        chosen["best"] = True
    history_path = logs_dir / f"auto_calibration_{run_state.run}.csv"
    write_history_csv(history_path, history)
    if run_state.best_eval is None:
        return
    if run_state.baseline_eval and run_state.baseline_eval.objective <= run_state.best_eval.objective:
        return
    # Re-evaluate the best parameters at FULL scale on the master context (which is
    # always built without quality downscaling/dtype-cast), so the final, persisted
    # calibration is validated and logged at full resolution regardless of whatever
    # reduced quality the workers used during the search.
    if run_state.best_params and run_state.context is not None:
        try:
            full_eval = evaluate_run(run_state.context, run_state.best_params)
            (logs_dir / f"final_full_scale_{run_state.run}.json").write_text(
                json.dumps(
                    {
                        "run": run_state.run,
                        "objective_full_scale": full_eval.objective,
                        "metrics": {k: vars(v) for k, v in full_eval.metrics.items()},
                        "params": run_state.best_params,
                    },
                    default=str,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[master] [{run_state.run}] full-scale finalise objective={full_eval.objective:.6f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[master] [{run_state.run}] full-scale finalise eval failed: {exc!r}", flush=True)
    if save_calibration:
        save_best_calibration(run_state.context, run_state.best_params, out_folder)
    elif run_state.best_params:
        try:
            out_folder.mkdir(parents=True, exist_ok=True)
            (out_folder / "best_params.json").write_text(
                json.dumps(run_state.best_params, indent=2, default=str),
                encoding="utf-8",
            )
            print(
                f"[master] [{run_state.run}] no-save-calibration: wrote best params to {out_folder}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[master] [{run_state.run}] no-save-calibration best_params write failed: {exc!r}",
                flush=True,
            )


def _load_history_trials(
    history_path: Path,
    distributions: Mapping[str, optuna.distributions.BaseDistribution],
) -> List[optuna.trial.FrozenTrial]:
    trials: List[optuna.trial.FrozenTrial] = []
    if not history_path.exists():
        return trials
    with history_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if status.startswith("eval-error"):
                continue
            params_raw = row.get("params")
            if not params_raw:
                continue
            try:
                params = ast.literal_eval(params_raw)
            except Exception:
                try:
                    params = json.loads(params_raw)
                except Exception:
                    continue
            if not isinstance(params, dict):
                continue
            dist_keys = set(distributions.keys())
            if not dist_keys.issubset(params.keys()):
                continue
            params = {k: params[k] for k in dist_keys}
            try:
                objective = float(row.get("objective", "nan"))
            except Exception:
                continue
            if not math.isfinite(objective):
                continue
            trial = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=objective,
            )
            trials.append(trial)
    return trials


def _parse_history_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        return _parse_bool(str(value))
    except Exception:
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_history_mapping(raw: Any) -> Any:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    raw_str = str(raw)
    if not raw_str:
        return {}
    try:
        parsed = ast.literal_eval(raw_str)
    except Exception:
        try:
            parsed = json.loads(raw_str)
        except Exception:
            return raw
    return parsed if isinstance(parsed, dict) else raw


def _load_history_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, str, str, Any]] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get("status", "")
                    try:
                        iter_val = int(row.get("iter", "-1"))
                    except Exception:
                        iter_val = -1
                    try:
                        objective_val = float(row.get("objective", "nan"))
                    except Exception:
                        objective_val = float("nan")
                    params_val = _parse_history_mapping(row.get("params"))
                    params_key = json.dumps(params_val, sort_keys=True, default=str) if isinstance(params_val, dict) else str(params_val)
                    key = (iter_val, params_key, str(status), objective_val)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "iter": iter_val,
                            "objective": objective_val,
                            "feasible": _parse_history_bool(row.get("feasible")),
                            "status": status,
                            "params": params_val,
                            "metrics": _parse_history_mapping(row.get("metrics")),
                            "best": False,
                            "debug_pre": row.get("debug_pre", ""),
                            "debug_scale": row.get("debug_scale", ""),
                            "debug_params": row.get("debug_params", ""),
                            "settings": row.get("settings", ""),
                            "sanity": row.get("sanity", ""),
                        }
                    )
        except Exception:
            continue
    return rows


def _load_params_from_csv(
    csv_path: Path,
    seq: int,
) -> Optional[Dict[str, Any]]:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    iter_val = int(row.get("iter", "-1"))
                except Exception:
                    continue
                if iter_val != seq:
                    continue
                params_raw = row.get("params")
                if not params_raw:
                    return None
                try:
                    params = ast.literal_eval(params_raw)
                except Exception:
                    try:
                        params = json.loads(params_raw)
                    except Exception:
                        return None
                if isinstance(params, dict):
                    return params
    except Exception:
        return None
    return None


def requeue_failed_main(args: argparse.Namespace) -> None:
    queue = Path(args.queue)
    dirs = _ensure_queue_dirs(queue)
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    runs = set(args.runs or [])
    label_weights = _parse_label_weights(getattr(args, "label_weights", None))
    quality_spec = _build_quality_spec(args.quality_scale, args.quality_dtype)
    sanity_scale = args.sanity_scale if args.sanity_scale is not None else args.quality_scale
    sanity_dtype = args.sanity_dtype if args.sanity_dtype is not None else args.quality_dtype
    moved_dir = dirs["failed"] / "requeued"
    if not args.keep_failed:
        moved_dir.mkdir(parents=True, exist_ok=True)

    failed_files = list(dirs["failed"].glob("*.json"))
    requeued = 0
    skipped = 0
    for failed_path in failed_files:
        payload = _load_json(failed_path)
        run = payload.get("run")
        if runs and run not in runs:
            continue
        params = payload.get("params")
        if not params and logs_dir and run is not None:
            seq = int(payload.get("seq", -1))
            tmp_path = logs_dir / f"tmp_auto_calibration_{run}.csv"
            hist_path = logs_dir / f"auto_calibration_{run}.csv"
            params = _load_params_from_csv(tmp_path, seq) or _load_params_from_csv(hist_path, seq)
        if not params:
            skipped += 1
            continue
        seq = int(payload.get("seq", 0))
        phase = payload.get("phase") or "optuna"
        task_id = _make_task_id(run, phase, seq)
        new_payload = _task_payload(
            task_id,
            run,
            phase,
            params,
            seq,
            Path(args.config_dir),
            args.use_facies,
            args.per_label,
            args.use_label_weights,
            label_weights,
            args.enforce_lower,
            args.objective_integral,
            Path(args.bounds_file) if args.bounds_file else None,
            attempt=0,
            quality=quality_spec,
        )
        new_payload["manual_requeue"] = True
        new_payload["source_failed"] = failed_path.name
        _atomic_write_json(dirs["pending"] / f"{task_id}.json", new_payload)
        requeued += 1
        if not args.keep_failed:
            os.replace(failed_path, moved_dir / failed_path.name)

    print(f"Requeued {requeued} tasks; skipped {skipped} (missing params).")

def _run_tag(args) -> str:
    """Subfolder name reflecting the run configuration (facies/per-label/warmup/optuna/run-mode + timestamp)."""
    from datetime import datetime as _dt
    parts = [
        "facies{}".format(1 if getattr(args, "use_facies", False) else 0),
        "perlabel{}".format(1 if getattr(args, "per_label", False) else 0),
        "warmup{}".format(getattr(args, "warmup_iters", 0)),
        "optuna{}".format(getattr(args, "max_iters", 0)),
        str(getattr(args, "run_mode", "serial")),
    ]
    return "_".join(parts) + "_" + _dt.now().strftime("%Y%m%d_%H%M")


def _write_master_commands(logs_dir: Path, queue: Path, args) -> None:
    """Write commands.txt: the master command + every contributing machine's
    watchdog command (registered in <queue>/_commands by each watchdog)."""
    try:
        lines = [
            "# Auto-calibration run commands",
            "# logs_dir: {}".format(logs_dir),
            "# queue:    {}".format(queue),
            "# runs:     {}".format(" ".join(getattr(args, "runs", []) or [])),
            "",
            "## MASTER ({})".format(_hostname()),
            " ".join(sys.argv),
            "",
            "## WATCHDOGS",
        ]
        cmd_dir = queue / "_commands"
        files = sorted(cmd_dir.glob("watchdog_*.txt")) if cmd_dir.exists() else []
        if files:
            for f in files:
                try:
                    lines.append("# {}".format(f.stem))
                    lines.append(f.read_text(encoding="utf-8").strip())
                    lines.append("")
                except Exception:
                    pass
        else:
            lines.append("# (none registered yet - each watchdog appends here on start)")
        (logs_dir / "commands.txt").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def master_main(args: argparse.Namespace) -> None:
    queue = Path(args.queue)
    if not getattr(args, "no_clear_queue", False):
        _clear_queue(queue)   # clear leftover/orphan tasks on startup BY DEFAULT
    dirs = _ensure_queue_dirs(queue)

    logs_root = Path(args.logs_dir)
    logs_dir = logs_root if getattr(args, "exact_logs_dir", False) else logs_root / _run_tag(args)
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Seed an editable in-flight-per-run control file so the dispatch cap can be
    # changed live (the master re-reads max_in_flight_per_run.txt each loop and it
    # overrides --max-in-flight-per-run). Only seeds if absent (never clobbers).
    if getattr(args, "control_dir", None):
        try:
            _cd = Path(args.control_dir); _cd.mkdir(parents=True, exist_ok=True)
            _inf = _cd / "max_in_flight_per_run.txt"
            if not _inf.exists():
                _inf.write_text(
                    str(args.max_in_flight_per_run or args.max_in_flight or 0),
                    encoding="utf-8")
        except Exception:
            pass
    _write_master_commands(logs_dir, queue, args)
    _configure_memmap_env(getattr(args, "memmap_cache", None), getattr(args, "memmap_dir", None))
    master_log_path = dirs["worker_logs"] / f"master_{_hostname()}.log"
    orphan_dir = dirs["failed"] / "orphan_results"
    orphan_dir.mkdir(parents=True, exist_ok=True)

    bounds_map = load_bounds_map(Path(args.bounds_file)) if args.bounds_file else {}

    from darsia.presets.workflows.rig import Rig

    runs = args.runs
    param_ranges = _parse_param_ranges(args.param_ranges)
    warmup_levels_default, warmup_levels_by_idx = _parse_param_levels(args.param_levels)
    bounds_map = _apply_param_ranges(bounds_map, runs, param_ranges)
    label_weights = _parse_label_weights(getattr(args, "label_weights", None))
    quality_spec = _build_quality_spec(args.quality_scale, args.quality_dtype)
    sanity_scale = args.sanity_scale if args.sanity_scale is not None else args.quality_scale
    sanity_dtype = args.sanity_dtype if args.sanity_dtype is not None else args.quality_dtype
    run_states: Dict[str, RunState] = {}
    in_flight: Dict[str, TaskInfo] = {}
    stale_counts: Dict[str, int] = {}
    master_settings = {
        "use_facies": args.use_facies,
        "per_label": args.per_label,
        "use_label_weights": args.use_label_weights,
        "auto_label_weights": args.auto_label_weights,
        "label_weights": label_weights,
        "label_weight_grouping": args.label_weight_grouping,
        "enforce_lower": args.enforce_lower,
        "objective_integral": args.objective_integral,
        "no_save_calibration": args.no_save_calibration,
        "use_last_best": args.use_last_best,
        "use_history": args.use_history,
        "bounds_file": args.bounds_file,
        "param_ranges": args.param_ranges,
        "param_levels": args.param_levels,
        "warmup_iters": args.warmup_iters,
        "warmup_levels": args.warmup_levels,
        "warmup_high": args.warmup_high,
        "warmup_mode": args.warmup_mode,
        "skip_warmup": args.skip_warmup,
        "max_iters": args.max_iters,
        "run_mode": args.run_mode,
        "max_in_flight": args.max_in_flight,
        "max_in_flight_per_run": args.max_in_flight_per_run,
        "max_active_runs": getattr(args, "max_active_runs", 0),
        "max_retries": args.max_retries,
        "task_timeout_minutes": args.task_timeout_minutes,
        "heartbeat_timeout_seconds": args.heartbeat_timeout_seconds,
        "memmap_cache": args.memmap_cache,
        "memmap_dir": args.memmap_dir,
        "quality_scale": args.quality_scale,
        "quality_dtype": args.quality_dtype,
        "sanity_every": getattr(args, "sanity_every", 0),
        "sanity_scale": sanity_scale,
        "sanity_dtype": sanity_dtype,
        "titration_flash": os.environ.get("FFAC_TITRATION_FLASH", ""),
        "titration_recipe": os.environ.get("FFAC_TITRATION_RECIPE", ""),
        "static_light_correction": os.environ.get("FFAC_STATIC_LIGHT_CORRECTION", ""),
        "couple_aq_gas": os.environ.get("FFAC_COUPLE_AQ_GAS", ""),
    }

    def _log_master(message: str) -> None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}Z] {message}"
        print(line, flush=True)
        try:
            with master_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    master_proc = psutil.Process(os.getpid()) if psutil else None
    master_mem_interval = max(1.0, float(getattr(args, "master_mem_log_seconds", 60.0) or 60.0))
    last_master_mem = 0.0

    def _log_master_mem(force: bool = False) -> None:
        nonlocal last_master_mem
        if master_proc is None or psutil is None:
            return
        now = _now()
        if not force and (now - last_master_mem) < master_mem_interval:
            return
        last_master_mem = now
        _write_master_commands(logs_dir, queue, args)
        try:
            mem = master_proc.memory_info()
            vm = psutil.virtual_memory()
            _log_master(
                "master_mem rss_mb={:.2f} vms_mb={:.2f} sys_total_gb={:.2f} "
                "sys_avail_gb={:.2f} sys_used_pct={:.1f}".format(
                    mem.rss / (1024 * 1024),
                    mem.vms / (1024 * 1024),
                    vm.total / (1024 * 1024 * 1024),
                    vm.available / (1024 * 1024 * 1024),
                    vm.percent,
                )
            )
        except Exception:
            return

    def _init_run_state(run: str) -> RunState:
        ctx = build_context(
            run=run,
            config_dir=Path(args.config_dir),
            rig_cls=Rig,
            ref_config_path=Path(args.ref_config) if args.ref_config else None,
            use_facies=args.use_facies,
            bounds_map=bounds_map,
            enforce_lower=args.enforce_lower,
            per_label_params=args.per_label,
            use_label_weights=args.use_label_weights,
            label_weights=label_weights,
            static_light_correction=os.environ.get("FFAC_STATIC_LIGHT_CORRECTION", "off"),
        )
        run_label_weights = label_weights
        if args.use_label_weights and args.auto_label_weights and not label_weights:
            auto_weights = compute_auto_label_weights(ctx)
            if auto_weights:
                auto_weights = apply_label_weight_grouping(
                    auto_weights, ctx, args.label_weight_grouping
                )
                run_label_weights = auto_weights
                ctx.label_weights = auto_weights
                _log_master(f"[{run}] auto label weights -> {auto_weights}")
            else:
                _log_master(f"[{run}] auto label weights empty; using unweighted objective")
        elif args.label_weight_grouping and run_label_weights:
            run_label_weights = apply_label_weight_grouping(
                run_label_weights, ctx, args.label_weight_grouping
            )
            ctx.label_weights = run_label_weights
        history_path = logs_dir / f"auto_calibration_{run}.csv"
        tmp_history_path = logs_dir / f"tmp_auto_calibration_{run}.csv"
        sanity_tmp_path = logs_dir / f"tmp_auto_calibration_sanity_{run}.csv"
        history_source = (
            history_path if history_path.exists() else (tmp_history_path if tmp_history_path.exists() else None)
        )

        init_params = None
        if args.use_last_best and load_best_params_from_csv is not None and history_source is not None:
            try:
                init_params = load_best_params_from_csv(history_source)
            except Exception:
                init_params = None

        if args.skip_warmup or ((args.use_last_best or args.use_history) and history_source is not None):
            warmups = []
        else:
            warmups = _generate_warmup_params(
                ctx,
                warmup_iters=args.warmup_iters,
                warmup_levels=_parse_warmup_levels(args.warmup_levels),
                warmup_levels_by_idx=warmup_levels_by_idx,
                warmup_levels_default=warmup_levels_default,
                warmup_high=args.warmup_high,
                warmup_mode=args.warmup_mode,
            )
        distributions = _build_distributions(ctx.param_space)
        if getattr(args, "optuna_persist", False):
            storage_dir = Path(args.optuna_storage_dir) if args.optuna_storage_dir else logs_dir
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage_path = storage_dir / f"optuna_{run}.db"
            storage_uri = f"sqlite:///{storage_path}"
            study = optuna.create_study(
                direction="minimize",
                study_name=f"{run}_optuna",
                storage=storage_uri,
                load_if_exists=True,
            )
            if args.use_history and history_source is not None:
                if study.trials:
                    _log_master(
                        f"[{run}] optuna persisted study has {len(study.trials)} trials; skip CSV history load"
                    )
                else:
                    for trial in _load_history_trials(history_source, distributions):
                        study.add_trial(trial)
        else:
            study = optuna.create_study(direction="minimize")
            if args.use_history and history_source is not None:
                for trial in _load_history_trials(history_source, distributions):
                    study.add_trial(trial)
        state = RunState(
            run=run,
            context=ctx,
            label_weights=run_label_weights,
            study=study,
            warmup_params=warmups,
            max_iters=args.max_iters,
            distributions=distributions,
            init_params=init_params,
        )
        history_rows = _load_history_rows([history_path, tmp_history_path])
        if history_rows:
            state.history.extend(history_rows)
            max_iter = max(
                (row.get("iter", -1) for row in history_rows if isinstance(row.get("iter"), int)),
                default=-1,
            )
            if max_iter >= 0:
                state.next_iter = max_iter + 1
            chosen = _choose_best(history_rows)
            if chosen is not None:
                try:
                    state.best_eval = type(
                        "Eval",
                        (),
                        {"objective": float(chosen.get("objective", PENALTY_VALUE)), "feasible": bool(chosen.get("feasible"))},
                    )
                    state.best_params = chosen.get("params")
                    state.best_row = copy.deepcopy(chosen)
                    try:
                        state.best_seq = int(chosen.get("iter", 0))
                    except Exception:
                        state.best_seq = None
                except Exception:
                    pass
        done_warmup = sum(1 for _ in dirs["done"].glob(f"{run}_warmup_*.json"))
        done_optuna = sum(1 for _ in dirs["done"].glob(f"{run}_optuna_*.json"))
        done_init = sum(1 for _ in dirs["done"].glob(f"{run}_init_*.json"))
        if done_warmup or done_optuna or done_init:
            state.warmup_done = done_warmup
            state.warmup_cursor = min(done_warmup, len(state.warmup_params))
            state.optuna_done = done_optuna
            if done_init:
                state.init_done = True
                state.init_dispatched = True
            if state.warmup_done >= len(state.warmup_params) and (
                state.init_params is None or state.init_done
            ):
                state.phase = "optuna"
            _log_master(
                f"[{run}] resumed queue progress: warmup_done={state.warmup_done} "
                f"optuna_done={state.optuna_done} init_done={state.init_done} "
                f"next_iter={state.next_iter}"
            )
        if tmp_history_path.exists():
            state.header_written = True
        if sanity_tmp_path.exists():
            state.sanity_header_written = True
        return state

    active_run = runs[0] if runs else None
    poll = max(1.0, args.poll_seconds)
    max_retries = max(0, int(args.max_retries))
    task_timeout_seconds = float(args.task_timeout_minutes) * 60.0 if args.task_timeout_minutes else 0.0
    heartbeat_timeout_seconds = float(args.heartbeat_timeout_seconds)

    def _find_in_progress(task_id: str) -> Tuple[Optional[Path], Optional[str]]:
        matches = list(dirs["in_progress"].glob(f"{task_id}__*.json"))
        if not matches:
            return None, None
        path = matches[0]
        parts = path.stem.split("__", 1)
        worker_id = parts[1] if len(parts) > 1 else None
        return path, worker_id

    def _record_result(state: RunState, info: TaskInfo, payload: dict) -> None:
        row_settings = dict(master_settings)
        if state.label_weights:
            row_settings["label_weights"] = state.label_weights
        row = {
            "iter": payload.get("seq", info.seq),
            "objective": payload.get("objective", PENALTY_VALUE),
            "feasible": payload.get("feasible", False),
            "status": payload.get("status", "unknown"),
            "params": payload.get("params", info.params),
            "metrics": payload.get("metrics", {}),
            "best": False,
            "debug_pre": json.dumps(payload.get("debug_pre", ""), default=str)
            if payload.get("debug_pre")
            else "",
            "debug_scale": json.dumps(payload.get("debug_scale", ""), default=str)
            if payload.get("debug_scale")
            else "",
            "debug_params": json.dumps(payload.get("debug_params", ""), default=str)
            if payload.get("debug_params")
            else "",
            "settings": json.dumps(row_settings, default=str),
            "sanity": json.dumps(payload.get("sanity", ""), default=str)
            if payload.get("sanity")
            else "",
        }
        if info.phase == "sanity":
            row["status"] = f"sanity-{row['status']}"
            temp_path = logs_dir / f"tmp_auto_calibration_sanity_{state.run}.csv"
            state.sanity_header_written = _append_temp_csv(
                temp_path, row, state.sanity_header_written
            )
            state.sanity_pending = False
            return

        state.history.append(row)
        temp_path = logs_dir / f"tmp_auto_calibration_{state.run}.csv"
        state.header_written = _append_temp_csv(temp_path, row, state.header_written)

        # Update Optuna
        if info.phase == "optuna":
            if info.trial is not None:
                state.study.tell(info.trial, row["objective"])
            state.optuna_done += 1
        else:
            trial = optuna.trial.create_trial(
                params=info.params,
                distributions=state.distributions,
                value=row["objective"],
            )
            state.study.add_trial(trial)
            if info.phase == "init":
                state.init_done = True
                state.baseline_eval = type(
                    "Eval", (), {"objective": float(row["objective"]), "feasible": row["feasible"]}
                )
            else:
                state.warmup_done += 1

        # Track best
        obj = float(row["objective"])
        new_best = state.best_eval is None or obj < state.best_eval.objective
        if new_best:
            state.best_eval = type("Eval", (), {"objective": obj, "feasible": row["feasible"]})
            state.best_params = row["params"]
            state.best_row = copy.deepcopy(row)
            try:
                state.best_seq = int(row.get("iter", 0))
            except Exception:
                state.best_seq = None
        if row["feasible"]:
            if state.best_feasible is None or obj < state.best_feasible.objective:
                state.best_feasible = type("Eval", (), {"objective": obj, "feasible": True})
                state.best_feasible_params = row["params"]

        sanity_every = int(getattr(args, "sanity_every", 0) or 0)
        status = str(row.get("status", ""))
        if sanity_every > 0 and not status.startswith("eval-error"):
            completed = state.warmup_done + state.optuna_done + (1 if state.init_done else 0)
            last_completed = state.last_sanity_completed or 0
            if completed >= sanity_every and (completed - last_completed) >= sanity_every and not state.sanity_pending:
                best_row = state.best_row or row
                low_result = {
                    "objective": float(best_row.get("objective", PENALTY_VALUE)),
                    "feasible": bool(best_row.get("feasible", False)),
                    "status": str(best_row.get("status", "unknown")),
                }
                low_metrics = best_row.get("metrics")
                if isinstance(low_metrics, dict):
                    low_result["metrics"] = low_metrics
                sanity_spec = {
                    "scale": float(sanity_scale if sanity_scale is not None else 1.0),
                    "dtype": sanity_dtype,
                    "low_result": low_result,
                }
                best_seq = state.best_seq if state.best_seq is not None else int(row.get("iter", 0))
                best_params = state.best_params or row["params"]
                sanity_task_id = _make_task_id(state.run, "sanity", best_seq)
                sanity_payload = _task_payload(
                    sanity_task_id,
                    state.run,
                    "sanity",
                    best_params,
                    best_seq,
                    Path(args.config_dir),
                    args.use_facies,
                    args.per_label,
                    args.use_label_weights,
                    state.label_weights,
                    args.enforce_lower,
                    args.objective_integral,
                    Path(args.bounds_file) if args.bounds_file else None,
                    trial_number=None,
                    attempt=0,
                    quality=None,
                    sanity=sanity_spec,
                )
                _atomic_write_json(dirs["pending"] / f"{sanity_task_id}.json", sanity_payload)
                state.last_sanity_completed = completed
                state.sanity_pending = True

    def _append_orphan_sanity(payload: dict) -> None:
        run = payload.get("run")
        if not run:
            return
        row_settings = dict(master_settings)
        label_weights_payload = payload.get("label_weights")
        if isinstance(label_weights_payload, dict) and label_weights_payload:
            row_settings["label_weights"] = label_weights_payload
        row = {
            "iter": payload.get("seq", 0),
            "objective": payload.get("objective", PENALTY_VALUE),
            "feasible": payload.get("feasible", False),
            "status": f"sanity-{payload.get('status', 'unknown')}",
            "params": payload.get("params", {}),
            "metrics": payload.get("metrics", {}),
            "best": False,
            "debug_pre": json.dumps(payload.get("debug_pre", ""), default=str)
            if payload.get("debug_pre")
            else "",
            "debug_scale": json.dumps(payload.get("debug_scale", ""), default=str)
            if payload.get("debug_scale")
            else "",
            "debug_params": json.dumps(payload.get("debug_params", ""), default=str)
            if payload.get("debug_params")
            else "",
            "settings": json.dumps(row_settings, default=str),
            "sanity": json.dumps(payload.get("sanity", ""), default=str)
            if payload.get("sanity")
            else "",
        }
        temp_path = logs_dir / f"tmp_auto_calibration_sanity_{run}.csv"
        _append_temp_csv(temp_path, row, temp_path.exists())

    def _clear_sanity_pending(run: Optional[str]) -> None:
        if not run:
            return
        state = run_states.get(run)
        if state and state.sanity_pending:
            state.sanity_pending = False
            _log_master(f"[{run}] cleared sanity_pending (orphan sanity result)")

    def _requeue_task(state: RunState, info: TaskInfo, attempt: int, reason: str) -> None:
        task_id = _make_task_id(info.run, info.phase, info.seq)
        payload = _task_payload(
            task_id,
            info.run,
            info.phase,
            info.params,
            info.seq,
            Path(args.config_dir),
            args.use_facies,
            args.per_label,
            args.use_label_weights,
            state.label_weights,
            args.enforce_lower,
            args.objective_integral,
            Path(args.bounds_file) if args.bounds_file else None,
            trial_number=info.trial.number if info.trial is not None else None,
            attempt=attempt,
            quality=quality_spec,
        )
        payload["requeue_reason"] = reason
        _atomic_write_json(dirs["pending"] / f"{task_id}.json", payload)
        in_flight[task_id] = TaskInfo(
            task_id=task_id,
            run=info.run,
            phase=info.phase,
            seq=info.seq,
            params=info.params,
            trial=info.trial,
            attempt=attempt,
        )

    while True:
        _log_master_mem()
        # Process results
        for result_path in list(dirs["results"].glob("*.json")):
            payload = _load_json_retry(result_path)
            if payload is None:
                continue
            task_id = payload.get("task_id")
            if task_id not in in_flight:
                _log_master(
                    f"orphan result {result_path.name} task_id={task_id} (not in in_flight)"
                )
                if payload.get("phase") == "sanity":
                    _append_orphan_sanity(payload)
                    _clear_sanity_pending(payload.get("run"))
                try:
                    os.replace(result_path, orphan_dir / result_path.name)
                except OSError:
                    _safe_unlink(result_path)
                continue
            info = in_flight.pop(task_id)
            stale_counts.pop(task_id, None)
            state = run_states.get(info.run)
            if state is None:
                _log_master(
                    f"orphan result {result_path.name} task_id={task_id} (run_state missing)"
                )
                if payload.get("phase") == "sanity":
                    _append_orphan_sanity(payload)
                    _clear_sanity_pending(payload.get("run"))
                try:
                    os.replace(result_path, orphan_dir / result_path.name)
                except OSError:
                    _safe_unlink(result_path)
                continue

            status = payload.get("status", "unknown")
            attempt = int(payload.get("attempt", info.attempt))
            if status.startswith("eval-error") and attempt < max_retries:
                _requeue_task(state, info, attempt + 1, "eval-error")
                _safe_unlink(result_path)
                continue

            _record_result(state, info, payload)

            _safe_unlink(result_path)

        # Requeue stalled tasks
        if task_timeout_seconds or heartbeat_timeout_seconds:
            now = _now()
            for task_id, info in list(in_flight.items()):
                in_prog_path, worker_id = _find_in_progress(task_id)
                if in_prog_path is None or not worker_id:
                    continue
                hb_path = dirs["heartbeats"] / f"{worker_id}.json"
                last_seen = None
                hb_task_id = None
                hb_interval = None
                hb_payload = None
                if _safe_exists(hb_path):
                    hb_payload = _load_json_retry(hb_path)
                if hb_payload:
                    try:
                        last_seen = float(hb_payload.get("last_seen", 0.0))
                    except Exception:
                        last_seen = None
                    try:
                        hb_interval = float(hb_payload.get("heartbeat_interval_seconds", 0.0)) or None
                    except Exception:
                        hb_interval = None
                if last_seen is not None and last_seen < info.sent_at:
                    last_seen = None
                hb_age = now - last_seen if last_seen else now - info.sent_at
                hb_stale = heartbeat_timeout_seconds and hb_age > heartbeat_timeout_seconds
                runtime = now - info.sent_at
                task_stale = task_timeout_seconds and runtime > task_timeout_seconds
                if not (hb_stale or task_stale):
                    continue

                state = run_states.get(info.run)
                if state is None:
                    in_flight.pop(task_id, None)
                    continue
                attempt = info.attempt
                if hb_stale:
                    reason = "heartbeat-timeout"
                else:
                    reason = "task-timeout"
                stale_counts[task_id] = stale_counts.get(task_id, 0) + 1
                if stale_counts[task_id] < 2:
                    continue
                stale_counts.pop(task_id, None)
                _log_master(
                    f"requeue {task_id} reason={reason} attempt={attempt} "
                    f"hb_age={hb_age:.1f}s runtime={runtime:.1f}s"
                )
                if attempt < max_retries:
                    _requeue_task(state, info, attempt + 1, reason)
                else:
                    fail_payload = {
                        "task_id": task_id,
                        "run": info.run,
                        "phase": info.phase,
                        "seq": info.seq,
                        "params": info.params,
                        "objective": PENALTY_VALUE,
                        "feasible": False,
                        "status": f"eval-error:{reason}",
                        "metrics": {},
                    }
                    _record_result(state, info, fail_payload)
                in_flight.pop(task_id, None)
                # Always remove the stale claim file (for BOTH heartbeat- and
                # task-timeout) so it does not linger in in_progress and inflate
                # the load-balancing counts. The task has already been requeued
                # (or failed) above.
                if in_prog_path is not None and _safe_exists(in_prog_path):
                    _safe_unlink(in_prog_path)
                    _log_master(f"delete in_progress {in_prog_path.name} reason={reason}")
                if attempt >= max_retries:
                    failed_path = dirs["failed"] / f"{task_id}.json"
                    fail_info = {
                        "task_id": task_id,
                        "run": info.run,
                        "phase": info.phase,
                        "seq": info.seq,
                        "attempt": attempt,
                        "reason": reason,
                        "params": info.params,
                        "worker_id": worker_id,
                        "heartbeat_interval_seconds": hb_interval,
                        "heartbeat_age_seconds": hb_age,
                        "runtime_seconds": runtime,
                    }
                    _atomic_write_json(failed_path, fail_info)

        # Reap orphaned in_progress files NOT tracked by this master (e.g. left
        # over from a previous master run, or a claim the current master never
        # dispatched). Only remove when clearly stale (worker heartbeat old or
        # missing AND the claim file itself is old), so a freshly-claimed task on
        # a live worker is never touched.
        if heartbeat_timeout_seconds:
            _reap_now = _now()
            for _ip_path in list(dirs["in_progress"].glob("*__*.json")):
                _stem = _ip_path.stem
                if "__" not in _stem:
                    continue
                _task_id, _wk = _stem.split("__", 1)
                if _task_id in in_flight:
                    continue  # tracked -> handled by the timeout loop above
                _ls = None
                _hb = dirs["heartbeats"] / f"{_wk}.json"
                if _safe_exists(_hb):
                    _hp = _load_json_retry(_hb)
                    if _hp:
                        try:
                            _ls = float(_hp.get("last_seen", 0.0))
                        except Exception:
                            _ls = None
                if _ls is not None and (_reap_now - _ls) <= heartbeat_timeout_seconds:
                    continue  # worker still alive on this task
                try:
                    if (_reap_now - _ip_path.stat().st_mtime) <= heartbeat_timeout_seconds:
                        continue  # claimed too recently to be sure it is orphaned
                except Exception:
                    pass
                _safe_unlink(_ip_path)
                _log_master(f"reaped orphan in_progress {_ip_path.name} (untracked)")

        # Enqueue tasks
        active_runs = _pick_run_order(
            runs,
            args.run_mode,
            active_run,
            run_states,
            max(0, int(getattr(args, "max_active_runs", 0) or 0)),
        )
        for run in active_runs:
            state = run_states.get(run)
            if state is None:
                state = _init_run_state(run)
                run_states[run] = state
            # Move to optuna when warmups done (and init done if used)
            init_ready = (state.init_params is None) or state.init_done
            if (
                state.phase == "warmup"
                and init_ready
                and state.warmup_done >= len(state.warmup_params)
            ):
                state.phase = "optuna"

            # Limit in-flight per run (optionally overridden by control dir).
            in_flight_run = sum(1 for t in in_flight.values() if t.run == run)
            limit_override = _read_inflight_limit(args.control_dir)
            if limit_override is not None:
                limit = limit_override
            else:
                limit = args.max_in_flight_per_run or args.max_in_flight
            if limit and in_flight_run >= limit:
                continue

            if state.phase == "warmup":
                # Dispatch init params first if requested, then wait for result.
                if state.init_params is not None and not state.init_done:
                    if not state.init_dispatched and (not limit or in_flight_run < limit):
                        seq = state.next_iter
                        state.next_iter += 1
                        task_id = _make_task_id(run, "init", seq)
                        payload = _task_payload(
                            task_id,
                            run,
                            "init",
                            state.init_params,
                            seq,
                            Path(args.config_dir),
                            args.use_facies,
                            args.per_label,
                            args.use_label_weights,
                            state.label_weights,
                            args.enforce_lower,
                            args.objective_integral,
                            Path(args.bounds_file) if args.bounds_file else None,
                            quality=quality_spec,
                        )
                        _atomic_write_json(dirs["pending"] / f"{task_id}.json", payload)
                        in_flight[task_id] = TaskInfo(
                            task_id, run, "init", seq, state.init_params, attempt=0
                        )
                        state.init_dispatched = True
                        in_flight_run += 1
                    continue

                while (
                    state.warmup_cursor < len(state.warmup_params)
                    and (not limit or in_flight_run < limit)
                ):
                    params = state.warmup_params[state.warmup_cursor]
                    seq = state.next_iter
                    state.next_iter += 1
                    state.warmup_cursor += 1
                    task_id = _make_task_id(run, "warmup", seq)
                    payload = _task_payload(
                        task_id,
                        run,
                        "warmup",
                        params,
                        seq,
                        Path(args.config_dir),
                        args.use_facies,
                        args.per_label,
                        args.use_label_weights,
                        state.label_weights,
                        args.enforce_lower,
                        args.objective_integral,
                        Path(args.bounds_file) if args.bounds_file else None,
                        quality=quality_spec,
                    )
                    _atomic_write_json(dirs["pending"] / f"{task_id}.json", payload)
                    in_flight[task_id] = TaskInfo(task_id, run, "warmup", seq, params, attempt=0)
                    in_flight_run += 1
            elif state.phase == "optuna":
                while state.optuna_done + in_flight_run < state.max_iters and (
                    not limit or in_flight_run < limit
                ):
                    trial = state.study.ask()
                    params = suggest_params_trial(trial, state.context.param_space)
                    seq = state.next_iter
                    state.next_iter += 1
                    task_id = _make_task_id(run, "optuna", seq)
                    payload = _task_payload(
                        task_id,
                        run,
                        "optuna",
                        params,
                        seq,
                        Path(args.config_dir),
                        args.use_facies,
                        args.per_label,
                        args.use_label_weights,
                        state.label_weights,
                        args.enforce_lower,
                        args.objective_integral,
                        Path(args.bounds_file) if args.bounds_file else None,
                        trial_number=trial.number,
                        quality=quality_spec,
                    )
                    _atomic_write_json(dirs["pending"] / f"{task_id}.json", payload)
                    in_flight[task_id] = TaskInfo(task_id, run, "optuna", seq, params, trial, attempt=0)
                    in_flight_run += 1
                if state.optuna_done >= state.max_iters:
                    state.phase = "done"
                    _cf = state.context.calibration_folder
                    out_folder = (_cf.parent / (_cf.name + "_auto_opt")) if _cf else (logs_dir / f"{state.run}_auto_opt")
                    _finalize_run(
                        state,
                        logs_dir,
                        out_folder,
                        save_calibration=not args.no_save_calibration,
                    )
                    temp_path = logs_dir / f"tmp_auto_calibration_{state.run}.csv"
                    _safe_unlink(temp_path)
                    _release_run_context(state)
                    if args.run_mode == "serial":
                        idx = runs.index(run)
                        active_run = runs[idx + 1] if idx + 1 < len(runs) else None

        if len(run_states) == len(runs) and all(s.phase == "done" for s in run_states.values()):
            break
        time.sleep(poll)


def worker_loop(args: argparse.Namespace) -> None:
    queue = Path(args.queue)
    dirs = _ensure_queue_dirs(queue)
    bounds_file = Path(args.bounds_file) if args.bounds_file else None
    bounds_map = load_bounds_map(bounds_file) if bounds_file else {}
    param_ranges = _parse_param_ranges(args.param_ranges)
    bounds_map = _apply_param_ranges(bounds_map, None, param_ranges)
    label_weights_default = _parse_label_weights(getattr(args, "label_weights", None))
    if args.control_dir:
        os.environ["AUTO_CALIB_CACHE_CONTROL_DIR"] = args.control_dir
    _configure_memmap_env(getattr(args, "memmap_cache", None), getattr(args, "memmap_dir", None))

    from darsia.presets.workflows.rig import Rig

    context_cache: Dict[
        Tuple[
            str,
            str,
            bool,
            bool,
            bool,
            Optional[Tuple[Tuple[int, float], ...]],
            bool,
            Optional[str],
            float,
            Optional[str],
            str,
        ],
        Any,
    ] = {}

    def _normalize_label_weights(raw: Any) -> Dict[int, float]:
        weights: Dict[int, float] = {}
        if isinstance(raw, dict):
            items = raw.items()
        else:
            return {}
        for key, val in items:
            try:
                label = int(key)
                weight = float(val)
            except Exception:
                continue
            weights[label] = weight
        return weights

    def _get_context(payload: dict, quality_override: Optional[Dict[str, Any]] = None):
        run = payload["run"]
        config_dir = payload.get("config_dir") or args.config_dir
        use_facies = bool(payload.get("use_facies", args.use_facies))
        per_label = bool(payload.get("per_label_params", args.per_label))
        use_label_weights = bool(payload.get("use_label_weights", args.use_label_weights))
        raw_label_weights = payload.get("label_weights")
        if not isinstance(raw_label_weights, dict):
            raw_label_weights = label_weights_default
        label_weights = _normalize_label_weights(raw_label_weights)
        enforce_lower = bool(payload.get("enforce_lower", args.enforce_lower))
        bounds_file_path = payload.get("bounds_file") or (str(bounds_file) if bounds_file else None)
        objective_integral = str(
            payload.get("objective_integral", getattr(args, "objective_integral", "off")) or "off"
        )
        static_light_correction = str(
            payload.get("static_light_correction")
            or os.environ.get("FFAC_STATIC_LIGHT_CORRECTION", "")
            or "off"
        )
        quality = quality_override or payload.get("quality")
        scale = 1.0
        dtype = None
        if isinstance(quality, dict):
            try:
                scale = float(quality.get("scale", 1.0))
            except Exception:
                scale = 1.0
            dtype = quality.get("dtype")
        label_weights_key = tuple(sorted(label_weights.items())) if label_weights else None
        cache_key = (
            run,
            str(config_dir),
            use_facies,
            per_label,
            use_label_weights,
            label_weights_key,
            enforce_lower,
            bounds_file_path,
            scale,
            dtype,
            objective_integral,
            static_light_correction,
        )
        if cache_key in context_cache:
            return context_cache[cache_key]
        ctx = build_context(
            run=run,
            config_dir=Path(config_dir),
            rig_cls=Rig,
            ref_config_path=None,
            use_facies=use_facies,
            bounds_map=bounds_map,
            enforce_lower=enforce_lower,
            per_label_params=per_label,
            use_label_weights=use_label_weights,
            label_weights=label_weights,
            quality_scale=scale,
            quality_dtype=dtype,
            objective_integral=objective_integral,
            static_light_correction=static_light_correction,
        )
        context_cache[cache_key] = ctx
        return ctx

    worker_hostname = _hostname()
    worker_id = args.worker_id or f"{worker_hostname}_{os.getpid()}"
    worker_index = _parse_worker_index(worker_id)
    poll = max(0.5, args.poll_seconds)
    proc = psutil.Process(os.getpid()) if psutil else None
    stickiness_wait = max(0.0, float(getattr(args, "stickiness_wait_seconds", 0.0) or 0.0))

    thread_limit = getattr(args, "thread_limit", None)
    _set_thread_env(thread_limit)

    log_dir = Path(args.worker_log_dir) if args.worker_log_dir else dirs["worker_logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{worker_id}.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    thread_note = f" threads={thread_limit}" if thread_limit else ""
    print(f"[{worker_id}] started host={worker_hostname}{thread_note}")
    memmap_mode = os.getenv("DARSIA_MEMMAP_MODE", "off")
    memmap_dir = os.getenv("DARSIA_MEMMAP_DIR", "")
    if memmap_mode and memmap_mode != "off":
        memmap_note = f"[{worker_id}] memmap mode={memmap_mode}"
        if memmap_dir:
            memmap_note += f" dir={memmap_dir}"
        print(memmap_note)
    worker_state = _read_worker_state(log_dir, worker_id)
    last_run = worker_state.get("last_run") if worker_state else None
    max_tasks_per_worker = max(0, int(getattr(args, "max_tasks_per_worker", 0) or 0))
    tasks_done = 0

    heartbeat_dir = dirs["heartbeats"]
    heartbeat_path = heartbeat_dir / f"{worker_id}.json"
    heartbeat_interval = max(1.0, float(args.heartbeat_seconds))
    sanity_lock_stale = max(heartbeat_interval * 2.0, 120.0)
    task_lock = threading.Lock()
    current_task: Dict[str, Any] = {
        "task_id": None,
        "run": None,
        "started_at": None,
        "max_rss_mb": None,
        "max_vms_mb": None,
    }
    stop_event = threading.Event()
    last_heartbeat_error = 0.0

    def _memory_payload() -> Dict[str, Any]:
        if proc is None:
            return {}
        try:
            mem = proc.memory_info()
            return {
                "mem_rss_mb": round(mem.rss / (1024 * 1024), 2),
                "mem_vms_mb": round(mem.vms / (1024 * 1024), 2),
            }
        except Exception:
            return {"mem_error": True}

    def _heartbeat_loop() -> None:
        nonlocal last_heartbeat_error
        while not stop_event.is_set():
            mem_snapshot = _memory_payload()
            with task_lock:
                task_id = current_task.get("task_id")
                run = current_task.get("run")
                started_at = current_task.get("started_at")
                if task_id and mem_snapshot:
                    rss = mem_snapshot.get("mem_rss_mb")
                    vms = mem_snapshot.get("mem_vms_mb")
                    if isinstance(rss, (int, float)):
                        prev = current_task.get("max_rss_mb")
                        if prev is None or rss > prev:
                            current_task["max_rss_mb"] = rss
                    if isinstance(vms, (int, float)):
                        prev = current_task.get("max_vms_mb")
                        if prev is None or vms > prev:
                            current_task["max_vms_mb"] = vms
            now = _now()
            payload = {
                "worker_id": worker_id,
                "hostname": worker_hostname,
                "task_id": task_id,
                "run": run,
                "last_seen": now,
                "pid": os.getpid(),
                "heartbeat_interval_seconds": heartbeat_interval,
            }
            if started_at is not None:
                payload["task_started_at"] = started_at
                payload["task_age_seconds"] = now - float(started_at)
            payload.update(mem_snapshot)
            ok = _safe_write_json(heartbeat_path, payload)
            if not ok:
                if now - last_heartbeat_error > 60:
                    print(f"[{worker_id}] heartbeat write failed; will retry")
                    last_heartbeat_error = now
            time.sleep(heartbeat_interval)

    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    def _shutdown_worker(reason: str) -> None:
        stop_event.set()
        try:
            heartbeat_thread.join(timeout=heartbeat_interval + 1.0)
        except Exception:
            pass
        try:
            if heartbeat_path.exists():
                heartbeat_path.unlink()
        except Exception:
            pass
        print(f"[{worker_id}] exiting reason={reason}")

    def _find_worker_in_progress() -> Optional[Path]:
        matches = list(dirs["in_progress"].glob(f"*__{worker_id}.json"))
        if not matches:
            return None
        if len(matches) > 1:
            print(f"[{worker_id}] multiple in_progress found ({len(matches)}); resuming newest")
        return max(matches, key=lambda p: p.stat().st_mtime)

    def _count_pending_for_run(run: str) -> int:
        count = 0
        for path in dirs["pending"].glob(f"{run}_*.json"):
            if _task_run_from_name(path.name) == run:
                count += 1
        return count

    def _count_in_progress_for_run(run: str) -> int:
        count = 0
        for path in dirs["in_progress"].glob(f"{run}_*__*.json"):
            if _task_run_from_name(path.name) == run:
                count += 1
        return count

    while True:
        limit = _read_worker_limit(args.control_dir, worker_hostname)
        if limit is not None and worker_index is not None and worker_index >= limit:
            print(f"[{worker_id}] limit={limit} -> exiting")
            _shutdown_worker("limit")
            return
        claimed = _find_worker_in_progress()
        if claimed is None:
            task_path = None
            preferred_run = last_run
            sanity_lock_active = _sanity_lock_active(args.control_dir, worker_hostname, sanity_lock_stale)
            if last_run:
                pending_for_last = _count_pending_for_run(last_run)
                in_progress_for_last = _count_in_progress_for_run(last_run)
                if pending_for_last == 0 and stickiness_wait > 0:
                    waited = 0.0
                    while waited < stickiness_wait:
                        time.sleep(min(poll, stickiness_wait - waited))
                        waited += min(poll, stickiness_wait - waited)
                        pending_for_last = _count_pending_for_run(last_run)
                        if pending_for_last > 0:
                            break
                if pending_for_last == 0:
                    preferred_run = None
            for _ in range(MAX_CLAIM_ATTEMPTS):
                task_path = _select_pending_task(
                    dirs,
                    worker_id,
                    preferred_run=preferred_run,
                    allow_sanity=not sanity_lock_active,
                )
                if not task_path:
                    break
                claimed = dirs["in_progress"] / f"{task_path.stem}__{worker_id}.json"
                try:
                    os.replace(task_path, claimed)
                    break
                except OSError as exc:
                    print(f"[{worker_id}] claim failed for {task_path.name}: {exc}")
                    claimed = None
                    time.sleep(random.uniform(0.0, CLAIM_JITTER_MAX_SECONDS))
                    continue
            if claimed is None:
                time.sleep(poll)
                if args.once:
                    _shutdown_worker("once")
                    return
                continue
        payload = _load_json_retry(claimed, attempts=8, delay=0.1, jitter=READ_JITTER_MAX_SECONDS)
        if payload is None:
            print(f"[{worker_id}] claim read failed for {claimed.name}; returning to pending")
            base_name = claimed.stem.split("__", 1)[0]
            pending_path = dirs["pending"] / f"{base_name}.json"
            if claimed.exists():
                try:
                    os.replace(claimed, pending_path)
                except OSError:
                    pass
            continue
        sanity_lock_acquired = False
        if payload.get("phase") == "sanity":
            if not _acquire_sanity_lock(args.control_dir, worker_hostname, worker_id, sanity_lock_stale):
                base_name = claimed.stem.split("__", 1)[0]
                pending_path = dirs["pending"] / f"{base_name}.json"
                if claimed.exists():
                    try:
                        os.replace(claimed, pending_path)
                    except OSError:
                        pass
                time.sleep(poll)
                continue
            sanity_lock_acquired = True
        run_name = payload.get("run")
        if last_run and run_name and run_name != last_run:
            print(f"[{worker_id}] run switch {last_run}->{run_name}; restarting before switch")
            base_name = claimed.stem.split("__", 1)[0]
            pending_path = dirs["pending"] / f"{base_name}.json"
            if claimed.exists():
                try:
                    os.replace(claimed, pending_path)
                except OSError:
                    pass
            _write_worker_state(log_dir, worker_id, {"last_run": None})
            _shutdown_worker("run-switch")
            return
        with task_lock:
            current_task["task_id"] = payload.get("task_id")
            current_task["run"] = payload.get("run")
            current_task["started_at"] = _now()
        last_run = run_name or last_run
        _write_worker_state(log_dir, worker_id, {"last_run": last_run})
        params = payload.get("params", {})
        attempt = payload.get("attempt", 0)
        try:
            seq_value = int(payload.get("seq", -1))
        except Exception:
            seq_value = -1
        include_debug = seq_value == 0
        task_id = payload.get("task_id")
        mem_snapshot = _memory_payload()
        if mem_snapshot:
            with task_lock:
                current_task["max_rss_mb"] = mem_snapshot.get("mem_rss_mb")
                current_task["max_vms_mb"] = mem_snapshot.get("mem_vms_mb")
        if mem_snapshot:
            mem_info = f" rss={mem_snapshot.get('mem_rss_mb')}MB vms={mem_snapshot.get('mem_vms_mb')}MB"
        else:
            mem_info = ""
        cache_size = _read_cache_control(args.control_dir, worker_hostname)
        watchdog_state = _read_watchdog_state(args.control_dir, worker_hostname)
        desired_workers = watchdog_state.get("desired_workers")
        workers_running = watchdog_state.get("workers_running")
        thread_state = watchdog_state.get("thread_limit")
        cache_note = f" cache={cache_size}" if cache_size is not None else ""
        workers_note = ""
        if desired_workers is not None or workers_running is not None:
            workers_note = f" workers={workers_running}/{desired_workers}"
        thread_state_note = f" threads={thread_state}" if thread_state else ""
        cache_stats = _memmap_stats(run_name)
        cache_total_mb = cache_stats.get("cache_bytes_total")
        cache_run_mb = cache_stats.get("cache_bytes_run")
        cache_stats_note = ""
        if isinstance(cache_total_mb, (int, float)):
            cache_stats_note += f" memmap_total_mb={cache_total_mb / (1024 * 1024):.0f}"
        if isinstance(cache_run_mb, (int, float)):
            cache_stats_note += f" memmap_run_mb={cache_run_mb / (1024 * 1024):.0f}"
        print(
            f"[{worker_id}] start task={task_id} run={payload.get('run')} "
            f"phase={payload.get('phase')} seq={seq_value} attempt={attempt}{mem_info}"
            f"{cache_note}{workers_note}{thread_state_note}{cache_stats_note}"
        )

        max_rss = None
        max_vms = None
        try:
            task_start = _now()
            ctx = _get_context(payload)
            from auto_calibrate_color_to_mass import evaluate_run  # noqa: E402

            sanity_spec = payload.get("sanity")
            sanity_result = None
            if isinstance(sanity_spec, dict):
                full_result = evaluate_run(ctx, params)
                low_payload = sanity_spec.get("low_result")
                if isinstance(low_payload, dict):
                    try:
                        low_obj = float(low_payload.get("objective", "nan"))
                    except Exception:
                        low_obj = float("nan")
                    low_feasible = bool(low_payload.get("feasible", False))
                    low_status = str(low_payload.get("status", "unknown"))
                    low_metrics = low_payload.get("metrics")
                else:
                    low_ctx = _get_context(payload, quality_override=sanity_spec)
                    low_result = evaluate_run(low_ctx, params)
                    low_obj = low_result.objective
                    low_feasible = low_result.feasible
                    low_status = low_result.status
                    low_metrics = {
                        k: {
                            "injected_full": v.injected_full,
                            "total_full": v.total_full,
                        }
                        for k, v in low_result.metrics.items()
                    }
                eval_result = full_result
                full_metrics = {
                    k: {
                        "injected_full": v.injected_full,
                        "total_full": v.total_full,
                    }
                    for k, v in full_result.metrics.items()
                }
                metrics_delta = None
                if isinstance(low_metrics, dict):
                    metrics_delta = {}
                    for key, full_vals in full_metrics.items():
                        low_vals = low_metrics.get(key) if isinstance(low_metrics, dict) else None
                        if not isinstance(full_vals, dict) or not isinstance(low_vals, dict):
                            continue
                        try:
                            delta_inj = float(low_vals.get("injected_full", 0.0)) - float(full_vals.get("injected_full", 0.0))
                            delta_tot = float(low_vals.get("total_full", 0.0)) - float(full_vals.get("total_full", 0.0))
                        except Exception:
                            continue
                        metrics_delta[key] = {
                            "delta_injected_full": delta_inj,
                            "delta_total_full": delta_tot,
                        }
                sanity_result = {
                    "full_scale": 1.0,
                    "full_dtype": None,
                    "low_scale": sanity_spec.get("scale"),
                    "low_dtype": sanity_spec.get("dtype"),
                    "scale": sanity_spec.get("scale"),
                    "dtype": sanity_spec.get("dtype"),
                    "objective_full": full_result.objective,
                    "feasible_full": full_result.feasible,
                    "status_full": full_result.status,
                    "objective_low": low_obj,
                    "feasible_low": low_feasible,
                    "status_low": low_status,
                    "delta_objective": low_obj - full_result.objective,
                    "metrics_full": full_metrics,
                    "metrics_low": low_metrics if isinstance(low_metrics, dict) else None,
                    "metrics_delta": metrics_delta,
                }
            else:
                eval_result = evaluate_run(ctx, params)
            debug_pre = None
            debug_scale = None
            debug_params = None
            if include_debug:
                debug_pre = _summarize_debug(getattr(ctx, "_debug_info_pre", None))
                debug_scale = _summarize_debug(getattr(ctx, "_debug_info_scale", None))
                debug_params = _summarize_debug(getattr(ctx, "_debug_info_params", None))
                if debug_pre is not None:
                    setattr(ctx, "_debug_info_pre", None)
                if debug_scale is not None:
                    setattr(ctx, "_debug_info_scale", None)
                if debug_params is not None:
                    setattr(ctx, "_debug_info_params", None)
            with task_lock:
                max_rss = current_task.get("max_rss_mb")
                max_vms = current_task.get("max_vms_mb")
            result_payload = {
                "task_id": payload.get("task_id"),
                "run": payload.get("run"),
                "phase": payload.get("phase"),
                "seq": payload.get("seq"),
                "params": params,
                "objective": eval_result.objective,
                "feasible": eval_result.feasible,
                "status": eval_result.status,
                "metrics": {
                    k: {
                        "injected_full": v.injected_full,
                        "total_full": v.total_full,
                    }
                    for k, v in eval_result.metrics.items()
                },
                "debug_pre": debug_pre,
                "debug_scale": debug_scale,
                "debug_params": debug_params,
                "attempt": attempt,
                "worker_id": worker_id,
                "hostname": worker_hostname,
                "max_rss_mb": max_rss,
                "max_vms_mb": max_vms,
                "runtime_seconds": _now() - task_start,
            }
            if sanity_result is not None:
                result_payload["sanity"] = sanity_result
            result_payload.update(_memory_payload())
            if cache_stats:
                result_payload.update(cache_stats)
            peak_note = ""
            if max_rss is not None or max_vms is not None:
                peak_note = f" max_rss={max_rss}MB max_vms={max_vms}MB"
            print(
                f"[{worker_id}] done task={task_id} status={eval_result.status} "
                f"runtime={result_payload.get('runtime_seconds'):.2f}s{peak_note}"
            )
        except Exception as exc:
            trace = traceback.format_exc()
            print(f"[{worker_id}] task error: {exc}\n{trace}")
            with task_lock:
                max_rss = current_task.get("max_rss_mb")
                max_vms = current_task.get("max_vms_mb")
            result_payload = {
                "task_id": payload.get("task_id"),
                "run": payload.get("run"),
                "phase": payload.get("phase"),
                "seq": payload.get("seq"),
                "params": params,
                "objective": PENALTY_VALUE,
                "feasible": False,
                "status": f"eval-error:{exc}",
                "metrics": {},
                "debug_pre": None,
                "debug_scale": None,
                "debug_params": None,
                "attempt": attempt,
                "worker_id": worker_id,
                "hostname": worker_hostname,
                "max_rss_mb": max_rss,
                "max_vms_mb": max_vms,
                "runtime_seconds": _now() - task_start if "task_start" in locals() else None,
                "error_traceback": trace,
            }
            result_payload.update(_memory_payload())
            if cache_stats:
                result_payload.update(cache_stats)
            runtime = result_payload.get("runtime_seconds")
            runtime_str = f"{runtime:.2f}s" if isinstance(runtime, (int, float)) else "n/a"
            peak_note = ""
            if max_rss is not None or max_vms is not None:
                peak_note = f" max_rss={max_rss}MB max_vms={max_vms}MB"
            print(f"[{worker_id}] failed task={task_id} runtime={runtime_str}{peak_note}")

        result_path = dirs["results"] / f"{payload.get('task_id')}.json"
        if not _safe_write_json(result_path, result_payload, attempts=5, delay=0.2):
            print(f"[{worker_id}] result write failed for {task_id}")
            continue
        task_completed = True
        done_path = dirs["done"] / f"{payload.get('task_id')}__{worker_id}.json"
        try:
            os.replace(claimed, done_path)
        except FileNotFoundError:
            print(f"[{worker_id}] done move skipped; in_progress missing for {task_id}")
        except OSError as exc:
            print(f"[{worker_id}] done move failed for {task_id}: {exc}")
        finally:
            with task_lock:
                current_task["task_id"] = None
                current_task["run"] = None
                current_task["started_at"] = None
            if sanity_lock_acquired:
                _release_sanity_lock(args.control_dir, worker_hostname, worker_id)
        if max_tasks_per_worker > 0 and task_completed:
            tasks_done += 1
            if tasks_done >= max_tasks_per_worker:
                print(f"[{worker_id}] max_tasks_per_worker={max_tasks_per_worker} reached; exiting")
                _shutdown_worker("max-tasks")
                return


def watchdog_main(args: argparse.Namespace) -> None:
    import multiprocessing as mp

    workers = max(1, args.workers)
    hostname = _hostname()
    prefix = args.worker_id_prefix or args.worker_id or hostname
    queue = Path(args.queue)
    dirs = _ensure_queue_dirs(queue)
    # Seed an editable worker-count control file so the count can be changed live:
    # lower the number -> excess workers finish their current task then exit;
    # raise it -> the watchdog spawns more. Only seeds if absent (never clobbers).
    if args.control_dir:
        try:
            _ctrl_base = Path(args.control_dir); _ctrl_base.mkdir(parents=True, exist_ok=True)
            _ctrl_file = _ctrl_base / f"{hostname}.txt"
            if not _ctrl_file.exists():
                _ctrl_file.write_text(str(workers), encoding="utf-8")
        except Exception:
            pass
    try:
        _cmd_dir = queue / "_commands"
        _cmd_dir.mkdir(parents=True, exist_ok=True)
        (_cmd_dir / "watchdog_{}_{}.txt".format(hostname, os.getpid())).write_text(
            " ".join(sys.argv), encoding="utf-8")
    except Exception:
        pass
    log_dir = Path(args.worker_log_dir) if args.worker_log_dir else dirs["worker_logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"watchdog_{prefix}.log"
    current_thread_limit = args.thread_limit
    worker_stall_seconds = max(0.0, float(getattr(args, "worker_stall_seconds", 0) or 0))

    def _spawn(worker_id: str) -> mp.Process:
        worker_args = copy.copy(args)
        worker_args.worker_id = worker_id
        worker_args.thread_limit = current_thread_limit
        proc = mp.Process(target=worker_loop, args=(worker_args,), daemon=True)
        proc.start()
        return proc

    slots: Dict[int, Dict[str, Any]] = {}

    def _log_event(message: str) -> None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}Z] {message}"
        print(line, flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _ensure_worker(idx: int) -> None:
        if idx in slots and slots[idx]["proc"].is_alive():
            return
        worker_id = f"{prefix}_{idx}"
        proc = _spawn(worker_id)
        slots[idx] = {
            "worker_id": worker_id,
            "proc": proc,
            "last_start": _now(),
            "reported_dead": False,
            "thread_limit": current_thread_limit,
        }

    def _desired_workers() -> Tuple[int, Optional[int]]:
        limit = _read_worker_limit(args.control_dir, hostname)
        if limit is None:
            return workers, limit
        return max(0, int(limit)), limit

    current_desired, current_limit = _desired_workers()
    for idx in range(current_desired):
        _ensure_worker(idx)
        if args.stagger_seconds:
            time.sleep(args.stagger_seconds)
    _log_event(f"watchdog started workers={current_desired} limit={current_limit}")
    try:
        while True:
            now = _now()
            desired, new_limit = _desired_workers()
            if desired != current_desired:
                _log_event(f"worker limit change {current_desired}->{desired} (limit={new_limit})")
                current_desired = desired

            alive = 0
            # Spawn up to desired
            for idx in range(desired):
                if idx not in slots:
                    _ensure_worker(idx)
                    if args.stagger_seconds:
                        time.sleep(args.stagger_seconds)
                else:
                    proc = slots[idx]["proc"]
                    if proc.is_alive():
                        # Kill + respawn a HUNG worker: alive process but its
                        # heartbeat has gone stale beyond --worker-stall-seconds
                        # (and it has been running long enough that a missing/old
                        # heartbeat is not just a slow startup). Normal tasks are
                        # seconds long, so a generous threshold avoids false kills.
                        if worker_stall_seconds:
                            _hb = dirs["heartbeats"] / f"{slots[idx]['worker_id']}.json"
                            _ls = None
                            if _safe_exists(_hb):
                                _hp = _load_json_retry(_hb)
                                if _hp:
                                    try:
                                        _ls = float(_hp.get("last_seen", 0.0))
                                    except Exception:
                                        _ls = None
                            _start_age = now - float(slots[idx].get("last_start", now))
                            if (_ls is not None and (now - _ls) > worker_stall_seconds
                                    and _start_age > worker_stall_seconds):
                                _log_event(
                                    f"worker {slots[idx]['worker_id']} stalled "
                                    f"(hb_age={now - _ls:.0f}s) -> terminating for respawn"
                                )
                                try:
                                    proc.terminate()
                                except Exception:
                                    pass
                                slots[idx]["reported_dead"] = False
                                continue  # respawned next iteration once not alive
                        alive += 1
                        slots[idx]["reported_dead"] = False
                        continue
                    if not slots[idx].get("reported_dead"):
                        exit_code = proc.exitcode
                        hb_note = ""
                        hb_path = dirs["heartbeats"] / f"{slots[idx]['worker_id']}.json"
                        hb_payload = _load_json_retry(hb_path) if _safe_exists(hb_path) else None
                        if hb_payload:
                            last_seen = hb_payload.get("last_seen")
                            task_id = hb_payload.get("task_id")
                            rss = hb_payload.get("mem_rss_mb")
                            vms = hb_payload.get("mem_vms_mb")
                            hb_note = (
                                f" last_seen={last_seen} task_id={task_id} "
                                f"rss_mb={rss} vms_mb={vms}"
                            )
                        _log_event(
                            f"worker {slots[idx]['worker_id']} exited with code {exit_code}{hb_note}"
                        )
                        slots[idx]["reported_dead"] = True
                    if args.once:
                        continue
                    if now - float(slots[idx]["last_start"]) < float(args.restart_delay_seconds):
                        continue
                    slots[idx]["proc"] = _spawn(slots[idx]["worker_id"])
                    slots[idx]["last_start"] = now
                    slots[idx]["reported_dead"] = False
                    slots[idx]["thread_limit"] = current_thread_limit
                    alive += 1
            # Do not respawn workers above desired; allow them to finish and exit.
            for idx in sorted(list(slots.keys())):
                if idx < desired:
                    continue
                proc = slots[idx]["proc"]
                if not proc.is_alive():
                    slots.pop(idx, None)
            cache_size = _read_cache_control(args.control_dir, hostname)
            _write_watchdog_state(
                args.control_dir,
                hostname,
                current_desired,
                alive,
                current_thread_limit,
                cache_size,
            )
            if args.once and alive == 0:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        for slot in slots.values():
            try:
                slot["proc"].terminate()
            except Exception:
                continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed auto-calibration queue.")
    sub = parser.add_subparsers(dest="command", required=True)

    master = sub.add_parser("master")
    master.add_argument("--queue", required=True, help="Queue directory (shared).")
    master.add_argument(
        "--no-clear-queue", action="store_true",
        help="Do NOT wipe the queue on startup. By DEFAULT the master clears the queue's "
             "task folders for a clean run (removes leftover/orphan tasks from a previous run).",
    )
    master.add_argument("--runs", nargs="+", required=True)
    master.add_argument("--config-dir", default="config/run_ac")
    master.add_argument("--logs-dir", default=_default_calibration_log_root())
    master.add_argument(
        "--exact-logs-dir",
        action="store_true",
        help="Use --logs-dir exactly instead of creating a timestamped run subfolder.",
    )
    master.add_argument("--ref-config", default=None)
    master.add_argument(
        "--use-facies",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Enable facies (true/false).",
    )
    master.add_argument(
        "--per-label",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Use per-label parameters (true/false).",
    )
    master.add_argument(
        "--use-label-weights",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Enable label weighting (true/false).",
    )
    master.add_argument(
        "--auto-label-weights",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Derive label weights automatically when none are provided (true/false).",
    )
    master.add_argument(
        "--label-weights",
        default=None,
        help="Per-label weights: '7=1.0;8=0.5'. Requires --use-label-weights.",
    )
    master.add_argument(
        "--label-weight-grouping",
        default="none",
        choices=["none", "facies"],
        help="Normalize label weights by group (e.g. facies) when use_facies=false.",
    )
    master.add_argument("--enforce-lower", action="store_true")
    master.add_argument(
        "--objective-integral",
        default="off",
        help="Objective mode: off (point-wise), l1, l2, or drift[:LAMBDA] "
             "(point-wise + LAMBDA * total-variation of detected mass over the "
             "post-injection plateau; mass-conservation penalty, default LAMBDA=1.0).",
    )
    master.add_argument(
        "--no-save-calibration",
        action="store_true",
        help="Do not write optimised signal models back to the experiment results folder; "
             "write only logs/final_full_scale and best_params.json. Useful for parallel "
             "diagnostic sweeps sharing the same run/results cache.",
    )
    master.add_argument(
        "--optuna-persist",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Persist Optuna study to sqlite (true/false).",
    )
    master.add_argument(
        "--optuna-storage-dir",
        default=None,
        help="Directory to store Optuna sqlite DBs (defaults to --logs-dir).",
    )
    master.add_argument(
        "--use-last-best",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Seed from last best (true/false).",
    )
    master.add_argument(
        "--use-history",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Seed Optuna with all valid trials from history CSV (true/false).",
    )
    master.add_argument("--bounds-file", default=None)
    master.add_argument("--max-iters", type=int, default=40)
    master.add_argument("--warmup-iters", type=int, default=100)
    master.add_argument("--warmup-levels", default=None)
    master.add_argument(
        "--param-ranges",
        default=None,
        help="Per-value bounds overrides. Format: 'value2=0,1;value6=0,2'.",
    )
    master.add_argument(
        "--param-levels",
        default=None,
        help="Per-value warmup level counts. Format: '8' or 'value2=8;value6=6'.",
    )
    master.add_argument("--warmup-high", type=float, default=None)
    master.add_argument(
        "--warmup-mode",
        choices=["prefix", "suffix", "single"],
        default="prefix",
        help="Warmup sweep mode: prefix, suffix, or single (legacy).",
    )
    master.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup evaluations (useful when --use-last-best and data unchanged).",
    )
    master.add_argument("--run-mode", choices=["serial", "parallel"], default="serial")
    master.add_argument("--max-in-flight", type=int, default=0)
    master.add_argument("--max-in-flight-per-run", type=int, default=0)
    master.add_argument(
        "--max-active-runs",
        type=int,
        default=0,
        help="With --run-mode parallel, limit how many runs are active/kept in memory at once (0=all).",
    )
    master.add_argument(
        "--control-dir",
        default=None,
        help="Optional directory with dynamic limits (max_in_flight_per_run.txt).",
    )
    master.add_argument("--poll-seconds", type=float, default=2.0)
    master.add_argument("--max-retries", type=int, default=3)
    master.add_argument("--task-timeout-minutes", type=float, default=45)
    master.add_argument("--heartbeat-timeout-seconds", type=float, default=600)
    master.add_argument(
        "--master-mem-log-seconds",
        type=float,
        default=60.0,
        help="How often to log master/system memory (seconds).",
    )
    master.add_argument(
        "--sanity-every",
        type=int,
        default=0,
        help="After a new best is found, enqueue sanity if >=N tasks since last sanity (0=disabled).",
    )
    master.add_argument(
        "--sanity-scale",
        type=float,
        default=None,
        help="Downscale factor for sanity comparisons (defaults to --quality-scale).",
    )
    master.add_argument(
        "--sanity-dtype",
        default=None,
        help="Numeric dtype for sanity comparisons (defaults to --quality-dtype).",
    )
    master.add_argument(
        "--memmap-cache",
        choices=["off", "images", "arrays", "all"],
        default=None,
        help="Enable shared memmap cache (off/images/arrays/all).",
    )
    master.add_argument(
        "--memmap-dir",
        default=None,
        help="Directory for shared memmap cache (host-local recommended).",
    )
    master.add_argument(
        "--quality-scale",
        type=float,
        default=1.0,
        help="Downscale factor for all evaluations (1.0 = full resolution).",
    )
    master.add_argument(
        "--quality-dtype",
        default=None,
        help="Numeric dtype override for evaluations (float32/float64).",
    )

    worker = sub.add_parser("worker")
    worker.add_argument("--queue", required=True, help="Queue directory (shared).")
    worker.add_argument("--config-dir", default="config/run_ac")
    worker.add_argument(
        "--use-facies",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Enable facies (true/false).",
    )
    worker.add_argument(
        "--per-label",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Use per-label parameters (true/false).",
    )
    worker.add_argument(
        "--use-label-weights",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Enable label weighting (true/false).",
    )
    worker.add_argument(
        "--objective-integral",
        default="off",
        help="Objective mode: off (point-wise), l1, l2, or drift[:LAMBDA] "
             "(point-wise + LAMBDA * total-variation of detected mass over the "
             "post-injection plateau; mass-conservation penalty, default LAMBDA=1.0).",
    )
    worker.add_argument(
        "--auto-label-weights",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Ignored by worker (accepted for CLI consistency).",
    )
    worker.add_argument(
        "--label-weights",
        default=None,
        help="Per-label weights (must match master when used).",
    )
    worker.add_argument("--enforce-lower", action="store_true")
    worker.add_argument("--bounds-file", default=None)
    worker.add_argument(
        "--param-ranges",
        default=None,
        help="Per-value bounds overrides (must match master).",
    )
    worker.add_argument(
        "--param-levels",
        default=None,
        help="Ignored by worker (accepted for CLI consistency).",
    )
    worker.add_argument(
        "--control-dir",
        default=None,
        help="Optional directory with per-host worker limits (HOST.txt).",
    )
    worker.add_argument("--threads-per-worker", dest="thread_limit", type=int, default=None)
    worker.add_argument("--poll-seconds", type=float, default=1.0)
    worker.add_argument(
        "--stickiness-wait-seconds",
        type=float,
        default=20.0,
        help="Wait briefly for new tasks for the last run before switching runs.",
    )
    worker.add_argument("--worker-id", default=None)
    worker.add_argument("--heartbeat-seconds", type=float, default=60.0)
    worker.add_argument("--worker-log-dir", default=None)
    worker.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=0,
        help="Restart worker after N tasks (0=disabled).",
    )
    worker.add_argument(
        "--memmap-cache",
        choices=["off", "images", "arrays", "all"],
        default=None,
        help="Enable shared memmap cache (off/images/arrays/all).",
    )
    worker.add_argument(
        "--memmap-dir",
        default=None,
        help="Directory for shared memmap cache (host-local recommended).",
    )
    worker.add_argument("--once", action="store_true")

    watchdog = sub.add_parser("watchdog")
    watchdog.add_argument("--queue", required=True, help="Queue directory (shared).")
    watchdog.add_argument("--config-dir", default="config/run_ac")
    watchdog.add_argument(
        "--use-facies",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Enable facies (true/false).",
    )
    watchdog.add_argument(
        "--per-label",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Use per-label parameters (true/false).",
    )
    watchdog.add_argument(
        "--use-label-weights",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Enable label weighting (true/false).",
    )
    watchdog.add_argument(
        "--auto-label-weights",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Ignored by watchdog (accepted for CLI consistency).",
    )
    watchdog.add_argument(
        "--label-weights",
        default=None,
        help="Per-label weights (must match master when used).",
    )
    watchdog.add_argument("--enforce-lower", action="store_true")
    watchdog.add_argument("--bounds-file", default=None)
    watchdog.add_argument(
        "--param-ranges",
        default=None,
        help="Per-value bounds overrides (must match master).",
    )
    watchdog.add_argument(
        "--param-levels",
        default=None,
        help="Ignored by watchdog (accepted for CLI consistency).",
    )
    watchdog.add_argument(
        "--control-dir",
        default=None,
        help="Optional directory with per-host worker limits (HOST.txt).",
    )
    watchdog.add_argument("--poll-seconds", type=float, default=1.0)
    watchdog.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Default worker count if no per-host limit file is set.",
    )
    watchdog.add_argument(
        "--stickiness-wait-seconds",
        type=float,
        default=20.0,
        help="Wait briefly for new tasks for the last run before switching runs.",
    )
    watchdog.add_argument("--worker-id", default=None)
    watchdog.add_argument("--once", action="store_true")
    watchdog.add_argument("--stagger-seconds", type=float, default=0.0)
    watchdog.add_argument("--heartbeat-seconds", type=float, default=60.0)
    watchdog.add_argument("--worker-log-dir", default=None)
    watchdog.add_argument("--worker-id-prefix", default=None)
    watchdog.add_argument("--restart-delay-seconds", type=float, default=5.0)
    watchdog.add_argument("--threads-per-worker", dest="thread_limit", type=int, default=None)
    watchdog.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=0,
        help="Restart worker after N tasks (0=disabled).",
    )
    watchdog.add_argument(
        "--worker-stall-seconds",
        type=float,
        default=600.0,
        help="Terminate+respawn an ALIVE worker whose heartbeat is older than this "
             "(hung worker). Normal tasks are seconds long; 0=disabled.",
    )
    watchdog.add_argument(
        "--memmap-cache",
        choices=["off", "images", "arrays", "all"],
        default=None,
        help="Enable shared memmap cache (off/images/arrays/all).",
    )
    watchdog.add_argument(
        "--memmap-dir",
        default=None,
        help="Directory for shared memmap cache (host-local recommended).",
    )

    requeue = sub.add_parser("requeue-failed")
    requeue.add_argument("--queue", required=True, help="Queue directory (shared).")
    requeue.add_argument("--runs", nargs="*", default=None)
    requeue.add_argument("--config-dir", default="config/run_ac")
    requeue.add_argument("--logs-dir", default=None)
    requeue.add_argument("--bounds-file", default=None)
    requeue.add_argument("--use-facies", nargs="?", const=True, default=False, type=_parse_bool)
    requeue.add_argument("--per-label", nargs="?", const=True, default=False, type=_parse_bool)
    requeue.add_argument("--use-label-weights", nargs="?", const=True, default=False, type=_parse_bool)
    requeue.add_argument("--label-weights", default=None)
    requeue.add_argument("--enforce-lower", action="store_true")
    requeue.add_argument(
        "--objective-integral",
        default="off",
        help="Objective mode: off (point-wise), l1, l2, or drift[:LAMBDA] "
             "(point-wise + LAMBDA * total-variation of detected mass over the "
             "post-injection plateau; mass-conservation penalty, default LAMBDA=1.0).",
    )
    requeue.add_argument("--quality-scale", type=float, default=1.0)
    requeue.add_argument("--quality-dtype", default=None)
    requeue.add_argument("--keep-failed", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "master":
        master_main(args)
    elif args.command == "worker":
        worker_loop(args)
    elif args.command == "watchdog":
        watchdog_main(args)
    elif args.command == "requeue-failed":
        requeue_failed_main(args)


if __name__ == "__main__":
    main()
