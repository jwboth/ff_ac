"""Rebuild and verify AC rig inputs used by mass calibration.

The mass objective is only comparable across runs when every rig uses the same
measured depth field. This utility rebuilds depth + rig sequentially, follows
the per-run colour-correction stamp, and writes a machine-readable manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .ac_production_campaign import FINAL_PRODUCTION_RUNS
except ImportError:
    from ac_production_campaign import FINAL_PRODUCTION_RUNS


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = Path(
    r"Z:\Albus\Autokalibrering_log\input_standardization_20260727"
)
AC61_REST_PROTOCOL = REPO / "protocols" / "ac61" / "imaging_protocol_rest.csv"
EXPECTED_AC61_FIRST_REST = "2023-07-11 13:23:10"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    return path.read_text(encoding="utf-8").lstrip("\ufeff").strip().lower()


def _config_paths(run: str) -> tuple[list[Path], str]:
    config_dir = REPO / "config_seg6" / "run_ac"
    paths = [
        REPO / "config_seg6" / "common.toml",
        config_dir / f"{run}.toml",
    ]
    color_state = _read_state(config_dir / ".color_state" / f"{run}.txt")
    if color_state == "on":
        paths.append(REPO / "config_seg6" / "coloron.toml")
    elif color_state != "off":
        raise RuntimeError(
            f"{run}: invalid or missing colour state {color_state!r}; "
            "expected config_seg6/run_ac/.color_state/<run>.txt = on|off"
        )
    return paths, color_state


def _run_step(
    args: list[str],
    *,
    log_path: Path,
    label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {label} {_utc_now()} ===\n")
        stream.flush()
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}; see {log_path}"
        )


def _depth_record(run: str) -> dict[str, Any]:
    path = Path(r"Z:\Albus\Results") / run / "setup" / "rig" / "depth.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as payload:
        array = np.asarray(payload["array"])
    finite = array[np.isfinite(array)]
    if finite.size != array.size:
        raise ValueError(f"{run}: depth field contains non-finite values")
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()
    return {
        "run": run,
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "minimum_m": float(np.min(finite)),
        "maximum_m": float(np.max(finite)),
        "mean_m": float(np.mean(finite)),
        "std_m": float(np.std(finite)),
        "sha256_array": digest,
    }


def _verify_ac61_protocol() -> None:
    if not AC61_REST_PROTOCOL.exists():
        raise FileNotFoundError(AC61_REST_PROTOCOL)
    with AC61_REST_PROTOCOL.open(encoding="utf-8") as stream:
        next(stream, None)
        first = next(stream, "").rstrip("\r\n").split(",")
    if len(first) < 3 or first[2] != EXPECTED_AC61_FIRST_REST:
        actual = first[2] if len(first) >= 3 else "<missing>"
        raise RuntimeError(
            "AC61 resting protocol has not been corrected by +1 day: "
            f"expected {EXPECTED_AC61_FIRST_REST}, found {actual}"
        )


def _refresh_ac61_calibration(config_paths: list[Path], log_path: Path) -> None:
    config_args = [str(path) for path in config_paths]
    _run_step(
        [
            "scripts/calibration.py",
            "--config",
            *config_args,
            "--color-embedding",
        ],
        log_path=log_path,
        label="ac61 color embedding after protocol correction",
    )
    _run_step(
        [
            "scripts/calibration.py",
            "--config",
            *config_args,
            "--default-mass",
            "--reset",
        ],
        log_path=log_path,
        label="ac61 default mass seed after protocol correction",
    )


def standardize(
    runs: list[str],
    *,
    log_root: Path,
    verify_only: bool,
    refresh_ac61: bool,
) -> Path:
    _verify_ac61_protocol()
    log_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, run in enumerate(runs, start=1):
        config_paths, color_state = _config_paths(run)
        log_path = log_root / "logs" / f"{run}.log"
        print(
            f"[{index}/{len(runs)}] {run}: "
            f"color={color_state}, {'verify' if verify_only else 'depth+rig'}",
            flush=True,
        )
        if not verify_only:
            _run_step(
                [
                    "scripts/setup.py",
                    "--config",
                    *[str(path) for path in config_paths],
                    "--depth",
                    "--rig",
                ],
                log_path=log_path,
                label=f"{run} measured depth + rig",
            )
            if run == "ac61" and refresh_ac61:
                _refresh_ac61_calibration(config_paths, log_path)
        row = _depth_record(run)
        row["color_state"] = color_state
        rows.append(row)

    hashes = {row["sha256_array"] for row in rows}
    shapes = {tuple(row["shape"]) for row in rows}
    std_values = [float(row["std_m"]) for row in rows]
    consistent = len(hashes) == 1 and len(shapes) == 1
    measured = bool(std_values) and min(std_values) > 1e-6
    manifest = {
        "created_at_utc": _utc_now(),
        "status": "ok" if consistent and measured else "invalid",
        "runs": runs,
        "run_count": len(runs),
        "depth_source": str(REPO / "data" / "depth_measurements.csv"),
        "same_depth_hash": consistent,
        "measured_nonconstant_depth": measured,
        "ac61_protocol_first_rest": EXPECTED_AC61_FIRST_REST,
        "records": rows,
    }
    manifest_path = log_root / "depth_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if not consistent:
        raise RuntimeError(
            f"Depth verification failed: {len(hashes)} hashes and "
            f"{len(shapes)} shapes across {len(rows)} runs; see {manifest_path}"
        )
    if not measured:
        raise RuntimeError(
            f"Depth verification found a constant field; see {manifest_path}"
        )
    print(
        f"Verified one measured depth field for {len(rows)} runs. "
        f"Manifest: {manifest_path}",
        flush=True,
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=FINAL_PRODUCTION_RUNS,
        help="Runs to rebuild; defaults to the 40 final-production runs.",
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not rebuild; only verify the existing rig depth caches.",
    )
    parser.add_argument(
        "--no-refresh-ac61",
        action="store_true",
        help="Do not regenerate AC61 colour embedding/default mass after its protocol fix.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = [str(run).strip().lower() for run in args.runs]
    if len(set(runs)) != len(runs):
        raise SystemExit("Duplicate run ids are not allowed.")
    standardize(
        runs,
        log_root=args.log_root,
        verify_only=bool(args.verify_only),
        refresh_ac61=not bool(args.no_refresh_ac61),
    )


if __name__ == "__main__":
    main()
