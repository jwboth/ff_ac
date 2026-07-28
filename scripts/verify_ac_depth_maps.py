"""Verify that AC runs use the shared non-constant AC14 depth map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_MEASUREMENTS_SHA256 = (
    "fd650d91366ac743f64ef781736cc5b0ebd5461ca36bd02af1eac34789f5977a"
)
EXPECTED_DEPTH_ARRAY_SHA256 = (
    "375be487d0bb598964404432a386316a82496afbc3477aaa3fcf7b81c98fcd21"
)


def _sha256_bytes(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def verify_depth_maps(
    runs: list[str],
    *,
    results_root: Path,
    measurements_path: Path,
    expected_measurements_sha256: str = EXPECTED_MEASUREMENTS_SHA256,
    expected_depth_sha256: str = EXPECTED_DEPTH_ARRAY_SHA256,
) -> dict:
    measurement_sha = hashlib.sha256(measurements_path.read_bytes()).hexdigest()
    if measurement_sha.lower() != expected_measurements_sha256.lower():
        raise RuntimeError(
            f"Depth-measurement SHA mismatch: {measurement_sha} != "
            f"{expected_measurements_sha256}"
        )

    rows: list[dict] = []
    for run in runs:
        path = results_root / run / "setup" / "depth" / "depth_map.npz"
        if not path.exists():
            raise FileNotFoundError(f"{run}: missing depth map {path}")
        with np.load(path, allow_pickle=True) as payload:
            array = np.asarray(payload["array"])
        finite = np.asarray(array[np.isfinite(array)], dtype=np.float64)
        if finite.size == 0:
            raise RuntimeError(f"{run}: depth map has no finite values")
        array_sha = _sha256_bytes(array)
        if array_sha.lower() != expected_depth_sha256.lower():
            raise RuntimeError(
                f"{run}: depth-map SHA mismatch: {array_sha} != "
                f"{expected_depth_sha256}"
            )
        span = float(np.ptp(finite))
        if span <= 1e-6:
            raise RuntimeError(f"{run}: depth map is effectively constant")
        rows.append(
            {
                "run": run,
                "path": str(path),
                "shape": list(array.shape),
                "sha256": array_sha,
                "minimum_m": float(np.min(finite)),
                "maximum_m": float(np.max(finite)),
                "mean_m": float(np.mean(finite)),
                "std_m": float(np.std(finite)),
            }
        )

    return {
        "measurements_path": str(measurements_path),
        "measurements_sha256": measurement_sha,
        "expected_depth_sha256": expected_depth_sha256,
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(r"Z:\Albus\Results"),
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/depth_measurements.csv"),
    )
    parser.add_argument(
        "--expected-measurements-sha256",
        default=EXPECTED_MEASUREMENTS_SHA256,
    )
    parser.add_argument(
        "--expected-depth-sha256",
        default=EXPECTED_DEPTH_ARRAY_SHA256,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify_depth_maps(
        [str(run).lower() for run in args.runs],
        results_root=args.results_root,
        measurements_path=args.measurements,
        expected_measurements_sha256=args.expected_measurements_sha256,
        expected_depth_sha256=args.expected_depth_sha256,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in report["runs"]:
        print(
            f"{row['run']}: depth {row['minimum_m']:.6f}.."
            f"{row['maximum_m']:.6f} m std={row['std_m']:.6f} "
            f"sha256={row['sha256']}"
        )


if __name__ == "__main__":
    main()
