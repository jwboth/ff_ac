"""Prepare baseline color-to-mass calibrations for AC runs.

This runs the same seed steps used before distributed auto-calibration:
rig setup with color correction, color embedding calibration, and default
mass calibration. Existing seeded runs are skipped by default.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _default_log_root() -> Path:
    z_root = Path("Z:/Albus/Autokalibrering_log")
    if z_root.exists():
        return z_root / "production_titration_l1_prep"
    return Path("logs") / "production_titration_l1_prep"


def _physical_runs(results_dir: Path) -> list[str]:
    runs = [
        p.name.lower()
        for p in results_dir.iterdir()
        if p.is_dir() and p.name.lower().startswith("ac") and p.name.lower()[2:].isdigit()
    ]
    return sorted(runs, key=lambda s: int(s[2:]))


def _seed_marker(results_dir: Path, run: str) -> Path:
    return (
        results_dir
        / run
        / "calibration"
        / "color"
        / "relative_colorpath"
        / "color_to_mass"
        / "from_facies"
        / "color_path_interpretation"
        / "color_path_interpretation_0.json"
    )


def _run_step(cmd: list[str], cwd: Path, log_file) -> int:
    log_file.write(f"\n>>> {' '.join(cmd)}\n")
    log_file.flush()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.write(f"<<< exit={proc.returncode}\n")
    log_file.flush()
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="Z:/Albus/Results")
    parser.add_argument("--config-dir", default="config_seg6/run_ac")
    parser.add_argument("--common", default="config_seg6/common.toml")
    parser.add_argument("--coloron", default="config_seg6/coloron.toml")
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    results_dir = Path(args.results_dir)
    config_dir = Path(args.config_dir)
    common = Path(args.common)
    coloron = Path(args.coloron)
    logs_dir = Path(args.logs_dir) if args.logs_dir else _default_log_root()
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs = args.runs if args.runs else _physical_runs(results_dir)
    status_path = logs_dir / "seed_status.csv"
    fieldnames = ["timestamp", "run", "status", "step", "returncode", "marker"]
    write_header = not status_path.exists()

    with status_path.open("a", newline="", encoding="utf-8") as status_file:
        writer = csv.DictWriter(status_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for run in runs:
            marker = _seed_marker(results_dir, run)
            if marker.exists() and not args.force:
                writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "run": run,
                        "status": "skipped",
                        "step": "exists",
                        "returncode": 0,
                        "marker": str(marker),
                    }
                )
                status_file.flush()
                continue

            config = config_dir / f"{run}.toml"
            if not config.exists():
                writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "run": run,
                        "status": "failed",
                        "step": "config",
                        "returncode": 2,
                        "marker": str(marker),
                    }
                )
                status_file.flush()
                continue

            run_log_path = logs_dir / f"{run}.log"
            status = "ok"
            failed_step = ""
            returncode = 0
            with run_log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} {run} =====\n")
                steps = [
                    (
                        "setup-rig",
                        [
                            sys.executable,
                            "scripts/setup.py",
                            "--config",
                            str(common),
                            str(config),
                            str(coloron),
                            "--rig",
                        ],
                    ),
                    (
                        "color-embedding",
                        [
                            sys.executable,
                            "scripts/calibration.py",
                            "--config",
                            str(common),
                            str(config),
                            str(coloron),
                            "--color-embedding",
                        ],
                    ),
                    (
                        "default-mass",
                        [
                            sys.executable,
                            "scripts/calibration.py",
                            "--config",
                            str(common),
                            str(config),
                            str(coloron),
                            "--default-mass",
                            "--reset",
                        ],
                    ),
                ]
                for step_name, cmd in steps:
                    returncode = _run_step(cmd, repo, log_file)
                    if returncode:
                        status = "failed"
                        failed_step = step_name
                        break

            if status == "ok" and not marker.exists():
                status = "failed"
                failed_step = "marker"
                returncode = 3

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run": run,
                    "status": status,
                    "step": failed_step,
                    "returncode": returncode,
                    "marker": str(marker),
                }
            )
            status_file.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
