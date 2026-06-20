"""Launch AC production calibration after seed preparation completes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROLLOUT_RUNS = ["ac22", "ac26", "ac27", "ac31", "ac42", "ac48", "ac50", "ac51", "ac53", "ac58"]


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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return str(pid) in completed.stdout
    except Exception:
        return False


def _wait_for_pids(pids: list[int], log_file) -> None:
    if not pids:
        return
    log_file.write(f"Waiting for prep PIDs: {pids}\n")
    log_file.flush()
    while any(_pid_alive(pid) for pid in pids):
        time.sleep(30)
    log_file.write("Prep PIDs have exited.\n")
    log_file.flush()


def _validate_seed(results_dir: Path, runs: list[str]) -> list[str]:
    return [run for run in runs if not _seed_marker(results_dir, run).exists()]


def _latest_seed_failures(status_path: Path) -> dict[str, str]:
    if not status_path.exists():
        return {}
    latest: dict[str, dict[str, str]] = {}
    with status_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            run = str(row.get("run", ""))
            if run:
                latest[run] = row
    return {
        run: row.get("step", "") or row.get("status", "")
        for run, row in latest.items()
        if row.get("status") == "failed"
    }


def _start_process(cmd: list[str], cwd: Path, stdout: Path, stderr: Path, env: dict[str, str]) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    out = stdout.open("a", encoding="utf-8", errors="replace")
    err = stderr.open("a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=err, env=env)
    return int(proc.pid)


def _run_blocking(cmd: list[str], cwd: Path, stdout: Path, stderr: Path, env: dict[str, str]) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("a", encoding="utf-8", errors="replace") as out, stderr.open(
        "a", encoding="utf-8", errors="replace"
    ) as err:
        out.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} =====\n")
        out.write(" ".join(cmd) + "\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, env=env, text=True)
        out.write(f"exit={proc.returncode}\n")
        return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-pids", nargs="*", type=int, default=[])
    parser.add_argument("--results-dir", default="Z:/Albus/Results")
    parser.add_argument("--prep-logs-dir", default="Z:/Albus/Autokalibrering_log/production_titration_l1_prep")
    parser.add_argument("--prod-logs-root", default="Z:/Albus/Autokalibrering_log/production_titration_l1")
    parser.add_argument("--rollout-logs-dir", default="Z:/Albus/Autokalibrering_log/rollout_titration_l1/facies1_perlabel1_warmup150_optuna800_parallel_20260617_0127")
    parser.add_argument("--queue", default=r"\\Moderskipet\Darsia_Queue\Kalibrering_AC_production_titration_l1")
    parser.add_argument("--finalize-queue", default=r"\\Moderskipet\Darsia_Queue\Kalibrering_AC_rollout_titration_l1_finalize")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-iters", type=int, default=800)
    parser.add_argument("--warmup-iters", type=int, default=150)
    args = parser.parse_args()

    repo = Path.cwd()
    results_dir = Path(args.results_dir)
    prep_logs = Path(args.prep_logs_dir)
    prod_logs = Path(args.prod_logs_root)
    prod_logs.mkdir(parents=True, exist_ok=True)
    orchestrator_log = prod_logs / "orchestrator.log"
    env = os.environ.copy()
    env["FFAC_TITRATION_FLASH"] = "on"

    with orchestrator_log.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} launch watcher =====\n")
        _wait_for_pids(args.prep_pids, log)

        physical = _physical_runs(results_dir)
        production_runs = [run for run in physical if run not in set(ROLLOUT_RUNS)]
        failures = _latest_seed_failures(prep_logs / "seed_status.csv")
        missing = _validate_seed(results_dir, production_runs)
        log.write(f"physical={physical}\n")
        log.write(f"production_runs={production_runs}\n")
        log.write(f"seed_failures={failures}\n")
        log.write(f"missing_seed={missing}\n")
        log.flush()
        if failures or missing:
            log.write("Aborting production launch because seed prep is incomplete.\n")
            return 2

        finalize_cmd = [
            sys.executable,
            "scripts/distributed_auto_calibration_queue.py",
            "master",
            "--queue",
            args.finalize_queue,
            "--runs",
            *ROLLOUT_RUNS,
            "--config-dir",
            "config_seg6/run_ac",
            "--logs-dir",
            args.rollout_logs_dir,
            "--exact-logs-dir",
            "--use-history",
            "true",
            "--skip-warmup",
            "--use-facies",
            "true",
            "--per-label",
            "true",
            "--objective-integral",
            "off",
            "--bounds-file",
            "config/bounds_seg6_coupled.json",
            "--max-iters",
            "0",
            "--warmup-iters",
            str(args.warmup_iters),
            "--run-mode",
            "parallel",
            "--max-in-flight-per-run",
            "0",
            "--quality-dtype",
            "float32",
        ]
        finalize_rc = _run_blocking(
            finalize_cmd,
            repo,
            prod_logs / "finalize_rollout_stdout.log",
            prod_logs / "finalize_rollout_stderr.log",
            env,
        )
        log.write(f"finalize_rollout_rc={finalize_rc}\n")
        log.flush()
        if finalize_rc:
            return finalize_rc

        master_cmd = [
            sys.executable,
            "scripts/distributed_auto_calibration_queue.py",
            "master",
            "--queue",
            args.queue,
            "--runs",
            *production_runs,
            "--config-dir",
            "config_seg6/run_ac",
            "--logs-dir",
            str(prod_logs),
            "--use-facies",
            "true",
            "--per-label",
            "true",
            "--objective-integral",
            "off",
            "--bounds-file",
            "config/bounds_seg6_coupled.json",
            "--max-iters",
            str(args.max_iters),
            "--warmup-iters",
            str(args.warmup_iters),
            "--run-mode",
            "parallel",
            "--max-in-flight-per-run",
            "3",
            "--control-dir",
            str(Path(args.queue) / "control"),
            "--sanity-every",
            "100",
            "--sanity-scale",
            "1.00",
            "--quality-dtype",
            "float32",
        ]
        watchdog_cmd = [
            sys.executable,
            "scripts/distributed_auto_calibration_queue.py",
            "watchdog",
            "--queue",
            args.queue,
            "--config-dir",
            "config_seg6/run_ac",
            "--use-facies",
            "true",
            "--per-label",
            "true",
            "--control-dir",
            str(Path(args.queue) / "control"),
            "--workers",
            str(args.workers),
            "--worker-stall-seconds",
            "600",
            "--max-tasks-per-worker",
            "80",
        ]
        master_pid = _start_process(
            master_cmd,
            repo,
            prod_logs / "launcher_master_stdout.log",
            prod_logs / "launcher_master_stderr.log",
            env,
        )
        watchdog_pid = _start_process(
            watchdog_cmd,
            repo,
            prod_logs / "launcher_watchdog_stdout.log",
            prod_logs / "launcher_watchdog_stderr.log",
            env,
        )
        manifest = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "queue": args.queue,
            "production_runs": production_runs,
            "master_pid": master_pid,
            "watchdog_pid": watchdog_pid,
            "master_cmd": master_cmd,
            "watchdog_cmd": watchdog_cmd,
        }
        (prod_logs / "launcher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.write(json.dumps(manifest, indent=2) + "\n")
        log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
