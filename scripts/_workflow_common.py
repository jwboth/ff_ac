"""Helpers shared by the three distributed ff_ac workflow CLIs.

Keeps argument parsing and worker-command execution in one place so the three
``distributed_*_queue.py`` scripts stay thin and consistent.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

import queue_core as qc

REPO_ROOT = Path(__file__).resolve().parent.parent


def add_common_queue_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--queue", required=True, help="shared queue directory")
    p.add_argument("--tasks-per-file", type=int, default=40)


def add_watchdog_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--queue", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--control-dir", default=None)
    p.add_argument("--heartbeat-seconds", type=float, default=60.0)
    p.add_argument("--watchdog-interval-seconds", type=float, default=10.0)
    p.add_argument("--restart-delay-seconds", type=float, default=5.0)
    p.add_argument("--task-timeout-minutes", type=float, default=30.0)
    p.add_argument("--max-tasks-per-worker", type=int, default=50)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--stop-when-drained", action="store_true")


def add_worker_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--queue", required=True)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--worker-id", default=None)
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument("--heartbeat-seconds", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-tasks", type=int, default=50)
    p.add_argument("--exit-when-idle", action="store_true")
    p.add_argument("--idle-seconds", type=float, default=120.0)


def worker_config_from_args(a: argparse.Namespace) -> qc.WorkerConfig:
    wid = a.worker_id or f"{qc.host_id()}-w{a.worker_index}-{uuid.uuid4().hex[:4]}"
    return qc.WorkerConfig(
        worker_id=wid, heartbeat_seconds=a.heartbeat_seconds,
        poll_interval=a.poll_interval, max_retries=a.max_retries,
        max_tasks=a.max_tasks, exit_when_idle=a.exit_when_idle,
        idle_seconds=a.idle_seconds)


def build_worker_cmd(script: str, queue: str, extra=None) -> list:
    """argv for launching one worker of *script* (used by the watchdog)."""
    cmd = [sys.executable, str(script), "worker", "--queue", str(queue)]
    if extra:
        cmd += extra
    return cmd


def watchdog_config_from_args(a: argparse.Namespace) -> qc.WatchdogConfig:
    return qc.WatchdogConfig(
        workers=a.workers, control_dir=a.control_dir,
        heartbeat_seconds=a.heartbeat_seconds,
        watchdog_interval_seconds=a.watchdog_interval_seconds,
        restart_delay_seconds=a.restart_delay_seconds,
        task_timeout_minutes=a.task_timeout_minutes,
        max_tasks_per_worker=a.max_tasks_per_worker,
        max_retries=a.max_retries,
    )


def run_subprocess_task(cmd_template: str, mapping: dict) -> dict:
    """Run a worker command built from ``cmd_template`` and ``mapping``.

    ``cmd_template`` is a shell-like string with ``{placeholders}`` filled from
    ``mapping``. Returns a result dict with returncode and tail of output. This
    is how a worker delegates the actual science to ff_ac's own CLIs
    (``scripts/analysis.py`` etc.) without importing DarSIA in the queue layer.
    """
    cmd = cmd_template.format(**mapping)
    proc = subprocess.run(shlex.split(cmd), cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    res = {"cmd": cmd, "returncode": proc.returncode, "tail": out[-500:],
           "stdout": (proc.stdout or "")[-2000:]}
    if proc.returncode != 0:
        raise RuntimeError(f"worker command failed ({proc.returncode}): {out[-300:]}")
    return res
