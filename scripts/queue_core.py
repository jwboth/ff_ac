"""Shared distributed-queue core for ff_ac workflows.

API-agnostic orchestration layer ported (and simplified) from the ff_um
``distributed_*_queue.py`` scripts. The three ff_ac workflow CLIs all build on
top of it, so the queue logic lives in exactly one place.

A *queue* is a shared directory (typically an SMB share) with subfolders::

    pending/ in_progress/ completed/ failed/ heartbeats/ worker_logs/ control/

Tasks are grouped into *batches* (``tasks_per_file``) to avoid huge numbers of
tiny JSON files on SMB. Claiming a batch is an atomic ``os.replace`` from
``pending/`` to ``in_progress/`` - the first worker to win the rename owns it.
The module has no DarSIA/ffum imports; a worker is given a plain callable
``execute(task) -> dict`` and this core handles leasing, heartbeats, retries,
timeouts and result collection.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

LOG = logging.getLogger("queue_core")

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"
HEARTBEATS = "heartbeats"
WORKER_LOGS = "worker_logs"
CONTROL = "control"
SUBDIRS = (PENDING, IN_PROGRESS, COMPLETED, FAILED, HEARTBEATS, WORKER_LOGS, CONTROL)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def host_id() -> str:
    return platform.node() or socket.gethostname() or "host"


class Queue:
    """A directory-backed work queue."""

    def __init__(self, root):
        self.root = Path(root)

    def dir(self, name: str) -> Path:
        return self.root / name

    def ensure_layout(self) -> None:
        for name in SUBDIRS:
            self.dir(name).mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        import shutil
        for name in (PENDING, IN_PROGRESS, COMPLETED, FAILED):
            d = self.dir(name)
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    def enqueue(self, tasks: Iterable[dict], tasks_per_file: int = 40) -> int:
        self.ensure_layout()
        batch: list = []
        n_batches = 0
        for task in tasks:
            task.setdefault("id", uuid.uuid4().hex)
            task.setdefault("attempts", 0)
            batch.append(task)
            if len(batch) >= tasks_per_file:
                self._write_batch(batch)
                n_batches += 1
                batch = []
        if batch:
            self._write_batch(batch)
            n_batches += 1
        LOG.info("Enqueued %d batch(es) into %s", n_batches, self.dir(PENDING))
        return n_batches

    def _write_batch(self, tasks: list, name: Optional[str] = None) -> Path:
        name = name or f"batch_{int(time.time()*1000):013d}_{uuid.uuid4().hex[:8]}.json"
        tmp = self.dir(PENDING) / (name + ".tmp")
        final = self.dir(PENDING) / name
        tmp.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, final)
        return final

    def claim(self, worker_id: str) -> Optional["Lease"]:
        pending = self.dir(PENDING)
        if not pending.exists():
            return None
        for entry in sorted(pending.iterdir()):
            if entry.suffix != ".json":
                continue
            dest = self.dir(IN_PROGRESS) / f"{entry.stem}__{worker_id}.json"
            try:
                os.replace(entry, dest)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            try:
                data = json.loads(dest.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, FileNotFoundError):
                continue
            return Lease(self, dest, worker_id, data.get("tasks", []))
        return None

    def counts(self) -> dict:
        out = {}
        for name in (PENDING, IN_PROGRESS, COMPLETED, FAILED):
            d = self.dir(name)
            out[name] = sum(1 for p in d.glob("*.json")) if d.exists() else 0
        return out

    def is_drained(self) -> bool:
        c = self.counts()
        return c[PENDING] == 0 and c[IN_PROGRESS] == 0

    def requeue_stale(self, timeout_minutes: float) -> int:
        cutoff = time.time() - timeout_minutes * 60
        moved = 0
        ip = self.dir(IN_PROGRESS)
        if not ip.exists():
            return 0
        for entry in ip.glob("*.json"):
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
            except FileNotFoundError:
                continue
            stem = entry.stem.split("__", 1)[0]
            dest = self.dir(PENDING) / f"{stem}.json"
            try:
                os.replace(entry, dest)
                moved += 1
            except OSError:
                pass
        if moved:
            LOG.warning("Requeued %d stale batch(es)", moved)
        return moved


@dataclass
class Lease:
    queue: Queue
    path: Path
    worker_id: str
    tasks: list
    results: list = field(default_factory=list)

    def touch(self) -> None:
        try:
            os.utime(self.path, None)
        except FileNotFoundError:
            pass

    def complete(self, results: list) -> None:
        payload = {"tasks": self.tasks, "results": results,
                   "worker": self.worker_id, "finished": _utcnow()}
        dest = self.queue.dir(COMPLETED) / self.path.name
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest)
        self._remove_inprogress()

    def fail(self, error: str, max_retries: int) -> None:
        for t in self.tasks:
            t["attempts"] = int(t.get("attempts", 0)) + 1
        if all(t["attempts"] > max_retries for t in self.tasks):
            payload = {"tasks": self.tasks, "error": error, "worker": self.worker_id,
                       "failed": _utcnow()}
            dest = self.queue.dir(FAILED) / self.path.name
            dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._remove_inprogress()
            LOG.error("Batch %s permanently failed: %s", self.path.name, error)
        else:
            stem = self.path.stem.split("__", 1)[0]
            self.queue._write_batch(self.tasks, name=f"{stem}.json")
            self._remove_inprogress()
            LOG.warning("Batch %s requeued after error: %s", self.path.name, error)

    def _remove_inprogress(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def read_worker_limit(control_dir, default: int) -> int:
    if not control_dir:
        return default
    p = Path(control_dir) / f"{host_id()}.txt"
    try:
        return max(0, int(p.read_text().strip()))
    except (FileNotFoundError, ValueError):
        return default


def write_worker_limit(control_dir, count: int) -> None:
    d = Path(control_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{host_id()}.txt").write_text(str(int(count)), encoding="utf-8")


def write_heartbeat(queue: Queue, worker_id: str, status: str = "alive",
                    extra: Optional[dict] = None) -> None:
    payload = {"worker": worker_id, "host": host_id(), "status": status, "time": _utcnow()}
    if extra:
        payload.update(extra)
    p = queue.dir(HEARTBEATS) / f"{worker_id}.json"
    try:
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


@dataclass
class WorkerConfig:
    worker_id: str
    heartbeat_seconds: float = 60.0
    poll_interval: float = 5.0
    max_retries: int = 3
    max_tasks: Optional[int] = None
    exit_when_idle: bool = False
    idle_seconds: float = 120.0


def run_worker(queue: Queue, execute: Callable[[dict], dict], cfg: WorkerConfig) -> None:
    queue.ensure_layout()
    done = 0
    last_work = time.time()
    LOG.info("Worker %s starting", cfg.worker_id)
    while True:
        if cfg.max_tasks is not None and done >= cfg.max_tasks:
            LOG.info("Worker %s reached max_tasks=%d, exiting", cfg.worker_id, cfg.max_tasks)
            return
        lease = queue.claim(cfg.worker_id)
        if lease is None:
            write_heartbeat(queue, cfg.worker_id, "idle")
            if cfg.exit_when_idle and (time.time() - last_work) > cfg.idle_seconds:
                LOG.info("Worker %s idle, exiting", cfg.worker_id)
                return
            time.sleep(cfg.poll_interval)
            continue
        last_work = time.time()
        results: list = []
        last_hb = 0.0
        try:
            for task in lease.tasks:
                res = execute(task)
                if not isinstance(res, dict):
                    res = {"value": res}
                res.setdefault("id", task.get("id"))
                results.append(res)
                done += 1
                now = time.time()
                if now - last_hb >= cfg.heartbeat_seconds:
                    lease.touch()
                    write_heartbeat(queue, cfg.worker_id, "busy",
                                    {"done": done, "batch": lease.path.name})
                    last_hb = now
            lease.complete(results)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Worker %s failed batch %s", cfg.worker_id, lease.path.name)
            lease.fail(repr(exc), cfg.max_retries)


@dataclass
class WatchdogConfig:
    workers: int = 1
    control_dir: Optional[str] = None
    heartbeat_seconds: float = 60.0
    watchdog_interval_seconds: float = 10.0
    restart_delay_seconds: float = 5.0
    task_timeout_minutes: float = 30.0
    max_tasks_per_worker: Optional[int] = 50
    max_retries: int = 3


def run_watchdog(queue: Queue, worker_cmd: list, cfg: WatchdogConfig,
                 stop_when_drained: bool = False) -> None:
    """Supervise a pool of worker *subprocesses* on this host.

    ``worker_cmd`` is the argv that launches a single worker - typically
    ``[python, this_script, "worker", "--queue", <queue>, ...]``. Launching real
    subprocesses (not multiprocessing with a closure) matches the ff_um design
    and avoids pickling problems on ``spawn`` platforms such as Windows.
    """
    import subprocess

    queue.ensure_layout()
    procs: dict = {}
    next_idx = 0

    def spawn():
        nonlocal next_idx
        cmd = list(worker_cmd) + ["--worker-index", str(next_idx)]
        procs[next_idx] = subprocess.Popen(cmd)
        next_idx += 1

    try:
        while True:
            target = read_worker_limit(cfg.control_dir, cfg.workers)
            for idx, p in list(procs.items()):
                if p.poll() is not None:
                    procs.pop(idx, None)
            while len(procs) < target:
                spawn()
            queue.requeue_stale(cfg.task_timeout_minutes)
            if stop_when_drained and queue.is_drained() and not procs:
                LOG.info("Queue drained, watchdog exiting")
                return
            c = queue.counts()
            LOG.info("watchdog: workers=%d/%d pending=%d in_progress=%d completed=%d failed=%d",
                     len(procs), target, c[PENDING], c[IN_PROGRESS], c[COMPLETED], c[FAILED])
            time.sleep(cfg.watchdog_interval_seconds)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()


def iter_results(queue: Queue) -> Iterator[dict]:
    d = queue.dir(COMPLETED)
    if not d.exists():
        return
    for batch in sorted(d.glob("*.json")):
        try:
            data = json.loads(batch.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        for r in data.get("results", []):
            yield r
