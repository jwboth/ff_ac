"""Distributed auto-calibration queue for ff_ac.

Ports the ff_um ``distributed_auto_calibration_queue.py`` command surface. The
master proposes calibration parameter trials (fixed warm-up anchors + random or
Optuna-suggested samples), enqueues them, and workers evaluate each trial's
objective (mass-balance error) by delegating to a DarSIA calibration command.
The best trial per run is reported by ``best``.

This is the first port: the queue, warm-up strategy, parameter-range parsing and
best-trial selection are implemented. The two remaining pieces to wire to new
DarSIA are (1) the objective evaluation inside the worker command and
(2) optional memmap acceleration; both are clearly marked.

Commands (mirroring the Word-document usage)::

    python scripts/distributed_auto_calibration_queue.py master \
        --queue \\share\Kalibrering --runs ac53 ac60 ... \
        --max-iters 500 --warmup-iters 0 --warmup-mode suffix \
        --param-ranges "value1=0,2;value2=0,2;...;value6=0,2" \
        --param-levels "value1=8;...;value6=8" --per-label true --use-facies true

    python scripts/distributed_auto_calibration_queue.py watchdog \
        --queue \\share\Kalibrering --workers 12 \
        --control-dir \\share\Kalibrering\worker_limits

    python scripts/distributed_auto_calibration_queue.py best \
        --queue \\share\Kalibrering --runs ac53 ...
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
import sys
from pathlib import Path

import queue_core as qc
import _workflow_common as wc

LOG = logging.getLogger("auto_calib_queue")

def _last_float(text):
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None



def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def parse_param_ranges(spec: str) -> dict[str, tuple[float, float]]:
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        name, rng = part.split("=", 1)
        lo, hi = rng.split(",")
        out[name.strip()] = (float(lo), float(hi))
    return out


def parse_param_levels(spec: str) -> dict[str, int]:
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        name, n = part.split("=", 1)
        out[name.strip()] = int(n)
    return out


def warmup_anchors(ranges: dict[str, tuple[float, float]]) -> list[dict]:
    """Fixed warm-up points: all-low, all-mid, all-high (as in ff_um)."""
    anchors = []
    for frac in (0.0, 0.5, 1.0):
        anchors.append({k: lo + frac * (hi - lo) for k, (lo, hi) in ranges.items()})
    return anchors


def level_sweeps(ranges, levels) -> list[dict]:
    """Per-parameter monotonic sweeps over the requested number of levels."""
    out = []
    base = {k: (lo + hi) / 2 for k, (lo, hi) in ranges.items()}
    for k, (lo, hi) in ranges.items():
        n = max(1, levels.get(k, 0))
        for i in range(n):
            v = lo + (hi - lo) * i / max(1, n - 1)
            trial = dict(base)
            trial[k] = v
            out.append(trial)
    return out


def random_trials(ranges, n, seed=0) -> list[dict]:
    rng = random.Random(seed)
    return [{k: rng.uniform(lo, hi) for k, (lo, hi) in ranges.items()}
            for _ in range(n)]


# -------------------------------------------------------------------------
def cmd_master(a: argparse.Namespace) -> None:
    if a.groups_file:
        import json
        groups = json.loads(Path(a.groups_file).read_text())
        reps = [g["representative"].lower() for g in groups.values()]
        a.runs = reps
        LOG.info("Calibrating %d group representatives: %s", len(reps), reps)
    if not a.runs:
        raise SystemExit("Provide --runs or --groups-file")
    ranges = parse_param_ranges(a.param_ranges)
    levels = parse_param_levels(a.param_levels) if a.param_levels else {}
    q = qc.Queue(a.queue)
    if a.reset_queue:
        q.reset()
    q.ensure_layout()

    trials: list[dict] = []
    if _bool(a.baseline_trial):
        # empty params = evaluate the loaded (e.g. AC60-seeded) calibration as-is,
        # so Optuna starts from the known-good baseline before perturbing values
        trials.append({})
    trials += warmup_anchors(ranges)
    if levels:
        trials += level_sweeps(ranges, levels)
    n_random = max(0, a.warmup_iters - len(trials))
    trials += random_trials(ranges, n_random)
    # remaining budget filled with random samples (placeholder for Optuna search)
    remaining = max(0, a.max_iters - len(trials))
    trials += random_trials(ranges, remaining, seed=1)

    tasks = []
    for run in a.runs:
        run_cfg = str(Path(a.config_dir) / f"{run}.toml")
        configs = f'"{a.common_config}" "{run_cfg}"'
        for i, params in enumerate(trials):
            tasks.append({"kind": "calibration", "run": run, "trial": i,
                          "params": params, "configs": configs,
                          "per_label": _bool(a.per_label),
                          "use_facies": _bool(a.use_facies)})
    n = q.enqueue(tasks, tasks_per_file=a.tasks_per_file)
    LOG.info("Enqueued %d trial-task(s) (%d trials x %d run(s)) in %d batch(es)",
             len(tasks), len(trials), len(a.runs), n)


def _make_execute(worker_cmd: str):
    def execute(task: dict) -> dict:
        params_str = ";".join(f"{k}={v:.6g}" for k, v in task["params"].items())
        res = wc.run_subprocess_task(worker_cmd, {
            "configs": task["configs"], "params": params_str})
        # worker prints objective value on its last line
        res["objective"] = _last_float(res.get("stdout", ""))
        res.update({"run": task["run"], "trial": task["trial"],
                    "params": task["params"]})
        return res
    return execute


def cmd_worker(a: argparse.Namespace) -> None:
    q = qc.Queue(a.queue)
    qc.run_worker(q, _make_execute(a.worker_cmd), wc.worker_config_from_args(a))


def cmd_watchdog(a: argparse.Namespace) -> None:
    q = qc.Queue(a.queue)
    extra = ["--worker-cmd", a.worker_cmd]
    if a.stop_when_drained:
        extra += ["--exit-when-idle", "--idle-seconds", "3", "--max-tasks", "100000"]
    worker_cmd = wc.build_worker_cmd(__file__, a.queue, extra)
    qc.run_watchdog(q, worker_cmd, wc.watchdog_config_from_args(a),
                    stop_when_drained=a.stop_when_drained)


def cmd_best(a: argparse.Namespace) -> None:
    q = qc.Queue(a.queue)
    best: dict[str, dict] = {}
    for r in qc.iter_results(q):
        run, obj = r.get("run"), r.get("objective")
        if obj is None:
            continue
        if run not in best or obj < best[run]["objective"]:
            best[run] = r
    for run in sorted(best):
        LOG.info("BEST %s: objective=%.6g params=%s",
                 run, best[run]["objective"], best[run]["params"])
    if not best:
        LOG.warning("No completed trials with objective found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("master")
    wc.add_common_queue_args(m)
    m.add_argument("--runs", nargs="*", default=[])
    m.add_argument("--config-dir", default="config/run_ac")
    m.add_argument("--common-config", default="config/common.toml")
    m.add_argument("--groups-file", default=None,
                   help="calibration_groups/groups.json: calibrate only representatives")
    m.add_argument("--baseline-trial", default="true",
                   help="evaluate the loaded (seeded) calibration as trial 0 before perturbing")
    m.add_argument("--max-iters", type=int, default=500)
    m.add_argument("--warmup-iters", type=int, default=0)
    m.add_argument("--warmup-mode", default="suffix",
                   choices=["prefix", "suffix", "single"])
    m.add_argument("--param-ranges", required=True)
    m.add_argument("--param-levels", default="")
    m.add_argument("--per-label", default="true")
    m.add_argument("--use-facies", default="false")
    m.add_argument("--use-history", default="false")
    m.add_argument("--use-last-best", default="false")
    m.add_argument("--reset-queue", action="store_true")
    # memmap acceleration (accepted; not yet active - CONFIRM)
    m.add_argument("--memmap-cache", default=None)
    m.add_argument("--quality-scale", type=float, default=1.0)
    m.add_argument("--quality-dtype", default="float64")
    m.set_defaults(func=cmd_master)

    CALIB_WORKER_CMD = (sys.executable + " scripts/calibration_worker.py "
                        '--config {configs} --params "{params}"')
    w = sub.add_parser("watchdog")
    wc.add_watchdog_args(w)
    w.add_argument("--per-label", default="true")
    w.add_argument("--use-facies", default="false")
    w.add_argument("--worker-cmd", default=CALIB_WORKER_CMD)
    w.set_defaults(func=cmd_watchdog)

    wk = sub.add_parser("worker")
    wc.add_worker_args(wk)
    wk.add_argument("--worker-cmd", default=CAL