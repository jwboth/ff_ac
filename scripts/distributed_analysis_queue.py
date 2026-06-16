"""Distributed mass/volume/segmentation analysis queue for ff_ac.

Ports the ff_um ``distributed_analysis_queue.py`` command surface onto the new
DarSIA preset workflow. The science (per-run analysis) is delegated to ff_ac's
own ``scripts/analysis.py`` CLI; this script only handles enqueueing,
parallel execution and merging via :mod:`queue_core`.

Commands (mirroring the ff_um Word-document usage)::

    # master: enqueue one analysis task per run
    python scripts/distributed_analysis_queue.py master \
        --queue \\share\Darsia_Queue \
        --runs ac17 ac18 ac53 ... --analysis mass --all --respect-blacklist --serial

    # watchdog: run workers on this host
    python scripts/distributed_analysis_queue.py watchdog \
        --queue \\share\Darsia_Queue --analysis mass --workers 14 \
        --control-dir \\share\Darsia_Queue\worker_limits

    # merge: collect per-run results
    python scripts/distributed_analysis_queue.py merge \
        --queue \\share\Darsia_Queue --runs ... --analysis mass --allow-missing

The exact analysis CLI invocation is configurable via ``--worker-cmd`` (default
points at ``scripts/analysis.py``); confirm the preset's flag names before use.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import queue_core as qc
import _workflow_common as wc

LOG = logging.getLogger("analysis_queue")

# Default command a worker runs per task. {config} and {flags} are filled in.
# CONFIRM: exact flag names of preset_analysis (scripts/analysis.py).
DEFAULT_WORKER_CMD = sys.executable + " scripts/analysis.py --config {configs} {flags}"


def _flags_for_analysis(analysis: str) -> str:
    return {"mass": "--mass", "volume": "--volume",
            "segmentation": "--segmentation", "all": "--all"}.get(analysis, "--mass")


# -------------------------------------------------------------------------
def cmd_master(a: argparse.Namespace) -> None:
    q = qc.Queue(a.queue)
    if a.reset_queue:
        q.reset()
    q.ensure_layout()
    flags = _flags_for_analysis(a.analysis)
    if a.respect_blacklist:
        flags += " --respect-blacklist"
    if a.all:
        flags += " --all"
    tasks = []
    for run in a.runs:
        run_cfg = str(Path(a.config_dir) / f"{run}.toml")
        # preset_analysis takes multiple --config files: common first, run second
        configs = f'"{a.common_config}" "{run_cfg}"'
        tasks.append({"kind": "analysis", "run": run, "analysis": a.analysis,
                      "configs": configs, "flags": flags})
    n = q.enqueue(tasks, tasks_per_file=1 if a.serial else a.tasks_per_file)
    LOG.info("Enqueued %d task(s) in %d batch(es) for analysis=%s",
             len(tasks), n, a.analysis)


def _make_execute(worker_cmd: str):
    def execute(task: dict) -> dict:
        return wc.run_subprocess_task(
            worker_cmd, {"configs": task["configs"], "flags": task["flags"]})
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


def cmd_merge(a: argparse.Namespace) -> None:
    q = qc.Queue(a.queue)
    rows = []
    for r in qc.iter_results(q):
        rows.append(r)
    out = Path(a.output_csv) if a.output_csv else Path(a.queue) / "analysis_merged.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    LOG.info("Merged %d result row(s) -> %s", len(rows), out)
    if not rows and not a.allow_missing:
        sys.exit("No results found (use --allow-missing to ignore)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("master")
    wc.add_common_queue_args(m)
    m.add_argument("--runs", nargs="+", required=True)
    m.add_argument("--config-dir", default="config/run_ac")
    m.add_argument("--common-config", default="config/common.toml")
    m.add_argument("--analysis", default="mass",
                   choices=["mass", "volume", "segmentation", "all"])
    m.add_argument("--all", action="store_true")
    m.add_argument("--respect-blacklist", action="store_true")
    m.add_argument("--serial", action="store_true")
    m.add_argument("--reset-queue", action="store_true")
    m.set_defaults(func=cmd_master)

    w = sub.add_parser("watchdog")
    wc.add_watchdog_args(w)
    w.add_argument("--analysis", default="mass")
    w.add_argument("--worker-cmd", default=DEFAULT_WORKER_CMD)
    w.set_defaults(func=cmd_watchdog)

    wk = sub.add_parser("worker")
    wc.add_worker_args(wk)
    wk.add_argument("--worker-cmd", default=DEFAULT_WORKER_CMD)
    wk.set_defaults(func=cmd_worker)

    g = sub.add_parser("merge")
    g.add_argument("--queue", required=True)
    g.add_argument("--runs", nargs="*", default=[])
    g.add_argument("--analysis", default="mass")
    g.add_argument("--output-csv", default=None)
    g.add_argument("--allow-missing", action="store_true")
    g.set_defaults(func=cmd_merge)

    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a.func(a)


if __name__ == "__main__":
    main()
