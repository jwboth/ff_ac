"""Launch five full AC calibration campaigns across CPU and local GPU lanes.

The CPU and GPU assignments are disjoint. CPU runs use the shared queue with
one persistent worker per run, while each GPU lane owns its runs locally and
keeps one prepared context alive until that run is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ac_production_campaign import (  # noqa: E402
    EXPECTED_RUNTIME_DEPTH_SHA256,
    FINAL_PRODUCTION_RUNS,
)
from scripts.verify_ac_depth_maps import (  # noqa: E402
    EXPECTED_DEPTH_ARRAY_SHA256,
    verify_depth_maps,
)


CAMPAIGN_ID = "final5_20260729"
DEFAULT_LOGS_ROOT = Path(
    rf"Z:\Albus\Autokalibrering_log\{CAMPAIGN_ID}"
)
DEFAULT_QUEUE_ROOT = (
    rf"\\Moderskipet\Darsia_Queue\Kalibrering_AC_{CAMPAIGN_ID}"
)
DEFAULT_SEED_PARAMS = Path(
    r"Z:\Albus\Autokalibrering_log"
    r"\final_production_20260723_16frames_24x1\final_seed_params.json"
)
RESULTS_ROOT = Path(r"Z:\Albus\Results")
MEASUREMENTS_PATH = REPO_ROOT / "data" / "depth_measurements.csv"
BOUNDS_FILE = "config/bounds_seg6_titration.json"
REQUIRED_TIMES_H = (4.1, 6.0, 8.0)
EXPECTED_FRAME_COUNT = 16


@dataclass(frozen=True)
class VariantSpec:
    name: str
    seed: int
    color_path_anchor: bool
    note: str
    cpu_workers_per_host: int


@dataclass(frozen=True)
class GpuLane:
    name: str
    host: str
    backend: str
    variant: str
    runs: tuple[str, ...]


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    kind: str
    queue: str
    command: tuple[str, ...]
    stdout: str
    env: dict[str, str]
    restartable: bool = True


VARIANTS = (
    VariantSpec(
        "control_s17",
        17,
        False,
        "Best six-run method: per-run color paths, strict AC14, point-wise L1.",
        3,
    ),
    VariantSpec(
        "control_s73",
        73,
        False,
        "Independent Optuna replication of the winning control method.",
        3,
    ),
    VariantSpec(
        "control_s151",
        151,
        False,
        "Second independent replication of the winning control method.",
        2,
    ),
    VariantSpec(
        "sharedpath_s17",
        17,
        True,
        "Runner-up: AC60-regularized color-path shape with point-wise L1.",
        2,
    ),
    VariantSpec(
        "sharedpath_s73",
        73,
        True,
        "Independent Optuna replication of the runner-up method.",
        2,
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_alternating(runs: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(runs[::2]), tuple(runs[1::2])


def cpu_runs(variant: VariantSpec) -> tuple[str, ...]:
    count = 2 * variant.cpu_workers_per_host
    return tuple(FINAL_PRODUCTION_RUNS[:count])


def gpu_runs(variant: VariantSpec) -> tuple[str, ...]:
    owned_by_cpu = set(cpu_runs(variant))
    return tuple(run for run in FINAL_PRODUCTION_RUNS if run not in owned_by_cpu)


def gpu_lanes() -> tuple[GpuLane, ...]:
    control17_a, control17_b = _split_alternating(
        gpu_runs(VARIANT_BY_NAME["control_s17"])
    )
    control73_a, control73_b = _split_alternating(
        gpu_runs(VARIANT_BY_NAME["control_s73"])
    )
    return (
        GpuLane(
            "cuda_control_s17",
            "moderskipet",
            "cuda",
            "control_s17",
            control17_a,
        ),
        GpuLane(
            "cuda_control_s73",
            "moderskipet",
            "cuda",
            "control_s73",
            control73_a,
        ),
        GpuLane(
            "cuda_control_s151",
            "moderskipet",
            "cuda",
            "control_s151",
            gpu_runs(VARIANT_BY_NAME["control_s151"]),
        ),
        GpuLane(
            "opencl_control_s17",
            "olav",
            "opencl",
            "control_s17",
            control17_b,
        ),
        GpuLane(
            "opencl_control_s73",
            "olav",
            "opencl",
            "control_s73",
            control73_b,
        ),
        GpuLane(
            "opencl_sharedpath_s17",
            "olav",
            "opencl",
            "sharedpath_s17",
            gpu_runs(VARIANT_BY_NAME["sharedpath_s17"]),
        ),
        GpuLane(
            "cuda_sharedpath_s73",
            "moderskipet",
            "cuda",
            "sharedpath_s73",
            gpu_runs(VARIANT_BY_NAME["sharedpath_s73"]),
        ),
    )


def validate_plan() -> dict:
    assignments: dict[tuple[str, str], str] = {}
    cpu_count = 0
    for variant in VARIANTS:
        for run in cpu_runs(variant):
            key = (variant.name, run)
            if key in assignments:
                raise RuntimeError(f"Duplicate assignment: {key}")
            assignments[key] = "cpu"
            cpu_count += 1
    for lane in gpu_lanes():
        for run in lane.runs:
            key = (lane.variant, run)
            if key in assignments:
                raise RuntimeError(
                    f"Duplicate assignment: {key} ({assignments[key]} and {lane.name})"
                )
            assignments[key] = lane.name

    expected = {
        (variant.name, run)
        for variant in VARIANTS
        for run in FINAL_PRODUCTION_RUNS
    }
    missing = expected - set(assignments)
    extra = set(assignments) - expected
    if missing or extra:
        raise RuntimeError(
            f"Invalid campaign coverage: missing={sorted(missing)} extra={sorted(extra)}"
        )
    cpu_workers_per_host = sum(
        variant.cpu_workers_per_host for variant in VARIANTS
    )
    if cpu_workers_per_host != 12:
        raise RuntimeError(
            f"Expected 12 CPU workers per host, got {cpu_workers_per_host}"
        )
    host_gpu_counts = {
        host: sum(1 for lane in gpu_lanes() if lane.host == host)
        for host in ("moderskipet", "olav")
    }
    if host_gpu_counts != {"moderskipet": 4, "olav": 3}:
        raise RuntimeError(f"Unexpected GPU lane counts: {host_gpu_counts}")
    return {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "run_count": len(FINAL_PRODUCTION_RUNS),
        "variant_count": len(VARIANTS),
        "calibration_count": len(expected),
        "cpu_assignments": cpu_count,
        "gpu_assignments": len(expected) - cpu_count,
        "cpu_workers_per_host": cpu_workers_per_host,
        "gpu_workers": host_gpu_counts,
        "max_iters": 1600,
        "warmup_iters": 150,
        "max_in_flight_per_run": 1,
        "cached_depth_sha256": EXPECTED_DEPTH_ARRAY_SHA256,
        "runtime_depth_sha256": EXPECTED_RUNTIME_DEPTH_SHA256,
        "variants": [
            {
                **asdict(variant),
                "cpu_runs": list(cpu_runs(variant)),
                "gpu_runs": list(gpu_runs(variant)),
            }
            for variant in VARIANTS
        ],
        "gpu_lanes": [asdict(lane) for lane in gpu_lanes()],
        "assignments": [
            {"variant": variant, "run": run, "resource": resource}
            for (variant, run), resource in sorted(assignments.items())
        ],
    }


def _load_seed_metadata(seed_path: Path) -> dict:
    payload = json.loads(seed_path.read_text(encoding="utf-8-sig"))
    metadata = payload.get("metadata")
    params_by_run = payload.get("params_by_run")
    if not isinstance(metadata, dict) or not isinstance(params_by_run, dict):
        raise RuntimeError(f"Invalid seed-parameter file: {seed_path}")
    missing = sorted(set(FINAL_PRODUCTION_RUNS) - set(params_by_run))
    if missing:
        raise RuntimeError(f"Seed parameters missing runs: {missing}")
    times = [float(value) for value in metadata.get("calibration_times_h", [])]
    if len(times) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAME_COUNT} calibration frames, got {len(times)}"
        )
    for required in REQUIRED_TIMES_H:
        if not any(abs(required - actual) < 1e-9 for actual in times):
            raise RuntimeError(f"Required calibration time {required:g} h is missing")
    return metadata


def _git_value(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    return completed.stdout.strip()


def run_preflight(logs_root: Path, seed_path: Path) -> dict:
    if len(FINAL_PRODUCTION_RUNS) != 40:
        raise RuntimeError(
            f"Expected 40 production runs, got {len(FINAL_PRODUCTION_RUNS)}"
        )
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed parameters unavailable: {seed_path}")
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(
            f"Results root unavailable in this session: {RESULTS_ROOT}"
        )
    plan = validate_plan()
    seed_metadata = _load_seed_metadata(seed_path)
    depth = verify_depth_maps(
        list(FINAL_PRODUCTION_RUNS),
        results_root=RESULTS_ROOT,
        measurements_path=MEASUREMENTS_PATH,
    )
    host = socket.gethostname().upper()
    payload = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "checked_at_utc": _utc_now(),
        "host": host,
        "ok": True,
        "ff_ac_commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "darsia_commit": _git_value(
            REPO_ROOT / "external" / "darsia", "rev-parse", "HEAD"
        ),
        "seed_file": str(seed_path),
        "seed_metadata": seed_metadata,
        "depth": depth,
        "plan_summary": {
            key: plan[key]
            for key in (
                "run_count",
                "variant_count",
                "calibration_count",
                "cpu_assignments",
                "gpu_assignments",
                "cpu_workers_per_host",
                "gpu_workers",
            )
        },
    }
    logs_root.mkdir(parents=True, exist_ok=True)
    (logs_root / f"depth_preflight_{host}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (logs_root / "campaign_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )
    return payload


def _variant_env(variant: VariantSpec, *, master: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["FFAC_REQUIRE_VARYING_DEPTH"] = "on"
    env["FFAC_EXPECTED_DEPTH_SHA256"] = EXPECTED_RUNTIME_DEPTH_SHA256
    env["FFAC_TITRATION_FLASH"] = "on"
    env["FFAC_TEMPLATE_REGISTRATION"] = "ac14_template"
    env["FFAC_TEMPLATE_REGISTRATION_MODE"] = "partial_affine"
    env["FFAC_TEMPLATE_REGISTRATION_STRICT"] = "on"
    if master:
        env["FFAC_MASTER_LIGHT_CONTEXT"] = "on"
    else:
        env.pop("FFAC_MASTER_LIGHT_CONTEXT", None)
    for name in (
        "FFAC_STATIC_LIGHT_CORRECTION",
        "FFAC_STATIC_LIGHT_REFERENCE",
        "FFAC_STATIC_LIGHT_SPATIAL_SIGMA",
        "FFAC_COUPLE_AQ_GAS",
        "FFAC_SIGNAL_PARAMETERIZATION",
        "FFAC_PHASE_SEPARATION",
    ):
        env.pop(name, None)
    if variant.color_path_anchor:
        env["FFAC_COLOR_PATH_ANCHOR"] = "ac60"
        env["FFAC_COLOR_PATH_ANCHOR_WEIGHT"] = "0.75"
        env["FFAC_COLOR_PATH_ANCHOR_STRICT"] = "on"
    else:
        env.pop("FFAC_COLOR_PATH_ANCHOR", None)
        env.pop("FFAC_COLOR_PATH_ANCHOR_WEIGHT", None)
        env.pop("FFAC_COLOR_PATH_ANCHOR_STRICT", None)
    return env


def _master_command(
    *,
    python: Path,
    queue: str,
    runs: Sequence[str],
    logs_dir: Path,
    control_dir: str,
    variant: VariantSpec,
    seed_path: Path,
    optuna_dir: Path,
    local_backend: str | None = None,
) -> tuple[str, ...]:
    command = [
        str(python),
        "scripts/distributed_auto_calibration_queue.py",
        "master",
        "--queue",
        queue,
        "--no-clear-queue",
        "--runs",
        *runs,
        "--config-dir",
        "config_seg6/run_ac",
        "--logs-dir",
        str(logs_dir),
        "--exact-logs-dir",
        "--use-facies",
        "true",
        "--per-label",
        "true",
        "--objective-integral",
        "off",
        "--bounds-file",
        BOUNDS_FILE,
        "--no-save-calibration",
        "--seed-params-file",
        str(seed_path),
        "--optuna-seed",
        str(variant.seed),
        "--optuna-storage-dir",
        str(optuna_dir),
        "--max-iters",
        "1600",
        "--warmup-iters",
        "150",
        "--run-mode",
        "parallel",
        "--max-active-runs",
        "1" if local_backend else str(len(runs)),
        "--max-in-flight-per-run",
        "1",
        "--control-dir",
        control_dir,
        "--sanity-every",
        "0",
        "--sanity-scale",
        "1.00",
        "--quality-dtype",
        "float32",
    ]
    if not local_backend:
        command.extend(["--optuna-persist", "true"])
    if local_backend:
        command.extend(
            [
                "--local-eval-backend",
                local_backend,
                "--local-run",
                runs[0],
            ]
        )
    return tuple(command)


def _watchdog_command(
    *,
    python: Path,
    queue: str,
    control_dir: str,
    workers: int,
    worker_prefix: str,
) -> tuple[str, ...]:
    return (
        str(python),
        "scripts/distributed_auto_calibration_queue.py",
        "watchdog",
        "--queue",
        queue,
        "--config-dir",
        "config_seg6/run_ac",
        "--use-facies",
        "true",
        "--per-label",
        "true",
        "--bounds-file",
        BOUNDS_FILE,
        "--control-dir",
        control_dir,
        "--workers",
        str(workers),
        "--worker-id-prefix",
        worker_prefix,
        "--worker-stall-seconds",
        "600",
        "--idle-exit-seconds",
        "300",
        "--threads-per-worker",
        "1",
        "--eval-backend",
        "prepared",
        "--run-affinity",
        "strict",
        "--stickiness-wait-seconds",
        "0",
        "--max-tasks-per-worker",
        "0",
    )


def build_process_specs(
    *,
    role: str,
    logs_root: Path,
    queue_root: str,
    seed_path: Path,
) -> tuple[ProcessSpec, ...]:
    role = role.lower()
    if role not in {"moderskipet", "olav"}:
        raise ValueError(f"Unknown role: {role}")
    python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        raise FileNotFoundError(f"Python unavailable: {python}")
    local_root = REPO_ROOT / "logs" / CAMPAIGN_ID / role
    local_optuna = (
        Path(os.environ.get("LOCALAPPDATA", str(REPO_ROOT / "logs")))
        / "ff_ac"
        / "optuna"
        / CAMPAIGN_ID
    )
    specs: list[ProcessSpec] = []

    for variant in VARIANTS:
        queue = f"{queue_root}_{variant.name}_cpu"
        control = f"{queue}\\control"
        variant_logs = logs_root / variant.name / "cpu"
        if role == "moderskipet":
            command = _master_command(
                python=python,
                queue=queue,
                runs=cpu_runs(variant),
                logs_dir=variant_logs,
                control_dir=control,
                variant=variant,
                seed_path=seed_path,
                optuna_dir=local_optuna / f"{variant.name}_cpu_master",
            )
            specs.append(
                ProcessSpec(
                    name=f"master_cpu_{variant.name}",
                    kind="master",
                    queue=queue,
                    command=command,
                    stdout=str(local_root / f"master_cpu_{variant.name}.log"),
                    env=_variant_env(variant, master=True),
                )
            )
        command = _watchdog_command(
            python=python,
            queue=queue,
            control_dir=control,
            workers=variant.cpu_workers_per_host,
            worker_prefix=f"{socket.gethostname()}_{variant.name}_cpu",
        )
        specs.append(
            ProcessSpec(
                name=f"watchdog_cpu_{variant.name}",
                kind="watchdog",
                queue=queue,
                command=command,
                stdout=str(local_root / f"watchdog_cpu_{variant.name}.log"),
                env=_variant_env(variant, master=False),
            )
        )

    for lane in gpu_lanes():
        if lane.host != role:
            continue
        variant = VARIANT_BY_NAME[lane.variant]
        queue = f"{queue_root}_{lane.name}"
        control = f"{queue}\\control"
        lane_logs = logs_root / lane.variant / lane.name
        command = _master_command(
            python=python,
            queue=queue,
            runs=lane.runs,
            logs_dir=lane_logs,
            control_dir=control,
            variant=variant,
            seed_path=seed_path,
            optuna_dir=local_optuna / lane.name,
            local_backend=lane.backend,
        )
        specs.append(
            ProcessSpec(
                name=f"gpu_{lane.name}",
                kind="gpu",
                queue=queue,
                command=command,
                stdout=str(local_root / f"gpu_{lane.name}.log"),
                env=_variant_env(variant, master=True),
            )
        )
    return tuple(specs)


def _active_campaign_processes() -> list[dict]:
    marker = CAMPAIGN_ID.lower()
    excluded = {os.getpid()}
    try:
        parent = psutil.Process(os.getpid()).parent()
        while parent is not None:
            excluded.add(parent.pid)
            parent = parent.parent()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    active = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if process.pid in excluded or marker not in command.lower():
            continue
        active.append(
            {
                "pid": process.pid,
                "name": process.info.get("name"),
                "command": command,
            }
        )
    return active


def _spawn(spec: ProcessSpec) -> subprocess.Popen:
    output = Path(spec.stdout)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("ab")
    try:
        process = subprocess.Popen(
            list(spec.command),
            cwd=REPO_ROOT,
            env=spec.env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        stream.close()
    return process


def _queue_complete(queue: str) -> bool:
    try:
        return (Path(queue) / "master_complete.json").exists()
    except OSError:
        return False


def launch(args: argparse.Namespace) -> int:
    logs_root = Path(args.logs_root)
    seed_path = Path(args.seed_params)
    preflight = run_preflight(logs_root, seed_path)
    specs = build_process_specs(
        role=args.role,
        logs_root=logs_root,
        queue_root=args.queue_root,
        seed_path=seed_path,
    )
    active = _active_campaign_processes()
    if active:
        raise RuntimeError(
            "Campaign processes already active: "
            + ", ".join(str(item["pid"]) for item in active)
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "preflight": preflight,
                    "processes": [
                        {
                            **asdict(spec),
                            "env": {
                                key: value
                                for key, value in spec.env.items()
                                if key.startswith("FFAC_")
                            },
                        }
                        for spec in specs
                    ],
                },
                indent=2,
            )
        )
        return 0

    managed: dict[str, dict] = {}
    role = args.role.lower()
    for spec in specs:
        process = _spawn(spec)
        managed[spec.name] = {
            "spec": spec,
            "process": process,
            "restarts": 0,
            "last_exit_code": None,
        }
        print(f"START {spec.name} pid={process.pid}", flush=True)
        if spec.kind == "master":
            time.sleep(2)
        elif spec.kind == "gpu":
            time.sleep(args.gpu_stagger_seconds)

    manifest_path = logs_root / f"launch_manifest_{socket.gethostname().upper()}.json"
    status_path = logs_root / f"supervisor_{socket.gethostname().upper()}.json"
    manifest = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "started_at_utc": _utc_now(),
        "host": socket.gethostname().upper(),
        "role": role,
        "ff_ac_commit": preflight["ff_ac_commit"],
        "darsia_commit": preflight["darsia_commit"],
        "processes": [
            {
                "name": name,
                "kind": item["spec"].kind,
                "pid": item["process"].pid,
                "queue": item["spec"].queue,
                "stdout": item["spec"].stdout,
                "command": list(item["spec"].command),
            }
            for name, item in managed.items()
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    while True:
        running = 0
        status_rows = []
        for name, item in managed.items():
            spec: ProcessSpec = item["spec"]
            process: subprocess.Popen = item["process"]
            exit_code = process.poll()
            if exit_code is not None:
                item["last_exit_code"] = exit_code
                if (
                    spec.restartable
                    and not _queue_complete(spec.queue)
                    and item["restarts"] < args.max_restarts
                ):
                    time.sleep(5)
                    process = _spawn(spec)
                    item["process"] = process
                    item["restarts"] += 1
                    exit_code = None
                    print(
                        f"RESTART {name} pid={process.pid} "
                        f"count={item['restarts']}",
                        flush=True,
                    )
            if exit_code is None:
                running += 1
            status_rows.append(
                {
                    "name": name,
                    "kind": spec.kind,
                    "pid": item["process"].pid,
                    "running": exit_code is None,
                    "exit_code": exit_code,
                    "last_exit_code": item["last_exit_code"],
                    "restarts": item["restarts"],
                    "queue_complete": _queue_complete(spec.queue),
                }
            )
        status_path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "campaign_id": CAMPAIGN_ID,
                    "updated_at_utc": _utc_now(),
                    "host": socket.gethostname().upper(),
                    "running_processes": running,
                    "processes": status_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if running == 0:
            failed = [
                row
                for row in status_rows
                if not row["queue_complete"] or row["exit_code"] not in (0, None)
            ]
            return 2 if failed else 0
        time.sleep(args.supervisor_poll_seconds)


def queue_status(queue: str) -> dict:
    root = Path(queue)

    def count(directory: str) -> int:
        path = root / directory
        try:
            return sum(1 for item in path.iterdir() if item.is_file())
        except (FileNotFoundError, OSError):
            return 0

    return {
        "queue": queue,
        "exists": root.exists(),
        "pending": count("pending"),
        "in_progress": count("in_progress"),
        "results": count("results"),
        "done": count("done"),
        "failed": count("failed"),
        "heartbeats": count("heartbeats"),
        "completed_runs": count("run_complete"),
        "master_complete": _queue_complete(queue),
    }


def campaign_status(args: argparse.Namespace) -> int:
    plan = validate_plan()
    queues = set()
    for variant in VARIANTS:
        queues.add(f"{args.queue_root}_{variant.name}_cpu")
    for lane in gpu_lanes():
        queues.add(f"{args.queue_root}_{lane.name}")
    payload = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "checked_at_utc": _utc_now(),
        "plan": {
            key: plan[key]
            for key in (
                "run_count",
                "variant_count",
                "calibration_count",
                "cpu_assignments",
                "gpu_assignments",
            )
        },
        "active_processes": _active_campaign_processes(),
        "queues": [queue_status(queue) for queue in sorted(queues)],
    }
    print(json.dumps(payload, indent=2))
    return 0


def stop_campaign(args: argparse.Namespace) -> int:
    command = [
        str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"),
        str(SCRIPT_DIR / "stop_ac_calibration_campaign.py"),
        "--marker",
        CAMPAIGN_ID,
        "--timeout",
        str(args.stop_timeout),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("launch", "status", "stop", "plan"),
    )
    parser.add_argument(
        "--role",
        choices=("moderskipet", "olav"),
        default=socket.gethostname().lower(),
    )
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT))
    parser.add_argument("--queue-root", default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--seed-params", default=str(DEFAULT_SEED_PARAMS))
    parser.add_argument("--gpu-stagger-seconds", type=float, default=15.0)
    parser.add_argument("--supervisor-poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--stop-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "launch":
        raise SystemExit(launch(args))
    if args.action == "status":
        raise SystemExit(campaign_status(args))
    if args.action == "stop":
        raise SystemExit(stop_campaign(args))
    print(json.dumps(validate_plan(), indent=2))


if __name__ == "__main__":
    main()
