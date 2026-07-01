"""Emit commands for full AC production calibration campaigns.

The AC53 testbed is intentionally narrow. This helper emits repeatable
PowerShell commands for the production-sized runs we use on the shared
Moderskipet queue.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PRODUCTION_RUNS = [
    "ac17",
    "ac18",
    "ac19",
    "ac20",
    "ac21",
    "ac23",
    "ac24",
    "ac25",
    "ac28",
    "ac29",
    "ac30",
    "ac32",
    "ac33",
    "ac34",
    "ac35",
    "ac39",
    "ac40",
    "ac41",
    "ac43",
    "ac44",
    "ac45",
    "ac46",
    "ac47",
    "ac49",
    "ac52",
    "ac54",
    "ac55",
    "ac56",
    "ac57",
    "ac59",
    "ac60",
    "ac61",
]

ROLLOUT_RUNS = [
    "ac22",
    "ac26",
    "ac27",
    "ac31",
    "ac42",
    "ac48",
    "ac50",
    "ac51",
    "ac53",
    "ac58",
]


HARDCASE_RUNS = [
    # Start with distinct failure modes and controls so early batches are useful.
    "ac29",  # worst baseline: under-detects injection end, then late inflation
    "ac61",  # early over-detection outlier
    "ac20",  # good baseline control
    "ac24",
    "ac25",
    "ac32",
    "ac40",
    "ac44",
    "ac46",
    "ac49",
    "ac52",
    "ac60",  # best baseline control
]


@dataclass(frozen=True)
class Variant:
    name: str
    objective_integral: str
    static_light: str
    bounds_file: str = "config/bounds_seg6_titration.json"
    titration: bool = True
    note: str = ""


VARIANTS = [
    Variant(
        "coupled_baseline_l1",
        "off",
        "off",
        bounds_file="config/bounds_seg6_coupled.json",
        note="previous baseline family: titration flash with coupled bounds and no static light correction",
    ),
    Variant(
        "coupled_static_global_l1",
        "off",
        "blue-gain",
        bounds_file="config/bounds_seg6_coupled.json",
        note="coupled bounds plus global static FluidFlower light gain",
    ),
    Variant(
        "coupled_static_spatial_l1",
        "off",
        "blue-spatial",
        bounds_file="config/bounds_seg6_coupled.json",
        note="coupled bounds plus spatial static FluidFlower light gain",
    ),
    Variant(
        "coupled_static_spatial_drift025",
        "drift:0.25",
        "blue-spatial",
        bounds_file="config/bounds_seg6_coupled.json",
        note="coupled bounds plus spatial static light gain and weak plateau drift penalty",
    ),
    Variant(
        "coupled_static_global_drift025",
        "drift:0.25",
        "blue-gain",
        bounds_file="config/bounds_seg6_coupled.json",
        note="coupled bounds plus global static light gain and weak plateau drift penalty",
    ),
    Variant(
        "titration_relaxed_l1",
        "off",
        "off",
        bounds_file="config/bounds_seg6_titration_relaxed.json",
        note="titration-specific but relaxed flash bounds, no static light correction",
    ),
    Variant(
        "titration_relaxed_drift025",
        "drift:0.25",
        "off",
        bounds_file="config/bounds_seg6_titration_relaxed.json",
        note="relaxed titration bounds plus weak plateau drift penalty",
    ),
    Variant(
        "coupled_static_spatial_drift050",
        "drift:0.5",
        "blue-spatial",
        bounds_file="config/bounds_seg6_coupled.json",
        note="coupled bounds plus spatial static light gain and medium plateau drift penalty",
    ),
    Variant(
        "titration_static_spatial_l1",
        "off",
        "blue-spatial",
        note="titration with spatial static FluidFlower light gain",
    ),
    Variant(
        "titration_static_spatial_drift025",
        "drift:0.25",
        "blue-spatial",
        note="spatial static light gain plus weak plateau drift penalty",
    ),
    Variant(
        "titration_static_global_l1",
        "off",
        "blue-gain",
        note="titration with global static FluidFlower light gain",
    ),
    Variant(
        "titration_static_global_drift025",
        "drift:0.25",
        "blue-gain",
        note="global static light gain plus weak plateau drift penalty",
    ),
    Variant(
        "titration_static_spatial_drift050",
        "drift:0.5",
        "blue-spatial",
        note="spatial static light gain plus medium plateau drift penalty",
    ),
]

VARIANT_SETS = {
    "hardcase8": [
        "coupled_baseline_l1",
        "coupled_static_global_l1",
        "coupled_static_spatial_l1",
        "coupled_static_spatial_drift025",
        "coupled_static_global_drift025",
        "titration_relaxed_l1",
        "titration_relaxed_drift025",
        "coupled_static_spatial_drift050",
    ],
    "holiday4": [
        "titration_static_spatial_l1",
        "titration_static_spatial_drift025",
        "titration_static_global_l1",
        "titration_static_global_drift025",
    ],
    "holiday5": [
        "titration_static_spatial_l1",
        "titration_static_spatial_drift025",
        "titration_static_global_l1",
        "titration_static_global_drift025",
        "titration_static_spatial_drift050",
    ],
    "spatial": [
        "titration_static_spatial_l1",
        "titration_static_spatial_drift025",
        "titration_static_spatial_drift050",
    ],
    "global": [
        "titration_static_global_l1",
        "titration_static_global_drift025",
    ],
    "all": [variant.name for variant in VARIANTS],
}


def _ps_quote(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _variant_by_name(name: str) -> Variant:
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    known = sorted([*VARIANT_SETS, *(variant.name for variant in VARIANTS)])
    raise SystemExit(f"Unknown variant {name!r}. Known: {', '.join(known)}")


def _select_variants(spec: str) -> list[Variant]:
    names: list[str] = []
    for item in [part.strip() for part in spec.split(",") if part.strip()]:
        if item in VARIANT_SETS:
            names.extend(VARIANT_SETS[item])
        else:
            names.append(item)

    selected: list[Variant] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        selected.append(_variant_by_name(name))
        seen.add(name)
    return selected


def _select_runs(run_set: str) -> list[str]:
    if run_set == "production":
        return PRODUCTION_RUNS
    if run_set == "rollout":
        return ROLLOUT_RUNS
    if run_set == "hardcases":
        return HARDCASE_RUNS
    if run_set == "all":
        return [*PRODUCTION_RUNS, *ROLLOUT_RUNS]
    runs = [part.strip().lower() for part in run_set.replace(",", " ").split() if part.strip()]
    if not runs:
        raise SystemExit("No runs selected.")
    return runs


def _env_lines(variant: Variant, spatial_sigma: float) -> list[str]:
    lines = [
        "$env:FFAC_TITRATION_FLASH = 'on'"
        if variant.titration
        else "Remove-Item Env:\\FFAC_TITRATION_FLASH -ErrorAction SilentlyContinue",
        f"$env:FFAC_STATIC_LIGHT_CORRECTION = '{variant.static_light}'"
        if variant.static_light != "off"
        else "Remove-Item Env:\\FFAC_STATIC_LIGHT_CORRECTION -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_COUPLE_AQ_GAS -ErrorAction SilentlyContinue",
    ]
    if "spatial" in variant.static_light:
        lines.append(f"$env:FFAC_STATIC_LIGHT_SPATIAL_SIGMA = '{spatial_sigma:g}'")
    else:
        lines.append("Remove-Item Env:\\FFAC_STATIC_LIGHT_SPATIAL_SIGMA -ErrorAction SilentlyContinue")
    return lines


def _common_master_args(args: argparse.Namespace, variant: Variant, queue: str, logs_dir: str, control: str) -> list[str]:
    runs = " ".join(_select_runs(args.run_set))
    command = [
        f"& {args.python}",
        "scripts/distributed_auto_calibration_queue.py master",
        "--queue",
        _ps_quote(queue),
        "--runs",
        runs,
        "--config-dir config_seg6/run_ac",
        "--logs-dir",
        _ps_quote(logs_dir),
        "--use-facies true",
        "--per-label true",
        "--objective-integral",
        _ps_quote(variant.objective_integral),
        "--bounds-file",
        _ps_quote(variant.bounds_file),
    ]
    if not args.save_calibration:
        command.append("--no-save-calibration")
    command.extend(
        [
            "--max-iters",
            str(args.max_iters),
            "--warmup-iters",
            str(args.warmup_iters),
            "--run-mode parallel",
            "--max-active-runs",
            str(args.max_active_runs),
            "--max-in-flight-per-run",
            str(args.max_in_flight_per_run),
            "--control-dir",
            _ps_quote(control),
            "--sanity-every",
            str(args.sanity_every),
            "--sanity-scale 1.00",
            "--quality-dtype float32",
        ]
    )
    if args.exact_logs_dir:
        command.append("--exact-logs-dir")
    if args.use_history:
        command.extend(["--use-history", "true"])
    if args.no_clear_queue or args.resume_existing:
        command.append("--no-clear-queue")
    return command


def _watchdog_args(
    args: argparse.Namespace,
    variant: Variant,
    queue: str,
    control: str,
    workers: int | None = None,
) -> list[str]:
    worker_count = args.workers if workers is None else workers
    command = [
        f"& {args.python}",
        "scripts/distributed_auto_calibration_queue.py watchdog",
        "--queue",
        _ps_quote(queue),
        "--config-dir config_seg6/run_ac",
        "--use-facies true",
        "--per-label true",
        "--control-dir",
        _ps_quote(control),
        "--workers",
        str(worker_count),
        "--worker-stall-seconds 600",
    ]
    if args.max_tasks_per_worker > 0:
        command.extend(["--max-tasks-per-worker", str(args.max_tasks_per_worker)])
    return command


def _worker_counts(args: argparse.Namespace, selected: list[Variant]) -> list[int]:
    total = int(getattr(args, "total_workers", 0) or 0)
    if total <= 0:
        return [int(args.workers)] * len(selected)
    if total < len(selected):
        raise SystemExit(
            f"--total-workers {total} is lower than the number of variants ({len(selected)}). "
            "Use fewer variants or at least one worker per variant."
        )
    base, extra = divmod(total, len(selected))
    return [base + (1 if idx < extra else 0) for idx in range(len(selected))]


def commands(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    selected = _select_variants(args.variant)
    worker_counts = _worker_counts(args, selected)
    run_count = len(_select_runs(args.run_set))
    queue_root = args.queue_root.rstrip("\\/")
    logs_root = args.logs_root.rstrip("\\/")
    control_root = args.control_root.rstrip("\\/") if args.control_root else queue_root
    if args.resume_existing:
        args.exact_logs_dir = True
        args.use_history = True

    print("# AC production calibration campaign.")
    print(f"# Run set: {args.run_set} ({run_count} runs).")
    print(f"# Variants: {', '.join(variant.name for variant in selected)}")
    print("# Start one master per variant, then one watchdog per active machine for the same variant.")
    print("# --workers is per watchdog. With holiday4, the default 3 gives 12 workers per machine.")
    print("# Sync this repository to every worker machine before starting spatial variants.")
    print(f"cd {_ps_quote(repo)}")
    print()

    for variant in selected:
        queue = f"{queue_root}_{variant.name}"
        logs_dir = f"{logs_root}\\{variant.name}"
        if args.resume_existing:
            variant_log_root = Path(logs_dir)
            existing = [
                path for path in variant_log_root.glob("*")
                if path.is_dir() and (path / "commands.txt").exists()
            ] if variant_log_root.exists() else []
            if existing:
                logs_dir = str(max(existing, key=lambda path: path.stat().st_mtime))
        control = f"{control_root}_{variant.name}\\control"
        print(f"# --- {variant.name}: {variant.note} ---")
        print("# master terminal:")
        for line in _env_lines(variant, args.spatial_sigma):
            print(line)
        print("$env:FFAC_MASTER_LIGHT_CONTEXT = 'on'")
        print(" ".join(_common_master_args(args, variant, queue, logs_dir, control)))
        print()
        print("# watchdog terminal on each machine:")
        for line in _env_lines(variant, args.spatial_sigma):
            print(line)
        print("Remove-Item Env:\\FFAC_MASTER_LIGHT_CONTEXT -ErrorAction SilentlyContinue")
        print(" ".join(_watchdog_args(args, variant, queue, control)))
        print()


def _variant_env(base_env: dict[str, str], variant: Variant, *, master: bool, spatial_sigma: float) -> dict[str, str]:
    env = dict(base_env)
    if variant.titration:
        env["FFAC_TITRATION_FLASH"] = "on"
    else:
        env.pop("FFAC_TITRATION_FLASH", None)
        env.pop("FFAC_TITRATION_RECIPE", None)
    if variant.static_light != "off":
        env["FFAC_STATIC_LIGHT_CORRECTION"] = variant.static_light
    else:
        env.pop("FFAC_STATIC_LIGHT_CORRECTION", None)
    env.pop("FFAC_COUPLE_AQ_GAS", None)
    if "spatial" in variant.static_light:
        env["FFAC_STATIC_LIGHT_SPATIAL_SIGMA"] = f"{spatial_sigma:g}"
    else:
        env.pop("FFAC_STATIC_LIGHT_SPATIAL_SIGMA", None)
    if master:
        env["FFAC_MASTER_LIGHT_CONTEXT"] = "on"
    else:
        env.pop("FFAC_MASTER_LIGHT_CONTEXT", None)
    return env


def _powershell_process(command: list[str], repo: Path, env: dict[str, str], log_path: Path, *, dry_run: bool) -> int | None:
    block = " ".join(command)
    print(block)
    if dry_run:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                block,
            ],
            cwd=repo,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    finally:
        out.close()
    return int(proc.pid)


def launch(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    selected = _select_variants(args.variant)
    worker_counts = _worker_counts(args, selected)
    run_count = len(_select_runs(args.run_set))
    queue_root = args.queue_root.rstrip("\\/")
    logs_root = args.logs_root.rstrip("\\/")
    control_root = args.control_root.rstrip("\\/") if args.control_root else queue_root
    role = args.role.lower()
    start_master = role in {"local", "all", "master", "masters"}
    start_watchdog = role in {"local", "all", "watchdog", "watchdogs"}
    if not start_master and not start_watchdog:
        raise SystemExit(f"Unknown launch role {args.role!r}. Use local, master, watchdog, or all.")

    print("# AC production calibration campaign launcher.")
    print(f"# Role: {args.role}. Run set: {args.run_set} ({run_count} runs).")
    print(f"# Variants: {', '.join(variant.name for variant in selected)}")
    print(f"# Worker distribution: {', '.join(str(count) for count in worker_counts)}")
    print(f"# Max active runs: {args.max_active_runs}; max in-flight per run: {args.max_in_flight_per_run}")
    print(f"# Repo: {repo}")
    print()

    launched: list[dict[str, str | int]] = []
    hostname = socket.gethostname()
    for variant, worker_count in zip(selected, worker_counts):
        queue = f"{queue_root}_{variant.name}"
        logs_dir = f"{logs_root}\\{variant.name}"
        control = f"{control_root}_{variant.name}\\control"
        variant_log_dir = Path(logs_dir)
        if not variant_log_dir.is_absolute():
            variant_log_dir = repo / variant_log_dir
        if not args.dry_run:
            control_dir = Path(control)
            control_dir.mkdir(parents=True, exist_ok=True)
            default_file = control_dir / "default.txt"
            if not default_file.exists():
                default_file.write_text(str(args.workers), encoding="utf-8")
            if start_watchdog:
                (control_dir / f"{hostname}.txt").write_text(str(worker_count), encoding="utf-8")

        print(f"# --- {variant.name}: {variant.note} ---")
        print(f"# queue: {queue}")
        if start_master:
            master_env = _variant_env(os.environ, variant, master=True, spatial_sigma=args.spatial_sigma)
            master_log = variant_log_dir / "launcher_master_stdout.log"
            print("# master:")
            pid = _powershell_process(
                _common_master_args(args, variant, queue, logs_dir, control),
                repo,
                master_env,
                master_log,
                dry_run=args.dry_run,
            )
            if pid is not None:
                launched.append({"variant": variant.name, "role": "master", "pid": pid, "log": str(master_log)})
        if start_watchdog:
            watchdog_env = _variant_env(os.environ, variant, master=False, spatial_sigma=args.spatial_sigma)
            watchdog_log = variant_log_dir / "launcher_watchdog_stdout.log"
            print("# watchdog:")
            pid = _powershell_process(
                _watchdog_args(args, variant, queue, control, worker_count),
                repo,
                watchdog_env,
                watchdog_log,
                dry_run=args.dry_run,
            )
            if pid is not None:
                launched.append({"variant": variant.name, "role": "watchdog", "pid": pid, "log": str(watchdog_log)})
        print()

    if args.dry_run:
        print("Dry run only; no processes started.")
        return

    manifest_root = Path(args.logs_root)
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest = manifest_root / f"launcher_processes_{args.role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps(launched, indent=2), encoding="utf-8")
    print(f"Started {len(launched)} process(es).")
    print(f"Process manifest: {manifest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cmd = sub.add_parser("commands", help="Emit PowerShell master/watchdog commands.")
    cmd.add_argument("--variant", default="holiday4", help="Variant name, comma-list, holiday4, holiday5, spatial, global, or all.")
    cmd.add_argument(
        "--run-set",
        default="all",
        help="production, rollout, all, or an explicit space/comma-separated run list.",
    )
    cmd.add_argument("--repo", default=".")
    cmd.add_argument("--python", default=".\\.venv\\Scripts\\python.exe")
    cmd.add_argument("--queue-root", default=r"\\Moderskipet\Darsia_Queue\Kalibrering_AC_holiday")
    cmd.add_argument("--control-root", default=None)
    cmd.add_argument("--logs-root", default=r"Z:\Albus\Autokalibrering_log\holiday_2026")
    cmd.add_argument("--max-iters", type=int, default=800)
    cmd.add_argument("--warmup-iters", type=int, default=150)
    cmd.add_argument("--max-active-runs", type=int, default=3)
    cmd.add_argument("--max-in-flight-per-run", type=int, default=2)
    cmd.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Workers per variant watchdog. Use 3 for four concurrent variants on a 12-worker machine.",
    )
    cmd.add_argument("--sanity-every", type=int, default=100)
    cmd.add_argument("--spatial-sigma", type=float, default=6.0)
    cmd.add_argument("--max-tasks-per-worker", type=int, default=80)
    cmd.add_argument(
        "--resume-existing",
        action="store_true",
        help="Restart against the latest existing timestamped log folder per variant.",
    )
    cmd.add_argument("--exact-logs-dir", action="store_true")
    cmd.add_argument("--use-history", action="store_true")
    cmd.add_argument("--save-calibration", action="store_true")
    cmd.add_argument("--no-clear-queue", action="store_true")
    cmd.set_defaults(func=commands)

    launcher = sub.add_parser("launch", help="Start production/hardcase campaign processes in background.")
    launcher.add_argument("--variant", default="hardcase8", help="Variant name, comma-list, hardcase8, holiday4, holiday5, spatial, global, or all.")
    launcher.add_argument(
        "--run-set",
        default="hardcases",
        help="production, rollout, hardcases, all, or an explicit space/comma-separated run list.",
    )
    launcher.add_argument(
        "--role",
        default="local",
        help="local starts masters and watchdogs; watchdog starts only watchdogs; master starts only masters.",
    )
    launcher.add_argument("--repo", default=".")
    launcher.add_argument("--python", default=".\\.venv\\Scripts\\python.exe")
    launcher.add_argument("--queue-root", default=r"\\Moderskipet\Darsia_Queue\Kalibrering_AC_hardcase")
    launcher.add_argument("--control-root", default=None)
    launcher.add_argument("--logs-root", default=r"Z:\Albus\Autokalibrering_log\hardcase_2026")
    launcher.add_argument("--max-iters", type=int, default=800)
    launcher.add_argument("--warmup-iters", type=int, default=150)
    launcher.add_argument("--max-active-runs", type=int, default=3)
    launcher.add_argument("--max-in-flight-per-run", type=int, default=2)
    launcher.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Workers per variant watchdog. hardcase8 with 1 worker gives 8 workers per machine.",
    )
    launcher.add_argument(
        "--total-workers",
        type=int,
        default=0,
        help="Distribute this many workers across selected variants for this machine, e.g. 12 over hardcase8 -> 2,2,2,2,1,1,1,1.",
    )
    launcher.add_argument("--sanity-every", type=int, default=100)
    launcher.add_argument("--spatial-sigma", type=float, default=6.0)
    launcher.add_argument("--max-tasks-per-worker", type=int, default=80)
    launcher.add_argument("--resume-existing", action="store_true")
    launcher.add_argument("--exact-logs-dir", action="store_true")
    launcher.add_argument("--use-history", action="store_true")
    launcher.add_argument("--save-calibration", action="store_true")
    launcher.add_argument("--no-clear-queue", action="store_true")
    launcher.add_argument("--dry-run", action="store_true")
    launcher.set_defaults(func=launch)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
