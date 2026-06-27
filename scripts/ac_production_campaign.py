"""Emit commands for full AC production calibration campaigns.

The AC53 testbed is intentionally narrow. This helper emits repeatable
PowerShell commands for the production-sized runs we use on the shared
Moderskipet queue.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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
    if args.no_clear_queue:
        command.append("--no-clear-queue")
    return command


def _watchdog_args(args: argparse.Namespace, variant: Variant, queue: str, control: str) -> list[str]:
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
        str(args.workers),
        "--worker-stall-seconds 600",
    ]
    if args.max_tasks_per_worker > 0:
        command.extend(["--max-tasks-per-worker", str(args.max_tasks_per_worker)])
    return command


def commands(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    selected = _select_variants(args.variant)
    run_count = len(_select_runs(args.run_set))
    queue_root = args.queue_root.rstrip("\\/")
    logs_root = args.logs_root.rstrip("\\/")
    control_root = args.control_root.rstrip("\\/") if args.control_root else queue_root

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
        control = f"{control_root}_{variant.name}\\control"
        print(f"# --- {variant.name}: {variant.note} ---")
        print("# master terminal:")
        for line in _env_lines(variant, args.spatial_sigma):
            print(line)
        print(" ".join(_common_master_args(args, variant, queue, logs_dir, control)))
        print()
        print("# watchdog terminal on each machine:")
        for line in _env_lines(variant, args.spatial_sigma):
            print(line)
        print(" ".join(_watchdog_args(args, variant, queue, control)))
        print()


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
    cmd.add_argument("--max-in-flight-per-run", type=int, default=3)
    cmd.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Workers per variant watchdog. Use 3 for four concurrent variants on a 12-worker machine.",
    )
    cmd.add_argument("--sanity-every", type=int, default=100)
    cmd.add_argument("--spatial-sigma", type=float, default=6.0)
    cmd.add_argument("--max-tasks-per-worker", type=int, default=80)
    cmd.add_argument("--save-calibration", action="store_true")
    cmd.add_argument("--no-clear-queue", action="store_true")
    cmd.set_defaults(func=commands)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
