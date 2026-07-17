"""Launch the frozen AC53/AC60 parameter-transfer diagnostic.

The campaign evaluates two fixed parameter anchors with four preprocessing
choices. Each target run receives exactly one distributed worker evaluation;
there are no Optuna or random warmup trials.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CAMPAIGN_ID = "anchor_transfer_20260716"
DEFAULT_AC53_PARAMS = Path(
    r"Z:\Albus\Autokalibrering_log\rollout_titration_l1"
    r"\facies1_perlabel1_warmup150_optuna800_parallel_20260617_0127"
    r"\final_full_scale_ac53.json"
)
DEFAULT_AC60_PARAMS = Path(
    r"Z:\Albus\Autokalibrering_log\production_titration_l1"
    r"\facies1_perlabel1_warmup150_optuna800_parallel_20260621_1411"
    r"\final_full_scale_ac60.json"
)
DEFAULT_RUNS = [
    "ac20",
    "ac24",
    "ac25",
    "ac26",
    "ac40",
    "ac44",
    "ac52",
    "ac53",
    "ac60",
]
OLD_CAMPAIGN_MARKERS = (
    "Kalibrering_AC_screen_step1",
    "Kalibrering_AC_geometry_screen",
)


@dataclass(frozen=True)
class Variant:
    name: str
    anchor: str
    params_file: Path
    template: bool = False
    photometric: bool = False


def _variants(args: argparse.Namespace) -> list[Variant]:
    sources = {
        "ac53": Path(args.ac53_params),
        "ac60": Path(args.ac60_params),
    }
    # Expensive photometric variants come first so --total-workers 12 assigns
    # their watchdogs two workers each; the four lighter variants get one.
    specifications = [
        ("photo", False, True),
        ("template_photo", True, True),
        ("none", False, False),
        ("template", True, False),
    ]
    variants: list[Variant] = []
    for suffix, template, photometric in specifications:
        for anchor in ("ac53", "ac60"):
            variants.append(
                Variant(
                    name=f"{anchor}_{suffix}",
                    anchor=anchor,
                    params_file=sources[anchor],
                    template=template,
                    photometric=photometric,
                )
            )
    return variants


def _ps_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _runs(args: argparse.Namespace) -> list[str]:
    selected = [
        token.strip().lower()
        for token in str(args.runs).replace(",", " ").split()
        if token.strip()
    ]
    if not selected:
        raise SystemExit("No target runs selected")
    forbidden = {"ac29", "ac51"}.intersection(selected)
    if forbidden:
        raise SystemExit(
            "Excluded run(s) cannot be used in the anchor screen: "
            + ", ".join(sorted(forbidden))
        )
    return selected


def _validate(variants: list[Variant]) -> None:
    for variant in variants:
        path = variant.params_file
        if not path.exists():
            raise FileNotFoundError(f"Missing {variant.anchor} parameter file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        params = payload.get("params") if isinstance(payload, dict) else None
        if not isinstance(params, dict) or not params:
            raise ValueError(f"No top-level params mapping in {path}")


def _worker_counts(total: int, count: int) -> list[int]:
    if total < count:
        raise SystemExit(
            f"--total-workers {total} is lower than the {count} variants; "
            "use at least one worker per variant"
        )
    base, extra = divmod(total, count)
    return [base + (1 if index < extra else 0) for index in range(count)]


def _paths(args: argparse.Namespace, variant: Variant) -> tuple[str, Path, str]:
    queue = (
        str(args.queue_root).rstrip("\\/")
        + f"_{args.campaign_id}_{variant.name}"
    )
    logs = Path(args.logs_root) / args.campaign_id / variant.name
    control = str(Path(queue) / "control")
    return queue, logs, control


def _master_command(
    args: argparse.Namespace,
    variant: Variant,
    queue: str,
    logs: Path,
    control: str,
) -> list[str]:
    return [
        f"& {_ps_quote(args.python)}",
        "scripts/distributed_auto_calibration_queue.py master",
        "--queue",
        _ps_quote(queue),
        "--runs",
        " ".join(_runs(args)),
        "--config-dir config_seg6/run_ac",
        "--logs-dir",
        _ps_quote(logs),
        "--exact-logs-dir",
        "--use-facies true",
        "--per-label true",
        "--objective-integral off",
        "--bounds-file config/bounds_seg6_coupled.json",
        "--fixed-params-file",
        _ps_quote(variant.params_file),
        "--no-save-calibration",
        "--max-iters 0",
        "--warmup-iters 0",
        "--run-mode parallel",
        "--max-active-runs 3",
        "--max-in-flight-per-run 1",
        "--control-dir",
        _ps_quote(control),
        "--sanity-every 0",
        "--quality-scale 1.0",
        "--quality-dtype float32",
        "--poll-seconds 1",
    ]


def _watchdog_command(
    args: argparse.Namespace,
    queue: str,
    control: str,
    workers: int,
) -> list[str]:
    return [
        f"& {_ps_quote(args.python)}",
        "scripts/distributed_auto_calibration_queue.py watchdog",
        "--queue",
        _ps_quote(queue),
        "--config-dir config_seg6/run_ac",
        "--use-facies true",
        "--per-label true",
        "--bounds-file config/bounds_seg6_coupled.json",
        "--control-dir",
        _ps_quote(control),
        "--workers",
        str(workers),
        "--worker-stall-seconds 900",
        "--max-tasks-per-worker 12",
        "--idle-exit-seconds 120",
        "--threads-per-worker 1",
    ]


def _environment(
    base: dict[str, str],
    variant: Variant,
    *,
    master: bool,
) -> dict[str, str]:
    env = dict(base)
    env["FFAC_TITRATION_FLASH"] = "on"
    for key in (
        "FFAC_STATIC_LIGHT_CORRECTION",
        "FFAC_STATIC_LIGHT_REFERENCE",
        "FFAC_STATIC_LIGHT_SPATIAL_SIGMA",
        "FFAC_COUPLE_AQ_GAS",
        "FFAC_TEMPLATE_REGISTRATION",
        "FFAC_TEMPLATE_REGISTRATION_MODE",
        "FFAC_TEMPLATE_REGISTRATION_STRICT",
        "FFAC_TEMPLATE_REGISTRATION_SCALE",
        "FFAC_TEMPLATE_REGISTRATION_MAX_COL",
        "FFAC_TEMPLATE_REGISTRATION_MAX_FEATURES",
        "FFAC_TEMPLATE_REGISTRATION_KEEP_MATCHES",
        "FFAC_TEMPLATE_REGISTRATION_MIN_INLIER_FRAC",
        "FFAC_TEMPLATE_REGISTRATION_MAX_SHIFT_PX",
        "FFAC_TEMPLATE_REGISTRATION_MIN_SCALE",
        "FFAC_TEMPLATE_REGISTRATION_MAX_SCALE",
        "FFAC_PHOTOMETRIC_ANCHOR_RUN",
        "FFAC_PHOTOMETRIC_ANCHOR_STRICT",
        "FFAC_PHOTOMETRIC_ANCHOR_STRIDE",
        "FFAC_PHOTOMETRIC_ANCHOR_GAIN_MIN",
        "FFAC_PHOTOMETRIC_ANCHOR_GAIN_MAX",
        "FFAC_MASTER_LIGHT_CONTEXT",
        "FFAC_TITRATION_RECIPE",
    ):
        env.pop(key, None)
    if variant.template:
        env["FFAC_TEMPLATE_REGISTRATION"] = "ac14_template"
        env["FFAC_TEMPLATE_REGISTRATION_MODE"] = "partial_affine"
        env["FFAC_TEMPLATE_REGISTRATION_STRICT"] = "on"
    if variant.photometric:
        env["FFAC_PHOTOMETRIC_ANCHOR_RUN"] = variant.anchor
        env["FFAC_PHOTOMETRIC_ANCHOR_STRICT"] = "on"
        env["FFAC_PHOTOMETRIC_ANCHOR_STRIDE"] = "8"
        env["FFAC_PHOTOMETRIC_ANCHOR_GAIN_MIN"] = "0.5"
        env["FFAC_PHOTOMETRIC_ANCHOR_GAIN_MAX"] = "2.0"
    if master:
        env["FFAC_MASTER_LIGHT_CONTEXT"] = "on"
    return env


def _environment_lines(variant: Variant, *, master: bool) -> list[str]:
    lines = [
        "$env:FFAC_TITRATION_FLASH = 'on'",
        "Remove-Item Env:\\FFAC_STATIC_LIGHT_CORRECTION -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_STATIC_LIGHT_REFERENCE -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_STATIC_LIGHT_SPATIAL_SIGMA -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_COUPLE_AQ_GAS -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TITRATION_RECIPE -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_SCALE -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_MAX_COL -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_MAX_FEATURES -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_KEEP_MATCHES -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_MIN_INLIER_FRAC -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_MAX_SHIFT_PX -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_MIN_SCALE -ErrorAction SilentlyContinue",
        "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_MAX_SCALE -ErrorAction SilentlyContinue",
    ]
    if variant.template:
        lines.extend(
            [
                "$env:FFAC_TEMPLATE_REGISTRATION = 'ac14_template'",
                "$env:FFAC_TEMPLATE_REGISTRATION_MODE = 'partial_affine'",
                "$env:FFAC_TEMPLATE_REGISTRATION_STRICT = 'on'",
            ]
        )
    else:
        lines.extend(
            [
                "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION -ErrorAction SilentlyContinue",
                "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_MODE -ErrorAction SilentlyContinue",
                "Remove-Item Env:\\FFAC_TEMPLATE_REGISTRATION_STRICT -ErrorAction SilentlyContinue",
            ]
        )
    if variant.photometric:
        lines.extend(
            [
                f"$env:FFAC_PHOTOMETRIC_ANCHOR_RUN = '{variant.anchor}'",
                "$env:FFAC_PHOTOMETRIC_ANCHOR_STRICT = 'on'",
                "$env:FFAC_PHOTOMETRIC_ANCHOR_STRIDE = '8'",
                "$env:FFAC_PHOTOMETRIC_ANCHOR_GAIN_MIN = '0.5'",
                "$env:FFAC_PHOTOMETRIC_ANCHOR_GAIN_MAX = '2.0'",
            ]
        )
    else:
        lines.extend(
            [
                "Remove-Item Env:\\FFAC_PHOTOMETRIC_ANCHOR_RUN -ErrorAction SilentlyContinue",
                "Remove-Item Env:\\FFAC_PHOTOMETRIC_ANCHOR_STRICT -ErrorAction SilentlyContinue",
                "Remove-Item Env:\\FFAC_PHOTOMETRIC_ANCHOR_STRIDE -ErrorAction SilentlyContinue",
                "Remove-Item Env:\\FFAC_PHOTOMETRIC_ANCHOR_GAIN_MIN -ErrorAction SilentlyContinue",
                "Remove-Item Env:\\FFAC_PHOTOMETRIC_ANCHOR_GAIN_MAX -ErrorAction SilentlyContinue",
            ]
        )
    if master:
        lines.append("$env:FFAC_MASTER_LIGHT_CONTEXT = 'on'")
    else:
        lines.append("Remove-Item Env:\\FFAC_MASTER_LIGHT_CONTEXT -ErrorAction SilentlyContinue")
    return lines


def _start_process(
    command: list[str],
    *,
    repo: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> int | None:
    block = " ".join(command)
    print(block)
    if dry_run:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output = log_path.open("ab")
    try:
        process = subprocess.Popen(
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
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    finally:
        output.close()
    return int(process.pid)


def commands(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    variants = _variants(args)
    _validate(variants)
    counts = _worker_counts(args.total_workers, len(variants))
    role = args.role.lower()
    print(f"# Frozen anchor transfer: {args.campaign_id}")
    print(f"# Runs: {', '.join(_runs(args))}")
    print(f"# Worker distribution on this machine: {counts}")
    print(f"cd {_ps_quote(repo)}")
    print()
    for variant, worker_count in zip(variants, counts):
        queue, logs, control = _paths(args, variant)
        print(f"# --- {variant.name} ---")
        if role in {"local", "all", "master", "masters"}:
            print("# master")
            print("\n".join(_environment_lines(variant, master=True)))
            print(" ".join(_master_command(args, variant, queue, logs, control)))
            print()
        if role in {"local", "all", "watchdog", "watchdogs"}:
            print("# watchdog")
            print("\n".join(_environment_lines(variant, master=False)))
            print(" ".join(_watchdog_command(args, queue, control, worker_count)))
            print()


def launch(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.allow_old_processes:
        try:
            import psutil
        except ImportError as exc:
            raise SystemExit("psutil is required for launch preflight") from exc
        old_processes = _old_campaign_roots(psutil)
        if old_processes:
            raise SystemExit(
                f"Found {len(old_processes)} old screen campaign process(es). "
                "Wait for the old run to finish, then run 'stop-old' before launch."
            )

    repo = Path(args.repo).resolve()
    variants = _variants(args)
    _validate(variants)
    counts = _worker_counts(args.total_workers, len(variants))
    role = args.role.lower()
    start_master = role in {"local", "all", "master", "masters"}
    start_watchdog = role in {"local", "all", "watchdog", "watchdogs"}
    if not start_master and not start_watchdog:
        raise SystemExit("--role must be local, master, or watchdog")

    host = socket.gethostname()
    launched: list[dict[str, str | int]] = []
    print(f"# Campaign {args.campaign_id}; role={role}; host={host}")
    print(f"# Worker distribution: {counts}")
    for variant, worker_count in zip(variants, counts):
        queue, logs, control = _paths(args, variant)
        if not args.dry_run:
            Path(control).mkdir(parents=True, exist_ok=True)
        print(f"# --- {variant.name} ---")
        if start_master:
            pid = _start_process(
                _master_command(args, variant, queue, logs, control),
                repo=repo,
                env=_environment(os.environ, variant, master=True),
                log_path=logs / "launcher_master_stdout.log",
                dry_run=args.dry_run,
            )
            if pid is not None:
                launched.append(
                    {
                        "variant": variant.name,
                        "role": "master",
                        "pid": pid,
                        "log": str(logs / "launcher_master_stdout.log"),
                    }
                )
        if start_watchdog:
            pid = _start_process(
                _watchdog_command(args, queue, control, worker_count),
                repo=repo,
                env=_environment(os.environ, variant, master=False),
                log_path=logs / f"launcher_watchdog_{host}.log",
                dry_run=args.dry_run,
            )
            if pid is not None:
                launched.append(
                    {
                        "variant": variant.name,
                        "role": "watchdog",
                        "pid": pid,
                        "log": str(logs / f"launcher_watchdog_{host}.log"),
                    }
                )
        print()

    if not args.dry_run:
        manifest_dir = Path(args.logs_root) / args.campaign_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = manifest_dir / f"launcher_processes_{host}.json"
        manifest.write_text(json.dumps(launched, indent=2), encoding="utf-8")
        print(f"Started {len(launched)} process(es). Manifest: {manifest}")


def status(args: argparse.Namespace) -> None:
    variants = _variants(args)
    rows: list[dict[str, str | int]] = []
    for variant in variants:
        queue, logs, _control = _paths(args, variant)
        queue_path = Path(queue)
        row: dict[str, str | int] = {"variant": variant.name}
        for folder in ("pending", "in_progress", "results", "done", "failed", "heartbeats"):
            path = queue_path / folder
            row[folder] = len(list(path.glob("*"))) if path.exists() else 0
        row["complete"] = int((queue_path / "master_complete.json").exists())
        row["finals"] = len(list(logs.glob("final_full_scale_ac*.json"))) if logs.exists() else 0
        rows.append(row)
    print(json.dumps(rows, indent=2))


def _old_campaign_roots(psutil_module: object) -> dict[int, object]:
    current_pid = os.getpid()
    roots: dict[int, object] = {}
    for process in psutil_module.process_iter(["pid", "name", "cmdline"]):
        if process.pid == current_pid:
            continue
        try:
            command_line = " ".join(process.info.get("cmdline") or ())
        except (psutil_module.AccessDenied, psutil_module.NoSuchProcess):
            continue
        if any(marker.lower() in command_line.lower() for marker in OLD_CAMPAIGN_MARKERS):
            roots[process.pid] = process
    return roots


def stop_old(args: argparse.Namespace) -> None:
    """Stop complete process trees belonging to superseded screen campaigns."""
    try:
        import psutil
    except ImportError as exc:
        raise SystemExit("psutil is required for stop-old") from exc

    current_pid = os.getpid()
    roots = _old_campaign_roots(psutil)

    targets: dict[int, psutil.Process] = dict(roots)
    for process in roots.values():
        try:
            for child in process.children(recursive=True):
                if child.pid != current_pid:
                    targets[child.pid] = child
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    if not targets:
        print("No old screen_step1/geometry_screen processes found.")
        return

    def _depth(process: psutil.Process) -> int:
        depth = 0
        try:
            parent = process.parent()
            while parent is not None and parent.pid in targets:
                depth += 1
                parent = parent.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        return depth

    ordered = sorted(targets.values(), key=_depth, reverse=True)
    print(
        f"Found {len(roots)} matching process(es) and "
        f"{len(targets) - len(roots)} descendant(s)."
    )
    for process in ordered:
        try:
            print(f"{process.pid:>7}  {process.name():<18}  {' '.join(process.cmdline())[:180]}")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    if args.dry_run:
        print("Dry run: no processes stopped.")
        return

    for process in ordered:
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    _gone, alive = psutil.wait_procs(ordered, timeout=5)
    for process in alive:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    _gone, alive = psutil.wait_procs(alive, timeout=5)
    if alive:
        remaining = ", ".join(str(process.pid) for process in alive)
        raise SystemExit(f"Could not stop process PID(s): {remaining}")
    print(f"Stopped {len(targets)} old campaign process(es).")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--python", default=r".\.venv\Scripts\python.exe")
    parser.add_argument(
        "--runs",
        default=" ".join(DEFAULT_RUNS),
        help="Space- or comma-separated targets; AC29 and AC51 are rejected.",
    )
    parser.add_argument("--ac53-params", default=str(DEFAULT_AC53_PARAMS))
    parser.add_argument("--ac60-params", default=str(DEFAULT_AC60_PARAMS))
    parser.add_argument(
        "--queue-root",
        default=r"\\Moderskipet\Darsia_Queue\Kalibrering_AC",
    )
    parser.add_argument(
        "--logs-root",
        default=r"Z:\Albus\Autokalibrering_log",
    )
    parser.add_argument("--total-workers", type=int, default=12)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command_parser = subparsers.add_parser("commands", help="Print PowerShell commands")
    _add_common_args(command_parser)
    command_parser.add_argument("--role", default="local")
    command_parser.set_defaults(func=commands)

    launch_parser = subparsers.add_parser("launch", help="Start background processes")
    _add_common_args(launch_parser)
    launch_parser.add_argument("--role", default="local")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.add_argument(
        "--allow-old-processes",
        action="store_true",
        help="Bypass the guard against superseded screen campaign processes.",
    )
    launch_parser.set_defaults(func=launch)

    status_parser = subparsers.add_parser("status", help="Report queue and final counts")
    _add_common_args(status_parser)
    status_parser.set_defaults(func=status)

    stop_parser = subparsers.add_parser(
        "stop-old",
        help="Stop process trees from the superseded screen campaigns",
    )
    stop_parser.add_argument("--dry-run", action="store_true")
    stop_parser.set_defaults(func=stop_old)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
