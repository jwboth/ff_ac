"""AC53 calibration log diagnostics and command generator.

This script does not run DarSIA. It only reads existing queue CSV logs and
prints comparable metrics, or emits PowerShell command blocks for one-variant
AC53 calibration tests.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


def _default_calibration_log_root() -> Path:
    env_root = os.environ.get("FFAC_CALIBRATION_LOG_ROOT")
    if env_root:
        return Path(env_root)
    preferred = Path(r"Z:\Albus\Autokalibrering_log")
    try:
        if Path(preferred.drive + "\\").exists():
            return preferred
    except Exception:
        pass
    return Path("logs")


def _variant_logs_dir(args: argparse.Namespace, variant: "Variant") -> Path:
    root = Path(args.logs_root) if getattr(args, "logs_root", None) else _default_calibration_log_root()
    return root / "ac53_testbed" / variant.name


def _default_search_roots(raw_logs_dir: str | None) -> list[Path]:
    if raw_logs_dir:
        return [Path(raw_logs_dir)]
    root = _default_calibration_log_root()
    candidates = [
        root / "ac53_testbed",
        Path("logs") / "ac53_testbed",
        root,
        Path("logs"),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        unique.append(path)
        seen.add(key)
    return unique


def _rel_to_any(path: Path, roots: Iterable[Path]) -> Path:
    for root in roots:
        try:
            return path.parent.relative_to(root)
        except ValueError:
            continue
    return path.parent


@dataclass(frozen=True)
class Metrics:
    ratios: dict[float, float]
    raw_l1: float
    injection_l1: float
    plateau_l1: float
    plateau_tv_norm: float
    plateau_mean: float
    plateau_min: float
    plateau_max: float

    def ratio_at(self, target: float) -> float:
        if not self.ratios:
            return math.nan
        key = min(self.ratios, key=lambda t: abs(t - target))
        return self.ratios[key]

    @property
    def growth_2p5_to_48(self) -> float:
        return self.ratio_at(48.0) - self.ratio_at(2.497)


@dataclass
class Trial:
    source: Path
    iteration: str
    objective: float
    metrics: Metrics
    settings: dict


@dataclass(frozen=True)
class Variant:
    name: str
    objective_integral: str
    bounds_file: str
    titration: bool = False
    static_light: str = "off"
    couple_aq_gas: bool = False
    note: str = ""


VARIANTS = [
    Variant(
        "simple_l1",
        "off",
        "config/bounds_seg6_coupled.json",
        note="control: colour-on SimpleFlash, point-wise L1",
    ),
    Variant(
        "simple_drift025",
        "drift:0.25",
        "config/bounds_seg6_coupled.json",
        note="weak closed-cell plateau drift penalty",
    ),
    Variant(
        "simple_drift050",
        "drift:0.5",
        "config/bounds_seg6_coupled.json",
        note="medium closed-cell plateau drift penalty",
    ),
    Variant(
        "simple_drift100",
        "drift",
        "config/bounds_seg6_coupled.json",
        note="strong closed-cell plateau drift penalty",
    ),
    Variant(
        "titration_l1",
        "off",
        "config/bounds_seg6_coupled.json",
        titration=True,
        note="BTB/carbonate aqueous transfer, point-wise L1",
    ),
    Variant(
        "titration_drift025",
        "drift:0.25",
        "config/bounds_seg6_coupled.json",
        titration=True,
        note="BTB/carbonate transfer plus weak drift penalty",
    ),
    Variant(
        "titration_drift050",
        "drift:0.5",
        "config/bounds_seg6_coupled.json",
        titration=True,
        note="BTB/carbonate transfer plus medium drift penalty",
    ),
    Variant(
        "titration_tight_l1",
        "off",
        "config/bounds_seg6_titration.json",
        titration=True,
        note="BTB/carbonate transfer with titration-specific tighter bounds",
    ),
    Variant(
        "titration_tight_drift025",
        "drift:0.25",
        "config/bounds_seg6_titration.json",
        titration=True,
        note="titration-specific tighter bounds plus weak drift penalty",
    ),
    Variant(
        "titration_static_l1",
        "off",
        "config/bounds_seg6_titration.json",
        titration=True,
        static_light="blue-gain",
        note="titration-specific bounds plus static FluidFlower light gain",
    ),
    Variant(
        "titration_static_drift025",
        "drift:0.25",
        "config/bounds_seg6_titration.json",
        titration=True,
        static_light="blue-gain",
        note="static FluidFlower light gain plus weak drift penalty",
    ),
    Variant(
        "titration_coupled_l1",
        "off",
        "config/bounds_seg6_titration.json",
        titration=True,
        couple_aq_gas=True,
        note="tighter titration bounds with gas onset coupled to aqueous saturation",
    ),
    Variant(
        "titration_coupled_drift025",
        "drift:0.25",
        "config/bounds_seg6_titration.json",
        titration=True,
        couple_aq_gas=True,
        note="coupled aq/gas onset plus weak drift penalty",
    ),
    Variant(
        "flashwide_l1",
        "off",
        "config/bounds_seg6_flashwide.json",
        note="diagnostic: wider flash ramps, unchanged signal bounds",
    ),
]

VARIANT_SETS = {
    "core4": [
        "simple_l1",
        "simple_drift025",
        "titration_l1",
        "titration_drift025",
    ],
    "next4": [
        "titration_tight_l1",
        "titration_tight_drift025",
        "titration_static_l1",
        "titration_static_drift025",
    ],
    "next6": [
        "titration_tight_l1",
        "titration_tight_drift025",
        "titration_static_l1",
        "titration_static_drift025",
        "titration_coupled_l1",
        "titration_coupled_drift025",
    ],
    "all": [variant.name for variant in VARIANTS],
}


def _parse_metrics(raw: str) -> Metrics | None:
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return None

    rows: list[tuple[float, float, float, float]] = []
    for key, value in parsed.items():
        try:
            time_h = float(str(key).removesuffix("h"))
            injected = float(value.get("injected_full", 0.0) or 0.0)
            detected = float(value.get("total_full", 0.0) or 0.0)
        except Exception:
            continue
        ratio = detected / injected if injected else math.nan
        rows.append((time_h, injected, detected, ratio))

    if not rows:
        return None
    rows.sort()

    plateau = [row for row in rows if row[0] >= 2.497 - 1e-3]
    injection = [row for row in rows if row[0] <= 2.497 + 1e-3]
    plateau_ratios = [row[3] for row in plateau if math.isfinite(row[3])]
    injected_final = plateau[0][1] if plateau else rows[-1][1]
    plateau_tv = sum(abs(plateau[i][2] - plateau[i - 1][2]) for i in range(1, len(plateau)))

    return Metrics(
        ratios={time_h: ratio for time_h, _, _, ratio in rows},
        raw_l1=sum(abs(detected - injected) for _, injected, detected, _ in rows),
        injection_l1=sum(abs(detected - injected) for _, injected, detected, _ in injection),
        plateau_l1=sum(abs(detected - injected) for _, injected, detected, _ in plateau),
        plateau_tv_norm=plateau_tv / injected_final if injected_final else math.nan,
        plateau_mean=mean(plateau_ratios) if plateau_ratios else math.nan,
        plateau_min=min(plateau_ratios) if plateau_ratios else math.nan,
        plateau_max=max(plateau_ratios) if plateau_ratios else math.nan,
    )


def _parse_settings(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def iter_trials(path: Path) -> Iterable[Trial]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") and row["status"] != "ok":
                continue
            metrics = _parse_metrics(row.get("metrics", ""))
            if metrics is None:
                continue
            try:
                objective = float(row.get("objective", "nan"))
            except ValueError:
                objective = math.nan
            yield Trial(
                source=path,
                iteration=str(row.get("iter", "")),
                objective=objective,
                metrics=metrics,
                settings=_parse_settings(row.get("settings", "")),
            )


def _fmt_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _score(trial: Trial, criterion: str) -> float:
    metrics = trial.metrics
    r25 = metrics.ratio_at(2.497)
    r48 = metrics.ratio_at(48.0)
    nonzero_guard = 0.0 if r25 >= 0.05 or r48 >= 0.05 else 1000.0

    if criterion == "objective":
        return trial.objective
    if criterion == "raw_l1":
        return metrics.raw_l1
    if criterion == "end_injection":
        return abs(r25 - 1.0)
    if criterion == "late_48h":
        return abs(r48 - 1.0)
    if criterion == "plateau_level_drift":
        return nonzero_guard + abs(metrics.plateau_mean - 1.0) + metrics.plateau_tv_norm
    if criterion == "balanced":
        return (
            nonzero_guard
            + abs(r25 - 1.0)
            + abs(r48 - 1.0)
            + 0.5 * metrics.plateau_tv_norm
        )
    if criterion == "plateau_balanced":
        return (
            nonzero_guard
            + 0.35 * abs(r25 - 1.0)
            + 0.20 * abs(r48 - 1.0)
            + 0.25 * abs(metrics.plateau_mean - 1.0)
            + 0.20 * metrics.plateau_tv_norm
            + 0.10 * (metrics.plateau_max - metrics.plateau_min)
        )
    if criterion == "production":
        return (
            nonzero_guard
            + metrics.plateau_l1
            + 0.25 * metrics.injection_l1
            + 0.10 * metrics.plateau_tv_norm
        )
    if criterion == "late_weighted":
        return metrics.plateau_l1 + 0.25 * metrics.injection_l1
    raise ValueError(f"unknown criterion: {criterion}")


def _print_trial(prefix: str, trial: Trial, score: float | None = None) -> None:
    metrics = trial.metrics
    bits = [
        prefix,
        f"iter={trial.iteration:>4s}",
        f"obj={trial.objective:.6g}",
    ]
    if score is not None:
        bits.append(f"score={score:.6g}")
    bits.extend(
        [
            f"r2.5={_fmt_float(metrics.ratio_at(2.497))}",
            f"r9={_fmt_float(metrics.ratio_at(9.417))}",
            f"r22={_fmt_float(metrics.ratio_at(22.25))}",
            f"r48={_fmt_float(metrics.ratio_at(48.0))}",
            f"growth={_fmt_float(metrics.growth_2p5_to_48)}",
            f"tv={_fmt_float(metrics.plateau_tv_norm)}",
            f"plateau={_fmt_float(metrics.plateau_min)}-{_fmt_float(metrics.plateau_max)}",
        ]
    )
    print(" ".join(bits))


def summarize(args: argparse.Namespace) -> None:
    logs_roots = _default_search_roots(args.logs_dir)
    files = _calibration_log_files(logs_roots, args.run)
    if not files:
        raise SystemExit(
            f"No tmp_auto_calibration_{args.run}.csv or auto_calibration_{args.run}.csv "
            f"files found below {', '.join(str(root) for root in logs_roots)}"
        )

    criteria = [
        "objective",
        "raw_l1",
        "end_injection",
        "late_48h",
        "plateau_level_drift",
        "balanced",
        "plateau_balanced",
        "production",
        "late_weighted",
    ]

    print(f"{args.run.upper()} existing log summary")
    for path in files:
        trials = list(iter_trials(path))
        if not trials:
            continue
        best = min(trials, key=lambda trial: trial.objective)
        settings = best.settings
        rel = _rel_to_any(path, logs_roots)
        print(
            f"{rel} rows={len(trials)} "
            f"mode={settings.get('objective_integral', '?')} "
            f"titr={settings.get('titration_flash', '?') or 'off'} "
            f"static={settings.get('static_light_correction', '?') or 'off'} "
            f"couple={settings.get('couple_aq_gas', '?') or 'off'} "
            f"qscale={settings.get('quality_scale', '?')} "
            f"qdtype={settings.get('quality_dtype', '?')} "
            f"bounds={settings.get('bounds_file', '?')}"
        )
        _print_trial("  best", best)

    if not args.winners:
        return

    selected = [item.strip() for item in args.selected.split(",") if item.strip()]
    selected_files = [
        path for path in files if not selected or any(pattern in str(path.parent) for pattern in selected)
    ]
    print("\nAlternative winners")
    for path in selected_files:
        trials = list(iter_trials(path))
        if not trials:
            continue
        rel = _rel_to_any(path, logs_roots)
        print(f"\n## {rel} rows={len(trials)}")
        for criterion in criteria:
            winner = min(trials, key=lambda trial: _score(trial, criterion))
            _print_trial(f"{criterion:20s}", winner, _score(winner, criterion))


def _calibration_log_files(logs_roots: Iterable[Path], run: str) -> list[Path]:
    files: set[Path] = set()
    for logs_dir in logs_roots:
        if not logs_dir.exists():
            continue
        files.update(logs_dir.rglob(f"tmp_auto_calibration_{run}.csv"))
        files.update(logs_dir.rglob(f"auto_calibration_{run}.csv"))
    return sorted(
        files,
        key=lambda path: path.stat().st_mtime,
    )


def _subset_ratio_mae(metrics: Metrics, times: list[float]) -> float:
    values = [metrics.ratios[t] for t in times if math.isfinite(metrics.ratios.get(t, math.nan))]
    if not values:
        return math.inf
    return mean(abs(value - 1.0) for value in values)


def cross_validate(args: argparse.Namespace) -> None:
    logs_roots = _default_search_roots(args.logs_dir)
    files = _calibration_log_files(logs_roots, args.run)
    if not files:
        raise SystemExit(
            f"No tmp_auto_calibration_{args.run}.csv or auto_calibration_{args.run}.csv "
            f"files found below {', '.join(str(root) for root in logs_roots)}"
        )
    selected = [item.strip() for item in args.selected.split(",") if item.strip()]
    files = [path for path in files if not selected or any(pattern in str(path.parent) for pattern in selected)]
    if not files:
        raise SystemExit("No matching calibration logs after --selected filtering.")

    print(f"{args.run.upper()} timepoint cross-validation")
    for path in files:
        trials = list(iter_trials(path))
        if not trials:
            continue
        times = sorted(trials[0].metrics.ratios)
        even_times = [time for idx, time in enumerate(times) if idx % 2 == 0]
        odd_times = [time for idx, time in enumerate(times) if idx % 2 == 1]
        splits = [
            ("even->odd", even_times, odd_times),
            ("odd->even", odd_times, even_times),
        ]
        rel = _rel_to_any(path, logs_roots)
        print(f"\n## {rel} rows={len(trials)} source={path.name}")
        for split_name, train_times, eval_times in splits:
            winner = min(trials, key=lambda trial: _subset_ratio_mae(trial.metrics, train_times))
            train_mae = _subset_ratio_mae(winner.metrics, train_times)
            eval_mae = _subset_ratio_mae(winner.metrics, eval_times)
            print(
                f"{split_name:9s} iter={winner.iteration:>4s} "
                f"train_mae={_fmt_float(train_mae, 4)} "
                f"eval_mae={_fmt_float(eval_mae, 4)} "
                f"r2.5={_fmt_float(winner.metrics.ratio_at(2.497))} "
                f"r35={_fmt_float(winner.metrics.ratio_at(35.167))} "
                f"r48={_fmt_float(winner.metrics.ratio_at(48.0))} "
                f"plateau={_fmt_float(winner.metrics.plateau_min)}-{_fmt_float(winner.metrics.plateau_max)}"
            )


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


def _variant_env_lines(variant: Variant) -> list[str]:
    lines = [
        "$env:FFAC_TITRATION_FLASH = 'on'"
        if variant.titration
        else "Remove-Item Env:\\FFAC_TITRATION_FLASH -ErrorAction SilentlyContinue",
        f"$env:FFAC_STATIC_LIGHT_CORRECTION = '{variant.static_light}'"
        if variant.static_light != "off"
        else "Remove-Item Env:\\FFAC_STATIC_LIGHT_CORRECTION -ErrorAction SilentlyContinue",
        "$env:FFAC_COUPLE_AQ_GAS = 'on'"
        if variant.couple_aq_gas
        else "Remove-Item Env:\\FFAC_COUPLE_AQ_GAS -ErrorAction SilentlyContinue",
    ]
    if not variant.titration:
        lines.append("Remove-Item Env:\\FFAC_TITRATION_RECIPE -ErrorAction SilentlyContinue")
    return lines


def emit_commands(args: argparse.Namespace) -> None:
    selected = _select_variants(args.variant)
    python = args.python
    repo = Path(args.repo).resolve()
    queue_root = args.queue_root.rstrip("\\/")
    control_root = args.control_root.rstrip("\\/") if args.control_root else queue_root

    print("# Variants can run in parallel when they use separate queues/logs and --no-save-calibration.")
    print("# Colour-on prep is only needed if ac53 has not already been prepared with colour on.")
    print(f"cd {_ps_quote(repo)}")
    if args.include_prep:
        print(".\\prep_color_seg6.ps1 -Color on -Runs ac53")
    print("New-Item -ItemType Directory -Force -Path 'config_seg6/run_ac/.color_state' | Out-Null")
    print("Set-Content -LiteralPath 'config_seg6/run_ac/.color_state/ac53.txt' -Value 'on'")
    print("New-Item -ItemType Directory -Force -Path 'config_seg6/run_ac/.titration_state' | Out-Null")
    print("Remove-Item -LiteralPath 'config_seg6/run_ac/.titration_state/ac53.txt' -ErrorAction SilentlyContinue")
    print("# Titration is controlled per terminal with FFAC_TITRATION_FLASH below, not with a shared stamp.")
    print()

    for variant in selected:
        queue = f"{queue_root}_{variant.name}"
        control = f"{control_root}_{variant.name}\\control"
        logs_dir = str(_variant_logs_dir(args, variant))
        print(f"# --- {variant.name}: {variant.note} ---")
        print("# master terminal:")
        for env_line in _variant_env_lines(variant):
            print(env_line)
        if variant.titration:
            print("# Optional recipe override:")
            print("# $env:FFAC_TITRATION_RECIPE = '1.25,0.726,34'")
        print(
            " ".join(
                [
                    f"& {python}",
                    "scripts/distributed_auto_calibration_queue.py master",
                    "--queue",
                    _ps_quote(queue),
                    "--runs ac53",
                    "--config-dir config_seg6/run_ac",
                    "--logs-dir",
                    _ps_quote(logs_dir),
                    "--use-facies true",
                    "--per-label true",
                    "--objective-integral",
                    _ps_quote(variant.objective_integral),
                    "--bounds-file",
                    _ps_quote(variant.bounds_file),
                    *(["--no-save-calibration"] if not args.save_calibration else []),
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
        )
        print("# watchdog command for a second terminal:")
        for env_line in _variant_env_lines(variant):
            print(env_line)
        if variant.titration:
            print("# Optional recipe override:")
            print("# $env:FFAC_TITRATION_RECIPE = '1.25,0.726,34'")
        print(
            " ".join(
                [
                    f"& {python}",
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
            )
        )
        print()


def _master_command(args: argparse.Namespace, variant: Variant, queue: str, control: str, logs_dir: str) -> list[str]:
    command = [
        args.python,
        "scripts/distributed_auto_calibration_queue.py",
        "master",
        "--queue",
        queue,
        "--runs",
        "ac53",
        "--config-dir",
        "config_seg6/run_ac",
        "--logs-dir",
        logs_dir,
        "--use-facies",
        "true",
        "--per-label",
        "true",
        "--objective-integral",
        variant.objective_integral,
        "--bounds-file",
        variant.bounds_file,
        "--max-iters",
        str(args.max_iters),
        "--warmup-iters",
        str(args.warmup_iters),
        "--run-mode",
        "parallel",
        "--max-in-flight-per-run",
        str(args.max_in_flight_per_run),
        "--control-dir",
        control,
        "--sanity-every",
        str(args.sanity_every),
        "--sanity-scale",
        "1.00",
        "--quality-dtype",
        "float32",
    ]
    if not args.save_calibration:
        command.insert(command.index("--max-iters"), "--no-save-calibration")
    return command


def _watchdog_command(args: argparse.Namespace, queue: str, control: str) -> list[str]:
    return [
        args.python,
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
        "--control-dir",
        control,
        "--workers",
        str(args.workers),
        "--worker-stall-seconds",
        "600",
    ]


def launch(args: argparse.Namespace) -> None:
    selected = _select_variants(args.variant)
    repo = Path(args.repo).resolve()
    queue_root = args.queue_root.rstrip("\\/")
    control_root = args.control_root.rstrip("\\/") if args.control_root else queue_root

    color_state = repo / "config_seg6" / "run_ac" / ".color_state"
    titration_state = repo / "config_seg6" / "run_ac" / ".titration_state"
    if not args.dry_run:
        color_state.mkdir(parents=True, exist_ok=True)
        (color_state / "ac53.txt").write_text("on\n", encoding="utf-8")
        titration_state.mkdir(parents=True, exist_ok=True)
        try:
            (titration_state / "ac53.txt").unlink()
        except FileNotFoundError:
            pass

    launched: list[dict[str, str | int]] = []
    for variant in selected:
        queue = f"{queue_root}_{variant.name}"
        control = f"{control_root}_{variant.name}\\control"
        logs_dir = str(_variant_logs_dir(args, variant))
        logs_dir_path = Path(logs_dir)
        variant_log_dir = logs_dir_path if logs_dir_path.is_absolute() else repo / logs_dir_path
        master_log = variant_log_dir / "launcher_master_stdout.log"
        watchdog_log = variant_log_dir / "launcher_watchdog_stdout.log"
        env = os.environ.copy()
        if variant.titration:
            env["FFAC_TITRATION_FLASH"] = "on"
        else:
            env.pop("FFAC_TITRATION_FLASH", None)
            env.pop("FFAC_TITRATION_RECIPE", None)
        if variant.static_light != "off":
            env["FFAC_STATIC_LIGHT_CORRECTION"] = variant.static_light
        else:
            env.pop("FFAC_STATIC_LIGHT_CORRECTION", None)
        if variant.couple_aq_gas:
            env["FFAC_COUPLE_AQ_GAS"] = "on"
        else:
            env.pop("FFAC_COUPLE_AQ_GAS", None)

        master_cmd = _master_command(args, variant, queue, control, logs_dir)
        watchdog_cmd = _watchdog_command(args, queue, control)
        print(f"\n# {variant.name}: {variant.note}")
        env_summary = {
            key: env[key]
            for key in ("FFAC_TITRATION_FLASH", "FFAC_STATIC_LIGHT_CORRECTION", "FFAC_COUPLE_AQ_GAS")
            if key in env
        }
        if env_summary:
            print("ENV      " + " ".join(f"{key}={value}" for key, value in env_summary.items()))
        print("MASTER   " + " ".join(master_cmd))
        print("WATCHDOG " + " ".join(watchdog_cmd))
        if args.dry_run:
            continue

        variant_log_dir.mkdir(parents=True, exist_ok=True)
        master_out = master_log.open("ab")
        watchdog_out = watchdog_log.open("ab")
        try:
            master_proc = subprocess.Popen(
                master_cmd,
                cwd=repo,
                env=env,
                stdout=master_out,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            watchdog_proc = subprocess.Popen(
                watchdog_cmd,
                cwd=repo,
                env=env,
                stdout=watchdog_out,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        finally:
            master_out.close()
            watchdog_out.close()
        launched.extend(
            [
                {
                    "variant": variant.name,
                    "role": "master",
                    "pid": master_proc.pid,
                    "log": str(master_log),
                },
                {
                    "variant": variant.name,
                    "role": "watchdog",
                    "pid": watchdog_proc.pid,
                    "log": str(watchdog_log),
                },
            ]
        )

    if args.dry_run:
        print("\nDry run only; no processes started.")
        return

    manifest_root = Path(args.logs_root) if getattr(args, "logs_root", None) else _default_calibration_log_root()
    manifest = manifest_root / "ac53_testbed" / "launcher_processes.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(launched, indent=2), encoding="utf-8")
    print(f"\nStarted {len(launched)} processes for {len(selected)} variant(s).")
    print(f"Process manifest: {manifest}")
    print("Use Task Manager or Stop-Process with the PIDs in the manifest if you need to stop them.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summarize", help="Summarize existing tmp calibration logs.")
    summary.add_argument("--run", default="ac53")
    summary.add_argument(
        "--logs-dir",
        default=None,
        help="Log root to scan. Default scans the shared calibration log root and legacy local logs.",
    )
    summary.add_argument(
        "--selected",
        default="20260610_0050,20260612_0014,20260612_2342",
        help="Comma-separated path fragments to include in the alternative-winner section.",
    )
    summary.add_argument("--winners", action="store_true", help="Print alternative metric winners.")
    summary.set_defaults(func=summarize)

    cv = sub.add_parser("cv", help="Post-hoc cross-validation on alternating calibration timepoints.")
    cv.add_argument("--run", default="ac53")
    cv.add_argument(
        "--logs-dir",
        default=None,
        help="Log root to scan. Default scans the shared calibration log root and legacy local logs.",
    )
    cv.add_argument(
        "--selected",
        default="",
        help="Comma-separated path fragments to include. Empty means all matching logs.",
    )
    cv.set_defaults(func=cross_validate)

    commands = sub.add_parser("commands", help="Emit PowerShell commands for AC53 test variants.")
    commands.add_argument("--variant", default="core4", help="Variant name, comma-list, 'core4', or 'all'.")
    commands.add_argument("--repo", default=".")
    commands.add_argument("--python", default=".\\.venv\\Scripts\\python.exe")
    commands.add_argument("--logs-root", default=None, help="Default: FFAC_CALIBRATION_LOG_ROOT or Z:\\Albus\\Autokalibrering_log.")
    commands.add_argument("--queue-root", default=r"\\Moderskipet\Darsia_Queue\Kalibrering_AC53")
    commands.add_argument("--control-root", default=None)
    commands.add_argument("--max-iters", type=int, default=800)
    commands.add_argument("--warmup-iters", type=int, default=150)
    commands.add_argument("--max-in-flight-per-run", type=int, default=3)
    commands.add_argument("--workers", type=int, default=8)
    commands.add_argument("--sanity-every", type=int, default=100)
    commands.add_argument("--include-prep", action="store_true")
    commands.add_argument(
        "--save-calibration",
        action="store_true",
        help="Allow finalization to write optimized signal models back to the shared results folder. "
             "Default is safer for parallel diagnostics: only logs and best_params.json are written.",
    )
    commands.set_defaults(func=emit_commands)

    launcher = sub.add_parser("launch", help="Start AC53 test variants in background processes.")
    launcher.add_argument("--variant", default="core4", help="Variant name, comma-list, 'core4', or 'all'.")
    launcher.add_argument("--repo", default=".")
    launcher.add_argument("--python", default=sys.executable)
    launcher.add_argument("--logs-root", default=None, help="Default: FFAC_CALIBRATION_LOG_ROOT or Z:\\Albus\\Autokalibrering_log.")
    launcher.add_argument("--queue-root", default=r"\\Moderskipet\Darsia_Queue\Kalibrering_AC53")
    launcher.add_argument("--control-root", default=None)
    launcher.add_argument("--max-iters", type=int, default=800)
    launcher.add_argument("--warmup-iters", type=int, default=150)
    launcher.add_argument("--max-in-flight-per-run", type=int, default=3)
    launcher.add_argument("--workers", type=int, default=3)
    launcher.add_argument("--sanity-every", type=int, default=100)
    launcher.add_argument("--save-calibration", action="store_true")
    launcher.add_argument("--dry-run", action="store_true")
    launcher.set_defaults(func=launch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
