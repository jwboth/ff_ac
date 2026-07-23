"""Prepare trusted seeds and preflight the final 40-run AC calibration.

This script does not start a queue. It creates one deterministic seed file from
compatible full-model campaigns and checks the inputs needed by the persistent
final calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .ac_production_campaign import (
        FINAL_EXCLUDED_RUNS,
        FINAL_MAX_ACTIVE_RUNS,
        FINAL_MAX_IN_FLIGHT_PER_RUN,
        FINAL_PRODUCTION_RUNS,
        FINAL_VARIANT_NAME,
    )
    from .auto_calibrate_color_to_mass import build_param_space, load_bounds_map
except ImportError:
    from ac_production_campaign import (
        FINAL_EXCLUDED_RUNS,
        FINAL_MAX_ACTIVE_RUNS,
        FINAL_MAX_IN_FLIGHT_PER_RUN,
        FINAL_PRODUCTION_RUNS,
        FINAL_VARIANT_NAME,
    )
    from auto_calibrate_color_to_mass import build_param_space, load_bounds_map


DEFAULT_OUTPUT_ROOT = Path(
    r"Z:\Albus\Autokalibrering_log\final_production_20260723_16frames_24x1"
)
DEFAULT_RESULTS_ROOT = Path(r"Z:\Albus\Results")
DEFAULT_BOUNDS = Path("config/bounds_seg6_titration.json")
DEFAULT_QUEUE_ROOT = (
    r"\\Moderskipet\Darsia_Queue\Kalibrering_AC_final_20260723_16frames_24x1"
)
ACTIVE_LABELS = [1, 2, 5, 7, 8]
FINAL_CALIBRATION_POINT_COUNT = 16
REQUIRED_REDISTRIBUTION_TIMES_H = (4.1, 6.0, 8.0)
HOLDOUT_TIMES_H = (3.5, 7.0, 12.0)


@dataclass(frozen=True)
class CandidateSource:
    name: str
    root: Path
    priority: int


SOURCES = [
    CandidateSource(
        "final_template_ac14_seed17",
        Path(
            r"Z:\Albus\Autokalibrering_log\final_geometry_20260717"
            r"\final_template_ac14_seed17"
        ),
        0,
    ),
    CandidateSource(
        "final_template_ac14_seed73",
        Path(
            r"Z:\Albus\Autokalibrering_log\final_geometry_20260717"
            r"\final_template_ac14_seed73"
        ),
        1,
    ),
    CandidateSource(
        "screen_template_ac14",
        Path(
            r"Z:\Albus\Autokalibrering_log\screen_step1_20260713"
            r"\titration_template_ac14_l1"
        ),
        2,
    ),
    CandidateSource(
        "production_titration_l1",
        Path(r"Z:\Albus\Autokalibrering_log\production_titration_l1"),
        3,
    ),
    CandidateSource(
        "rollout_titration_l1",
        Path(r"Z:\Albus\Autokalibrering_log\rollout_titration_l1"),
        4,
    ),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _find_candidate(source: CandidateSource, run: str) -> Path | None:
    if not source.root.exists():
        return None
    matches = list(source.root.rglob(f"final_full_scale_{run}.json"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _parameter_space(bounds_file: Path) -> list[dict[str, Any]]:
    return build_param_space(
        "ac20",
        load_bounds_map(bounds_file),
        signal_labels=ACTIVE_LABELS,
        per_label_params=True,
        use_facies=True,
        n_free_values=6,
        signal_parameterization="per-label",
    )


def _project_candidate(
    params: dict[str, Any],
    param_space: list[dict[str, Any]],
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    projected: dict[str, float | int] = {}
    adjustments: list[dict[str, Any]] = []
    for entry in param_space:
        name = str(entry["name"])
        if name not in params:
            raise ValueError(f"missing parameter {name}")
        low, high = entry["bounds"]
        raw = float(params[name])
        value = min(float(high), max(float(low), raw))
        if entry.get("type") == "int":
            final_value: float | int = int(round(value))
        else:
            final_value = value
        projected[name] = final_value
        if abs(float(final_value) - raw) > 1e-12:
            adjustments.append(
                {
                    "parameter": name,
                    "from": raw,
                    "to": final_value,
                    "bounds": [low, high],
                }
            )
    return projected, adjustments


def _candidate_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(params.items()))


def _collect_run_candidates(
    run: str,
    param_space: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for source in sorted(SOURCES, key=lambda item: item.priority):
        path = _find_candidate(source, run)
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            params = payload.get("params") if isinstance(payload, dict) else None
            if not isinstance(params, dict) or not params:
                raise ValueError("no top-level params mapping")
            projected, adjustments = _project_candidate(params, param_space)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{source.name}: {path}: {exc}")
            continue
        key = _candidate_key(projected)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "source": source.name,
                "source_path": str(path),
                "source_objective": payload.get("objective_full_scale"),
                "projected_to_final_bounds": bool(adjustments),
                "adjustments": adjustments,
                "params": projected,
            }
        )
    return candidates, warnings


def _read_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    return path.read_text(encoding="utf-8").lstrip("\ufeff").strip().lower()


def _preflight_run(
    repo: Path,
    results_root: Path,
    run: str,
    candidate_count: int,
) -> dict[str, Any]:
    config_root = repo / "config_seg6" / "run_ac"
    result = results_root / run
    color_state = _read_state(config_root / ".color_state" / f"{run}.txt")
    titration_state = _read_state(
        config_root / ".titration_state" / f"{run}.txt"
    )
    expected_color = "off" if run == "ac44" else "on"
    frame_selection = _preflight_frame_selection(repo, run)
    checks = {
        "config": (config_root / f"{run}.toml").exists(),
        "results": result.exists(),
        "shape_corrected_baseline": (
            result / "setup" / "rig" / "shape_corrected_baseline.npz"
        ).exists(),
        "facies": (result / "setup" / "rig" / "facies.npz").exists(),
        "color_to_mass": (
            result
            / "calibration"
            / "color"
            / "relative_colorpath"
            / "color_to_mass"
            / "from_facies"
        ).exists(),
        "color_state": color_state == expected_color,
        "titration_state": titration_state == "on",
        "seed_candidates": candidate_count > 0,
        "redistribution_frames": frame_selection["redistribution_ok"],
        "holdout_frames": frame_selection["holdout_ok"],
    }
    return {
        "run": run,
        "ok": all(checks.values()),
        "checks": checks,
        "color_state": color_state,
        "expected_color_state": expected_color,
        "titration_state": titration_state,
        "seed_candidates": candidate_count,
        "frame_selection": frame_selection,
    }


def _preflight_frame_selection(repo: Path, run: str) -> dict[str, Any]:
    import darsia
    from darsia.presets.workflows.config.fluidflower_config import (
        FluidFlowerConfig,
    )

    paths = [
        repo / "config_seg6" / "common.toml",
        repo / "config_seg6" / "run_ac" / f"{run}.toml",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = FluidFlowerConfig(
            paths,
            require_results=True,
            require_data=True,
        )
    experiment = darsia.ProtocolledExperiment.init_from_config(config)
    mass_data = config.calibration.mass.data
    selected_paths: list[Path] = []
    selected_times: list[float] = []
    missing_times: list[float] = []
    resolved_by_request: dict[float, Path | None] = {}
    for requested_time, tolerance_hours in (
        mass_data.get_times_with_uncertainty()
    ):
        path = experiment.find_images_for_times(
            float(requested_time),
            tol=float(tolerance_hours) * 3600.0,
        )
        resolved_by_request[float(requested_time)] = (
            Path(path) if path is not None else None
        )
        if path is None:
            missing_times.append(float(requested_time))
            continue
        path = Path(path)
        if path in selected_paths:
            continue
        selected_paths.append(path)
        selected_times.append(
            float(experiment.time_since_start(experiment.get_datetime(path)))
        )

    redistribution_ok = all(
        any(
            abs(requested - required) < 1e-9 and path is not None
            for requested, path in resolved_by_request.items()
        )
        for required in REQUIRED_REDISTRIBUTION_TIMES_H
    )
    holdout_paths = [
        experiment.find_images_for_times(value, tol=600.0)
        for value in HOLDOUT_TIMES_H
    ]
    resolved_holdouts = [Path(path) for path in holdout_paths if path is not None]
    holdout_ok = (
        len(resolved_holdouts) == len(HOLDOUT_TIMES_H)
        and len(set(resolved_holdouts)) == len(HOLDOUT_TIMES_H)
        and not set(resolved_holdouts).intersection(selected_paths)
    )
    return {
        "calibration_frame_count": len(selected_paths),
        "calibration_actual_times_h": selected_times,
        "missing_requested_times_h": missing_times,
        "redistribution_ok": redistribution_ok,
        "holdout_times_h": HOLDOUT_TIMES_H,
        "holdout_ok": holdout_ok,
    }


def _backup_current_calibrations(
    results_root: Path,
    output_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    backup_root = output_root / "pre_calibration_backup"
    manifest_rows: list[dict[str, Any]] = []
    for run in FINAL_PRODUCTION_RUNS:
        source_root = (
            results_root
            / run
            / "calibration"
            / "color"
            / "relative_colorpath"
            / "color_to_mass"
            / "from_facies"
        )
        run_backup = backup_root / run
        for name in ("signal_model", "flash"):
            source = source_root / name
            destination = run_backup / name
            if not source.exists():
                raise FileNotFoundError(f"Missing calibration backup source: {source}")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
            for path in sorted(destination.rglob("*")):
                if not path.is_file():
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest_rows.append(
                    {
                        "run": run,
                        "path": str(path.relative_to(backup_root)),
                        "bytes": path.stat().st_size,
                        "sha256": digest,
                    }
                )
    manifest_path = backup_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": _utc_now(),
                "run_count": len(FINAL_PRODUCTION_RUNS),
                "files": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return backup_root, manifest_rows


def _ps_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _configured_calibration_times(repo: Path) -> list[float]:
    from darsia.presets.workflows.config.fluidflower_config import (
        FluidFlowerConfig,
    )

    paths = [
        repo / "config_seg6" / "common.toml",
        repo / "config_seg6" / "run_ac" / "ac20.toml",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = FluidFlowerConfig(
            paths,
            require_results=True,
            require_data=True,
        )
    mass_config = getattr(getattr(config, "calibration", None), "mass", None)
    time_data = getattr(mass_config, "data", None)
    return sorted(float(value) for value in (time_data.image_times or []))


def _write_launch_script(
    *,
    path: Path,
    repo: Path,
    output_root: Path,
    queue_root: str,
    seed_path: Path,
    role: str,
) -> None:
    arguments = [
        "& .\\.venv\\Scripts\\python.exe",
        "scripts/ac_production_campaign.py launch",
        "--variant final_production",
        "--run-set final_production",
        f"--role {role}",
        "--queue-root",
        _ps_quote(queue_root),
        "--logs-root",
        _ps_quote(output_root),
        "--max-iters 1600",
        "--warmup-iters 150",
        f"--max-active-runs {FINAL_MAX_ACTIVE_RUNS}",
        f"--max-in-flight-per-run {FINAL_MAX_IN_FLIGHT_PER_RUN}",
        "--total-workers 12",
        "--max-tasks-per-worker 250",
        "--idle-exit-seconds 120",
        "--threads-per-worker 1",
        "--sanity-every 100",
        "--seed-params-file",
        _ps_quote(seed_path),
        "--exact-logs-dir",
    ]
    path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"Set-Location {_ps_quote(repo)}",
                " ".join(arguments),
                "if ($LASTEXITCODE -ne 0) { "
                "throw \"Final calibration launcher failed with exit code $LASTEXITCODE\" }",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_stop_script(
    *,
    path: Path,
    repo: Path,
    queue_root: str,
) -> None:
    base_queue = queue_root.rstrip("\\/")
    queue = f"{base_queue}_{FINAL_VARIANT_NAME}"
    arguments = [
        "& .\\.venv\\Scripts\\python.exe",
        "scripts/stop_ac_calibration_campaign.py",
        "--marker",
        _ps_quote(Path(queue).name),
        "--queue",
        _ps_quote(queue),
        "--timeout 30",
    ]
    path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"Set-Location {_ps_quote(repo)}",
                " ".join(arguments),
                "if ($LASTEXITCODE -ne 0) { "
                "throw \"Campaign stop failed with exit code $LASTEXITCODE\" }",
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    output_root = Path(args.output_root)
    results_root = Path(args.results_root)
    bounds_file = Path(args.bounds_file)
    if not bounds_file.is_absolute():
        bounds_file = repo / bounds_file

    if len(FINAL_PRODUCTION_RUNS) != 40:
        raise RuntimeError(
            f"Expected 40 final runs, got {len(FINAL_PRODUCTION_RUNS)}"
        )
    if set(FINAL_EXCLUDED_RUNS).intersection(FINAL_PRODUCTION_RUNS):
        raise RuntimeError("An excluded run leaked into FINAL_PRODUCTION_RUNS")

    calibration_times = _configured_calibration_times(repo)
    missing_required_times = [
        value
        for value in REQUIRED_REDISTRIBUTION_TIMES_H
        if not any(abs(value - actual) < 1e-9 for actual in calibration_times)
    ]
    schedule_ok = (
        len(calibration_times) == FINAL_CALIBRATION_POINT_COUNT
        and not missing_required_times
    )

    param_space = _parameter_space(bounds_file)
    params_by_run: dict[str, list[dict[str, Any]]] = {}
    candidate_warnings: dict[str, list[str]] = {}
    for run in FINAL_PRODUCTION_RUNS:
        candidates, warnings = _collect_run_candidates(run, param_space)
        params_by_run[run] = candidates
        if warnings:
            candidate_warnings[run] = warnings

    preflight_rows = [
        _preflight_run(repo, results_root, run, len(params_by_run[run]))
        for run in FINAL_PRODUCTION_RUNS
    ]
    template_path = (
        results_root
        / "ac14_template"
        / "setup"
        / "rig"
        / "shape_corrected_baseline.npz"
    )
    failures = [row["run"] for row in preflight_rows if not row["ok"]]
    if not template_path.exists():
        failures.append("ac14_template")
    if not schedule_ok:
        failures.append("calibration_schedule")

    output_root.mkdir(parents=True, exist_ok=True)
    backup_root, backup_rows = _backup_current_calibrations(
        results_root,
        output_root,
    )
    seed_path = output_root / "final_seed_params.json"
    preflight_path = output_root / "preflight.json"
    seed_payload = {
        "metadata": {
            "created_at_utc": _utc_now(),
            "method": (
                "full per-label TitrationFlash, strict AC14 partial-affine, "
                "equal-weight point-wise L1 over up to 16 frames"
            ),
            "calibration_times_h": calibration_times,
            "frame_weighting": "equal",
            "bounds_file": str(bounds_file),
            "parameter_count": len(param_space),
            "run_count": len(FINAL_PRODUCTION_RUNS),
            "excluded_runs": FINAL_EXCLUDED_RUNS,
            "candidate_sources": [
                {"name": source.name, "root": str(source.root)}
                for source in SOURCES
            ],
        },
        "params_by_run": params_by_run,
    }
    seed_path.write_text(
        json.dumps(seed_payload, indent=2, default=str), encoding="utf-8"
    )
    local_launcher = output_root / "start_master_and_12_workers.ps1"
    watchdog_launcher = output_root / "start_12_workers.ps1"
    stop_launcher = output_root / "stop_campaign_on_this_machine.ps1"
    _write_launch_script(
        path=local_launcher,
        repo=repo,
        output_root=output_root,
        queue_root=args.queue_root,
        seed_path=seed_path,
        role="local",
    )
    _write_launch_script(
        path=watchdog_launcher,
        repo=repo,
        output_root=output_root,
        queue_root=args.queue_root,
        seed_path=seed_path,
        role="watchdog",
    )
    _write_stop_script(
        path=stop_launcher,
        repo=repo,
        queue_root=args.queue_root,
    )
    preflight_payload = {
        "created_at_utc": _utc_now(),
        "ok": not failures,
        "run_count": len(FINAL_PRODUCTION_RUNS),
        "calibration_point_count": len(calibration_times),
        "calibration_times_h": calibration_times,
        "required_redistribution_times_h": REQUIRED_REDISTRIBUTION_TIMES_H,
        "calibration_schedule_ok": schedule_ok,
        "template_path": str(template_path),
        "template_exists": template_path.exists(),
        "failed": failures,
        "candidate_warnings": candidate_warnings,
        "runs": preflight_rows,
    }
    preflight_path.write_text(
        json.dumps(preflight_payload, indent=2, default=str), encoding="utf-8"
    )

    counts = [len(params_by_run[run]) for run in FINAL_PRODUCTION_RUNS]
    frame_counts: dict[int, int] = {}
    for row in preflight_rows:
        count = int(row["frame_selection"]["calibration_frame_count"])
        frame_counts[count] = frame_counts.get(count, 0) + 1
    print(f"Final runs: {len(FINAL_PRODUCTION_RUNS)}")
    print(f"Excluded: {', '.join(FINAL_EXCLUDED_RUNS)}")
    print(
        "Prior candidates per run: "
        f"min={min(counts)}, median={sorted(counts)[len(counts) // 2]}, "
        f"max={max(counts)}, total={sum(counts)}"
    )
    print(
        "Available calibration frames per run: "
        + ", ".join(
            f"{count} frames={runs} run(s)"
            for count, runs in sorted(frame_counts.items())
        )
    )
    print(f"Seed file: {seed_path}")
    print(f"Preflight: {preflight_path}")
    print(
        f"Calibration backup: {backup_root} "
        f"({len(backup_rows)} files)"
    )
    print(f"Primary launcher: {local_launcher}")
    print(f"Second-machine launcher: {watchdog_launcher}")
    print(f"Stop launcher (run on each machine): {stop_launcher}")
    if failures:
        raise SystemExit("Preflight failed for: " + ", ".join(failures))
    print("Preflight OK.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--bounds-file", default=str(DEFAULT_BOUNDS))
    parser.add_argument("--queue-root", default=DEFAULT_QUEUE_ROOT)
    parser.set_defaults(func=prepare)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
