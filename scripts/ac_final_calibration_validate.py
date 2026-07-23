"""Evaluate final AC calibrations on held-out 4.1, 6, and 8 hour frames."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from .ac_production_campaign import FINAL_PRODUCTION_RUNS
    from .auto_calibrate_color_to_mass import (
        build_context,
        evaluate_run,
        load_bounds_map,
    )
except ImportError:
    from ac_production_campaign import FINAL_PRODUCTION_RUNS
    from auto_calibrate_color_to_mass import (
        build_context,
        evaluate_run,
        load_bounds_map,
    )


DEFAULT_LOGS = Path(
    r"Z:\Albus\Autokalibrering_log\final_production_20260723_24x1"
    r"\final_production_titration_ac14_seed17"
)
DEFAULT_TIMES = [4.1, 6.0, 8.0]


def _selected_runs(spec: str) -> list[str]:
    if spec.strip().lower() == "final":
        return list(FINAL_PRODUCTION_RUNS)
    runs = [
        token.strip().lower()
        for token in spec.replace(",", " ").split()
        if token.strip()
    ]
    unknown = set(runs).difference(FINAL_PRODUCTION_RUNS)
    if unknown:
        raise SystemExit(
            "Run(s) are not in the final campaign: " + ", ".join(sorted(unknown))
        )
    return runs


def _parse_times(spec: str) -> list[float]:
    times = [float(token) for token in spec.replace(",", " ").split() if token]
    if not times:
        raise SystemExit("At least one holdout time is required")
    return times


def _set_final_environment() -> dict[str, str | None]:
    values = {
        "FFAC_TITRATION_FLASH": "on",
        "FFAC_TEMPLATE_REGISTRATION": "ac14_template",
        "FFAC_TEMPLATE_REGISTRATION_MODE": "partial_affine",
        "FFAC_TEMPLATE_REGISTRATION_STRICT": "on",
        "FFAC_STATIC_LIGHT_CORRECTION": None,
        "FFAC_STATIC_LIGHT_REFERENCE": None,
        "FFAC_STATIC_LIGHT_SPATIAL_SIGMA": None,
        "FFAC_COUPLE_AQ_GAS": None,
        "FFAC_MASTER_LIGHT_CONTEXT": None,
        "FFAC_SIGNAL_PARAMETERIZATION": None,
    }
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def _restore_environment(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _metric_rows(run: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for time_key, metric in metrics.items():
        injected = float(metric.injected_full)
        detected = float(metric.total_full)
        rows.append(
            {
                "run": run,
                "time_h": float(time_key.removesuffix("h")),
                "injected_kg": injected,
                "detected_kg": detected,
                "ratio": detected / injected if injected else None,
                "absolute_error_kg": abs(detected - injected),
            }
        )
    return sorted(rows, key=lambda row: row["time_h"])


def validate(args: argparse.Namespace) -> None:
    from darsia.presets.workflows.rig import Rig

    repo = Path(args.repo).resolve()
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir) if args.output_dir else logs_dir / "holdout"
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds_path = Path(args.bounds_file)
    if not bounds_path.is_absolute():
        bounds_path = repo / bounds_path
    bounds_map = load_bounds_map(bounds_path)
    runs = _selected_runs(args.runs)
    times = _parse_times(args.times)

    all_rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    previous_env = _set_final_environment()
    try:
        for index, run in enumerate(runs, start=1):
            final_path = logs_dir / f"final_full_scale_{run}.json"
            if not final_path.exists():
                failures[run] = f"missing {final_path}"
                continue
            try:
                payload = json.loads(final_path.read_text(encoding="utf-8"))
                params = payload.get("params")
                if not isinstance(params, dict) or not params:
                    raise ValueError("final JSON has no params mapping")
                context = build_context(
                    run=run,
                    config_dir=repo / "config_seg6" / "run_ac",
                    rig_cls=Rig,
                    use_facies=True,
                    bounds_map=bounds_map,
                    per_label_params=True,
                    quality_scale=1.0,
                    quality_dtype="float32",
                    objective_integral="off",
                    static_light_correction="off",
                    signal_parameterization="per-label",
                    evaluation_times_hours=times,
                    evaluation_time_tolerance_seconds=float(args.tolerance_minutes)
                    * 60.0,
                )
                result = evaluate_run(context, params)
                if result.status != "ok":
                    raise RuntimeError(result.status)
                rows = _metric_rows(run, result.metrics)
                all_rows.extend(rows)
                (output_dir / f"holdout_{run}.json").write_text(
                    json.dumps(
                        {
                            "run": run,
                            "requested_times_h": times,
                            "source_final": str(final_path),
                            "objective_holdout": result.objective,
                            "rows": rows,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"[{index}/{len(runs)}] {run}: {len(rows)} holdout frames")
            except Exception as exc:  # noqa: BLE001
                failures[run] = str(exc)
                print(f"[{index}/{len(runs)}] {run}: FAILED: {exc}")
            finally:
                if "context" in locals():
                    del context
                gc.collect()
    finally:
        _restore_environment(previous_env)

    csv_path = output_dir / "holdout_summary.csv"
    fieldnames = [
        "run",
        "time_h",
        "injected_kg",
        "detected_kg",
        "ratio",
        "absolute_error_kg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_dir / "holdout_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "requested_times_h": times,
                "runs_requested": len(runs),
                "runs_completed": len({row["run"] for row in all_rows}),
                "failures": failures,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    print(f"CSV: {csv_path}")
    if failures:
        raise SystemExit(
            f"Holdout validation failed for {len(failures)} run(s): "
            + ", ".join(failures)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("--repo", default=".")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--runs", default="final")
    parser.add_argument("--times", default="4.1 6 8")
    parser.add_argument("--tolerance-minutes", type=float, default=10.0)
    parser.add_argument(
        "--bounds-file",
        default="config/bounds_seg6_titration.json",
    )
    parser.set_defaults(func=validate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
