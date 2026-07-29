"""Compare CPU and GPU evaluation on identical real AC calibration contexts."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ac_production_campaign import EXPECTED_RUNTIME_DEPTH_SHA256
from scripts.auto_calibrate_color_to_mass import (
    EvalResult,
    build_context,
    evaluate_run,
    load_bounds_map,
    prepare_evaluation_context,
)


LOGGER = logging.getLogger(__name__)


def _canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parameter_record(
    *,
    record_id: str,
    params: dict[str, Any],
    objective: float | None,
    source: str,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "params": {str(key): value for key, value in params.items()},
        "objective": objective,
        "source": source,
    }


def _read_parameter_records(
    final_json: Path,
    history_csv: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final_data = json.loads(final_json.read_text(encoding="utf-8"))
    final_record = _parameter_record(
        record_id="final",
        params=dict(final_data["params"]),
        objective=float(final_data["objective_full_scale"]),
        source=str(final_json),
    )

    history: list[dict[str, Any]] = []
    with history_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            raw_params = (row.get("params") or "").strip()
            if not raw_params:
                continue
            parsed = ast.literal_eval(raw_params)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected parameter dictionary in {history_csv}, row {row_index + 2}"
                )
            try:
                objective = float(row.get("objective") or "nan")
            except ValueError:
                objective = math.nan
            history.append(
                _parameter_record(
                    record_id=f"history:{row.get('iter', row_index)}",
                    params=parsed,
                    objective=objective if math.isfinite(objective) else None,
                    source=str(history_csv),
                )
            )
    if not history:
        raise ValueError(f"No parameter rows found in {history_csv}")
    return final_record, history


def select_parameter_records(
    final_record: dict[str, Any],
    history: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Choose final, extrema, endpoints, and evenly spaced history records."""

    if limit < 1:
        raise ValueError("limit must be at least one")

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(record: dict[str, Any]) -> None:
        key = _canonical_params(record["params"])
        if key not in seen and len(selected) < limit:
            seen.add(key)
            selected.append(record)

    add(final_record)
    finite = [
        record
        for record in history
        if record["objective"] is not None
        and math.isfinite(float(record["objective"]))
    ]
    if finite:
        add(min(finite, key=lambda item: float(item["objective"])))
        add(max(finite, key=lambda item: float(item["objective"])))
    add(history[0])
    add(history[-1])

    remaining = max(0, limit - len(selected))
    count = max(2, remaining + 2)
    if len(history) == 1:
        indices = [0]
    else:
        indices = [
            round(index * (len(history) - 1) / max(1, count - 1))
            for index in range(count)
        ]
    for index in indices:
        add(history[index])
    for record in history:
        add(record)
    return selected


def _flatten_result(result: EvalResult) -> dict[str, float | None]:
    values: dict[str, float | None] = {"objective": float(result.objective)}
    for time_key, metric in sorted(result.metrics.items()):
        for field, value in asdict(metric).items():
            values[f"metrics.{time_key}.{field}"] = (
                None if value is None else float(value)
            )
    return values


def compare_results(
    cpu: EvalResult,
    gpu: EvalResult,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    cpu_values = _flatten_result(cpu)
    gpu_values = _flatten_result(gpu)
    keys = sorted(set(cpu_values) | set(gpu_values))
    fields: list[dict[str, Any]] = []
    failures: list[str] = []
    max_abs = 0.0
    max_rel = 0.0
    worst_abs_field = ""
    worst_rel_field = ""

    for key in keys:
        cpu_value = cpu_values.get(key)
        gpu_value = gpu_values.get(key)
        if cpu_value is None or gpu_value is None:
            passed = cpu_value is None and gpu_value is None
            item = {
                "field": key,
                "cpu": cpu_value,
                "gpu": gpu_value,
                "absolute_difference": None,
                "relative_difference": None,
                "passed": passed,
            }
        else:
            absolute = abs(gpu_value - cpu_value)
            scale = max(abs(cpu_value), abs(gpu_value), 1e-300)
            relative = absolute / scale
            passed = absolute <= atol or relative <= rtol
            item = {
                "field": key,
                "cpu": cpu_value,
                "gpu": gpu_value,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "passed": passed,
            }
            if absolute > max_abs:
                max_abs = absolute
                worst_abs_field = key
            if relative > max_rel:
                max_rel = relative
                worst_rel_field = key
        fields.append(item)
        if not item["passed"]:
            failures.append(key)

    metadata_match = (
        cpu.status == gpu.status
        and bool(cpu.feasible) == bool(gpu.feasible)
        and set(cpu.metrics) == set(gpu.metrics)
    )
    return {
        "passed": metadata_match and not failures,
        "metadata_match": metadata_match,
        "cpu_status": cpu.status,
        "gpu_status": gpu.status,
        "cpu_feasible": bool(cpu.feasible),
        "gpu_feasible": bool(gpu.feasible),
        "max_absolute_difference": max_abs,
        "max_absolute_field": worst_abs_field,
        "max_relative_difference": max_rel,
        "max_relative_field": worst_rel_field,
        "failed_fields": failures,
        "fields": fields,
    }


def _configure_phase_sharedpath_l1() -> None:
    values = {
        "FFAC_TITRATION_FLASH": "on",
        "FFAC_TEMPLATE_REGISTRATION": "ac14_template",
        "FFAC_TEMPLATE_REGISTRATION_MODE": "partial_affine",
        "FFAC_TEMPLATE_REGISTRATION_STRICT": "on",
        "FFAC_COLOR_PATH_ANCHOR": "ac60",
        "FFAC_COLOR_PATH_ANCHOR_WEIGHT": "0.75",
        "FFAC_COLOR_PATH_ANCHOR_STRICT": "on",
        "FFAC_REQUIRE_VARYING_DEPTH": "on",
        "FFAC_EXPECTED_DEPTH_SHA256": EXPECTED_RUNTIME_DEPTH_SHA256,
    }
    for key, value in values.items():
        os.environ[key] = value
    for key in (
        "FFAC_MASTER_LIGHT_CONTEXT",
        "FFAC_STATIC_LIGHT_CORRECTION",
        "FFAC_STATIC_LIGHT_REFERENCE",
        "FFAC_COUPLE_AQ_GAS",
        "FFAC_SIGNAL_PARAMETERIZATION",
        "FFAC_PHASE_SEPARATION",
    ):
        os.environ.pop(key, None)


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _close_context(context: Any) -> None:
    for name in ("_cuda_evaluator", "_opencl_evaluator"):
        evaluator = getattr(context, name, None)
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception:
                pass
            setattr(context, name, None)
    gc.collect()


def compare_run(
    *,
    run: str,
    backend: str,
    config_dir: Path,
    bounds_file: Path,
    final_json: Path,
    history_csv: Path,
    samples: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    from darsia.presets.workflows.rig import Rig

    final_record, history = _read_parameter_records(final_json, history_csv)
    records = select_parameter_records(final_record, history, samples)
    bounds_map = load_bounds_map(bounds_file)

    build_started = time.perf_counter()
    context = build_context(
        run=run,
        config_dir=config_dir,
        rig_cls=Rig,
        use_facies=True,
        bounds_map=bounds_map,
        per_label_params=True,
        quality_scale=1.0,
        quality_dtype="float32",
        objective_integral="off",
        static_light_correction="off",
        signal_parameterization="per-label",
        phase_separation="shared-signal",
        color_path_anchor="ac60",
        color_path_anchor_weight=0.75,
        color_path_anchor_strict=True,
    )
    build_seconds = time.perf_counter() - build_started
    depth_identity = dict(getattr(context.geometry, "_ffac_depth_identity", {}))

    try:
        cpu_prepare_started = time.perf_counter()
        prepare_evaluation_context(context, backend="prepared", release_images=True)
        cpu_prepare_seconds = time.perf_counter() - cpu_prepare_started

        cpu_results: list[tuple[EvalResult, float]] = []
        for record in records:
            started = time.perf_counter()
            result = evaluate_run(context, record["params"])
            cpu_results.append((result, time.perf_counter() - started))

        gpu_prepare_started = time.perf_counter()
        prepare_evaluation_context(context, backend=backend, release_images=True)
        gpu_prepare_seconds = time.perf_counter() - gpu_prepare_started

        comparisons: list[dict[str, Any]] = []
        for record, (cpu_result, cpu_seconds) in zip(records, cpu_results):
            started = time.perf_counter()
            gpu_result = evaluate_run(context, record["params"])
            gpu_seconds = time.perf_counter() - started
            comparison = compare_results(
                cpu_result,
                gpu_result,
                rtol=rtol,
                atol=atol,
            )
            comparison.update(
                {
                    "id": record["id"],
                    "source_objective": record["objective"],
                    "cpu_seconds": cpu_seconds,
                    "gpu_seconds": gpu_seconds,
                }
            )
            comparisons.append(comparison)

        return {
            "schema": 1,
            "run": run,
            "backend": backend,
            "passed": all(item["passed"] for item in comparisons),
            "candidate_count": len(comparisons),
            "rtol": rtol,
            "atol": atol,
            "build_seconds": build_seconds,
            "cpu_prepare_seconds": cpu_prepare_seconds,
            "gpu_prepare_seconds": gpu_prepare_seconds,
            "cpu_evaluation_seconds": sum(
                item["cpu_seconds"] for item in comparisons
            ),
            "gpu_evaluation_seconds": sum(
                item["gpu_seconds"] for item in comparisons
            ),
            "depth_map": depth_identity,
            "comparisons": comparisons,
        }
    finally:
        _close_context(context)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_results_root() -> Path:
    return Path(
        r"Z:\Albus\Autokalibrering_log\phase_sharedpath_l1_20260729b"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--backend", choices=("cuda", "opencl"), required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--config-dir", type=Path, default=Path("config_seg6/run_ac"))
    parser.add_argument(
        "--bounds-file",
        type=Path,
        default=Path("config/bounds_seg6_titration.json"),
    )
    parser.add_argument("--results-root", type=Path, default=_default_results_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/backend_parity/phase_sharedpath_l1_20260729b"),
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least one")
    _configure_phase_sharedpath_l1()

    repo = Path(__file__).resolve().parents[1]
    summaries: list[dict[str, Any]] = []
    for run in args.runs:
        source_dir = args.results_root / "phase_sharedpath_l1" / run
        LOGGER.info(
            "[%s] comparing CPU with %s using %d candidate(s)",
            run,
            args.backend,
            args.samples,
        )
        summary = compare_run(
            run=run,
            backend=args.backend,
            config_dir=args.config_dir,
            bounds_file=args.bounds_file,
            final_json=source_dir / f"final_full_scale_{run}.json",
            history_csv=source_dir / f"auto_calibration_{run}.csv",
            samples=args.samples,
            rtol=args.rtol,
            atol=args.atol,
        )
        summary["hostname"] = os.environ.get("COMPUTERNAME", "")
        summary["ff_ac_commit"] = _git_commit(repo)
        output = args.output_dir / f"cpu_vs_{args.backend}_{run}.json"
        _write_json(output, summary)
        summaries.append(summary)
        LOGGER.info(
            "[%s] %s: candidates=%d cpu=%.2fs gpu=%.2fs -> %s",
            run,
            args.backend,
            summary["candidate_count"],
            summary["cpu_evaluation_seconds"],
            summary["gpu_evaluation_seconds"],
            "PASS" if summary["passed"] else "FAIL",
        )

    campaign = {
        "schema": 1,
        "backend": args.backend,
        "passed": all(summary["passed"] for summary in summaries),
        "runs": summaries,
    }
    _write_json(args.output_dir / f"cpu_vs_{args.backend}_summary.json", campaign)
    print(json.dumps(campaign, indent=2))
    return 0 if campaign["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
