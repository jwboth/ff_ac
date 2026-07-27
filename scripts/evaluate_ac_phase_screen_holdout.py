"""Evaluate completed phase-screen winners on three untouched time points."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from darsia.presets.workflows.rig import Rig

try:
    from .ac_production_campaign import (
        PHASE_SCREEN_RUNS,
        _select_variants,
        _variant_env,
    )
    from .auto_calibrate_color_to_mass import (
        _mass_result_for_evaluation,
        build_context,
        evaluate_run,
        load_bounds_map,
    )
except ImportError:
    from ac_production_campaign import (
        PHASE_SCREEN_RUNS,
        _select_variants,
        _variant_env,
    )
    from auto_calibrate_color_to_mass import (
        _mass_result_for_evaluation,
        build_context,
        evaluate_run,
        load_bounds_map,
    )


DEFAULT_LOG_ROOT = Path(
    r"Z:\Albus\Autokalibrering_log\phase_screen_20260727"
)
REPO = Path(__file__).resolve().parents[1]
HOLDOUT_TIMES_H = (3.5, 7.0, 12.0)
MANAGED_ENV = {
    "FFAC_TITRATION_FLASH",
    "FFAC_TITRATION_RECIPE",
    "FFAC_STATIC_LIGHT_CORRECTION",
    "FFAC_STATIC_LIGHT_SPATIAL_SIGMA",
    "FFAC_STATIC_LIGHT_REFERENCE",
    "FFAC_COUPLE_AQ_GAS",
    "FFAC_TEMPLATE_REGISTRATION",
    "FFAC_TEMPLATE_REGISTRATION_MODE",
    "FFAC_TEMPLATE_REGISTRATION_STRICT",
    "FFAC_SIGNAL_PARAMETERIZATION",
    "FFAC_COLOR_PATH_ANCHOR",
    "FFAC_COLOR_PATH_ANCHOR_WEIGHT",
    "FFAC_COLOR_PATH_ANCHOR_STRICT",
    "FFAC_PHASE_SEPARATION",
    "FFAC_MASTER_LIGHT_CONTEXT",
}


@contextmanager
def _variant_environment(variant) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in MANAGED_ENV}
    configured = _variant_env(
        dict(os.environ),
        variant,
        master=False,
        spatial_sigma=6.0,
    )
    try:
        for key in MANAGED_ENV:
            if key in configured:
                os.environ[key] = configured[key]
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _latest_final(folder: Path, run: str) -> Path:
    matches = list(folder.rglob(f"final_full_scale_{run}.json"))
    if not matches:
        raise FileNotFoundError(
            f"No completed final_full_scale_{run}.json below {folder}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def _write_phase_previews(
    log_root: Path,
    variant_name: str,
    run: str,
    context,
    params: dict,
) -> list[str]:
    import cv2
    import numpy as np

    output_dir = log_root / "holdout_phase_previews" / variant_name / run
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for index, (image, _injected, time_h) in enumerate(context._loaded):
        result = _mass_result_for_evaluation(context, image, index, params)
        gas = np.clip(np.asarray(result.saturation_g.img), 0.0, 1.0)
        aqueous = np.clip(np.asarray(result.concentration_aq.img), 0.0, 1.0)
        preview = np.zeros((*gas.shape, 3), dtype=np.uint8)
        preview[..., 0] = np.rint(gas * 255.0).astype(np.uint8)
        preview[..., 2] = np.rint(aqueous * 255.0).astype(np.uint8)
        preview = cv2.resize(
            preview,
            None,
            fx=0.2,
            fy=0.2,
            interpolation=cv2.INTER_AREA,
        )
        # OpenCV writes BGR; swap so the file displays gas=red, aqueous=blue.
        path = output_dir / f"{time_h:.3f}h_phase_preview.png"
        cv2.imwrite(str(path), preview[..., ::-1])
        outputs.append(str(path))
    return outputs


def evaluate_holdouts(log_root: Path, runs: list[str]) -> Path:
    rows: list[dict] = []
    for variant in _select_variants("phase_screen"):
        variant_root = log_root / variant.name
        with _variant_environment(variant):
            for run in runs:
                final_path = _latest_final(variant_root, run)
                payload = json.loads(final_path.read_text(encoding="utf-8"))
                params = payload.get("params")
                if not isinstance(params, dict):
                    raise ValueError(f"{final_path}: missing params")
                context = build_context(
                    run=run,
                    config_dir=REPO / "config_seg6" / "run_ac",
                    rig_cls=Rig,
                    use_facies=True,
                    bounds_map=load_bounds_map(REPO / variant.bounds_file),
                    per_label_params=True,
                    quality_scale=1.0,
                    quality_dtype="float32",
                    # Holdouts are reported point-wise; they are not optimized.
                    objective_integral="off",
                    signal_parameterization=variant.signal_parameterization,
                    phase_separation=variant.phase_separation,
                    evaluation_times_hours=HOLDOUT_TIMES_H,
                )
                result = evaluate_run(context, params)
                if not result.metrics:
                    raise RuntimeError(
                        f"{variant.name}/{run}: empty holdout metrics ({result.status})"
                    )
                metrics = {key: vars(value) for key, value in result.metrics.items()}
                errors = [
                    abs(value["total_full"] - value["injected_full"])
                    for value in metrics.values()
                ]
                previews = _write_phase_previews(
                    log_root,
                    variant.name,
                    run,
                    context,
                    params,
                )
                rows.append(
                    {
                        "variant": variant.name,
                        "run": run,
                        "training_result": str(final_path),
                        "holdout_times_h": HOLDOUT_TIMES_H,
                        "holdout_mae": sum(errors) / len(errors),
                        "holdout_max_abs_error": max(errors),
                        "metrics": metrics,
                        "phase_previews": previews,
                    }
                )
                print(
                    f"{variant.name}/{run}: holdout MAE="
                    f"{rows[-1]['holdout_mae']:.6g}",
                    flush=True,
                )
    output = log_root / "phase_screen_holdout_summary.json"
    output.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--runs", nargs="+", default=PHASE_SCREEN_RUNS)
    args = parser.parse_args()
    output = evaluate_holdouts(
        args.logs_root,
        [str(run).lower() for run in args.runs],
    )
    print(f"Holdout summary: {output}")


if __name__ == "__main__":
    main()
