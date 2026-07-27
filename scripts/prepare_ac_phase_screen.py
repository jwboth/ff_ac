"""Preflight and document the six-run phase-separation calibration screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .ac_final_calibration_prepare import (
        _collect_run_candidates,
        _parameter_space,
        _preflight_frame_selection,
    )
    from .ac_production_campaign import (
        PHASE_SCREEN_RUNS,
        _select_variants,
    )
except ImportError:
    from ac_final_calibration_prepare import (
        _collect_run_candidates,
        _parameter_space,
        _preflight_frame_selection,
    )
    from ac_production_campaign import (
        PHASE_SCREEN_RUNS,
        _select_variants,
    )


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEPTH_MANIFEST = Path(
    r"Z:\Albus\Autokalibrering_log\input_standardization_20260727"
    r"\depth_input_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"Z:\Albus\Autokalibrering_log\phase_screen_20260727"
)
ACTIVE_LABELS = (1, 2, 5, 7, 8)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    return path.read_text(encoding="utf-8").lstrip("\ufeff").strip().lower()


def prepare(depth_manifest_path: Path, output_root: Path) -> Path:
    depth_manifest = json.loads(
        depth_manifest_path.read_text(encoding="utf-8")
    )
    depth_by_run = {
        row["run"]: row for row in depth_manifest.get("records", [])
    }
    depth_hashes = {
        depth_by_run[run]["sha256_array"]
        for run in PHASE_SCREEN_RUNS
        if run in depth_by_run
    }
    run_rows: list[dict] = []
    for run in PHASE_SCREEN_RUNS:
        frame_selection = _preflight_frame_selection(REPO, run)
        result_root = Path(r"Z:\Albus\Results") / run
        calibration_root = (
            result_root
            / "calibration"
            / "color"
            / "relative_colorpath"
            / "color_to_mass"
            / "from_facies"
        )
        checks = {
            "standardized_depth": run in depth_by_run,
            "measured_depth": (
                run in depth_by_run
                and float(depth_by_run[run]["std_m"]) > 1e-6
            ),
            "color_state_on": _read_state(
                REPO
                / "config_seg6"
                / "run_ac"
                / ".color_state"
                / f"{run}.txt"
            )
            == "on",
            "titration_state_on": _read_state(
                REPO
                / "config_seg6"
                / "run_ac"
                / ".titration_state"
                / f"{run}.txt"
            )
            == "on",
            "color_correction_cache": bool(
                list(
                    (
                        result_root / "setup" / "rig"
                    ).glob("color_correction_*_colorcorrection.npz")
                )
            ),
            "calibration_16_frames": (
                frame_selection["calibration_frame_count"] == 16
            ),
            "redistribution_frames": frame_selection["redistribution_ok"],
            "holdout_frames": frame_selection["holdout_ok"],
            "color_to_mass": calibration_root.exists(),
        }
        run_rows.append(
            {
                "run": run,
                "ok": all(checks.values()),
                "checks": checks,
                "depth": depth_by_run.get(run),
                "frame_selection": frame_selection,
            }
        )

    anchor_root = (
        Path(r"Z:\Albus\Results\ac60")
        / "calibration"
        / "color"
        / "relative_colorpath"
        / "color_to_mass"
        / "from_facies"
        / "color_path_interpretation"
    )
    anchor_paths = {
        str(label): str(anchor_root / f"color_path_interpretation_{label}.json")
        for label in ACTIVE_LABELS
    }
    anchor_ok = all(Path(path).exists() for path in anchor_paths.values())
    depth_ok = (
        len(depth_hashes) == 1
        and len(depth_by_run) >= len(PHASE_SCREEN_RUNS)
        and depth_manifest.get("status") == "ok"
    )
    variants = _select_variants("phase_screen")
    parameter_space = _parameter_space(
        REPO / "config" / "bounds_seg6_titration.json"
    )
    seed_rows: dict[str, list[dict]] = {}
    seed_warnings: dict[str, list[str]] = {}
    for run in PHASE_SCREEN_RUNS:
        candidates, warnings = _collect_run_candidates(run, parameter_space)
        for candidate in candidates:
            # Extra keys are ignored by control/shared-path param spaces and used
            # by the residual-gas variant.
            candidate["params"]["gas.residual_onset"] = 0.20
            candidate["params"]["gas.residual_width"] = 0.40
        seed_rows[run] = candidates
        if warnings:
            seed_warnings[run] = warnings
    seed_path = output_root / "phase_screen_seed_params.json"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(
        json.dumps(
            {
                "created_at_utc": _utc_now(),
                "purpose": (
                    "Evaluate trusted previous full-model winners before new "
                    "warmup and Optuna trials on standardized depth."
                ),
                "params_by_run": seed_rows,
                "warnings": seed_warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    seeds_ok = all(seed_rows.get(run) for run in PHASE_SCREEN_RUNS)
    payload = {
        "created_at_utc": _utc_now(),
        "status": (
            "ready"
            if (
                depth_ok
                and anchor_ok
                and seeds_ok
                and all(row["ok"] for row in run_rows)
            )
            else "blocked"
        ),
        "purpose": (
            "Separate depth/input validity from temporal weighting, color-path "
            "regularization, and gas/aqueous optical identifiability."
        ),
        "acceptance_rule": (
            "Prefer a new variant only if it improves both calibration-window "
            "balance and the untouched 3.5/7/12 h holdouts without implausible "
            "gas maps. Lowest training objective alone is insufficient."
        ),
        "limitations": [
            (
                "The residual-gas branch is an independent optical proxy based "
                "on off-path reflection/scattering, not direct phase ground truth."
            ),
            (
                "No variant writes back to experiment calibration folders during "
                "this screen."
            ),
        ],
        "runs": PHASE_SCREEN_RUNS,
        "run_roles": {
            "controls": ["ac20", "ac60"],
            "early_under_late_over": ["ac24", "ac32"],
            "persistent_hard_cases": ["ac26", "ac27"],
        },
        "variants": [
            {
                "name": variant.name,
                "objective": variant.objective_integral,
                "color_path_anchor": variant.color_path_anchor,
                "color_path_anchor_weight": variant.color_path_anchor_weight,
                "phase_separation": variant.phase_separation,
                "note": variant.note,
            }
            for variant in variants
        ],
        "objective_windows_h": {
            "I1": "<=0.92",
            "I2": ">0.92 and requested before 4.1",
            "early_post_injection_redistribution": (
                "requested 4.1-8.0; actual-frame tolerance 4.0-8.25"
            ),
            "late_redistribution": "after the requested 8 h frame",
        },
        "holdout_times_h": [3.5, 7.0, 12.0],
        "depth_manifest": str(depth_manifest_path),
        "depth_ok": depth_ok,
        "depth_hashes": sorted(depth_hashes),
        "anchor_paths": anchor_paths,
        "anchor_ok": anchor_ok,
        "seed_params_file": str(seed_path),
        "seed_candidates_per_run": {
            run: len(candidates) for run, candidates in seed_rows.items()
        },
        "seeds_ok": seeds_ok,
        "run_preflight": run_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "phase_screen_preflight.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["status"] != "ready":
        failed = [
            row["run"] for row in run_rows if not row["ok"]
        ]
        raise RuntimeError(
            f"Phase screen preflight blocked; failed runs={failed}, "
            f"depth_ok={depth_ok}, anchor_ok={anchor_ok}, seeds_ok={seeds_ok}. "
            f"See {output}"
        )
    print(f"Phase screen ready: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth-manifest",
        type=Path,
        default=DEFAULT_DEPTH_MANIFEST,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    prepare(args.depth_manifest, args.output_root)


if __name__ == "__main__":
    main()
