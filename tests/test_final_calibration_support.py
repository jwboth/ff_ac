from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.multiphase.flash import SimpleFlash, TitrationFlash
from scripts.ac_final_calibration_prepare import _project_candidate
from scripts.ac_production_campaign import (
    FINAL_EXCLUDED_RUNS,
    FINAL_PRODUCTION_RUNS,
    _common_master_args,
    _select_runs,
    _select_variants,
    build_parser as build_campaign_parser,
)
from scripts.auto_calibrate_color_to_mass import save_best_calibration
from scripts.distributed_auto_calibration_queue import (
    _load_seed_params_file,
    _merge_unique_params,
    _seed_params_for_run,
)


PARAM_SPACE = [
    {"name": "a", "bounds": (0.0, 2.0), "type": "float"},
    {"name": "b", "bounds": (1, 3), "type": "int"},
]


class _SignalFunction:
    def __init__(self) -> None:
        self.values = np.zeros(2, dtype=float)

    def update(self, *, values) -> None:
        self.values = np.asarray(values, dtype=float)

    def save(self, path: Path) -> None:
        path.with_suffix(".json").write_text(
            json.dumps({"values": self.values.tolist()}), encoding="utf-8"
        )


def test_titration_flash_round_trip_preserves_recipe_and_type(tmp_path):
    flash = TitrationFlash(
        0.0,
        0.91,
        0.75,
        1.67,
        alkalinity_M=1.4e-3,
        n_lut=128,
    )
    path = tmp_path / "flash" / "flash"

    flash.save(path)
    loaded = SimpleFlash.load(path)

    assert isinstance(loaded, TitrationFlash)
    assert loaded.max_value_aq == pytest.approx(0.91)
    assert loaded.max_value_g == pytest.approx(1.67)
    assert loaded.titration_params["alkalinity_M"] == pytest.approx(1.4e-3)
    assert loaded.n_lut == 128


def test_save_best_calibration_persists_signal_and_flash(tmp_path):
    signal = _SignalFunction()
    flash = TitrationFlash(0.0, 0.9, 0.75, 1.6, n_lut=64)
    calibration = SimpleNamespace(
        signal_model=SimpleNamespace(model={1: {1: signal}}),
        flash=flash,
    )
    context = SimpleNamespace(
        run="ac20",
        calibration=calibration,
        signal_labels=[1],
        calibration_folder=tmp_path / "color_to_mass",
    )
    params = {
        "signal.label1.value0": 0.0,
        "signal.label1.value1": 1.25,
        "flash.max_value_aq": 0.97,
        "flash.max_value_g": 1.72,
    }

    save_best_calibration(context, params, out_folder=tmp_path / "logs")

    assert signal.values.tolist() == pytest.approx([0.0, 1.25])
    loaded = SimpleFlash.load(
        tmp_path / "color_to_mass" / "flash" / "flash"
    )
    assert isinstance(loaded, TitrationFlash)
    assert loaded.max_value_aq == pytest.approx(0.97)
    assert loaded.max_value_g == pytest.approx(1.72)
    assert (tmp_path / "logs" / "best_params.json").exists()


def test_seed_candidates_are_loaded_validated_and_prepended(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {
                "params_by_run": {
                    "ac20": [
                        {"source": "first", "params": {"a": 0.5, "b": 1}},
                        {"source": "second", "params": {"a": 1.5, "b": 3}},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_seed_params_file(path)
    candidates = _seed_params_for_run(loaded, "AC20", PARAM_SPACE)
    merged = _merge_unique_params(candidates, [candidates[0], {"a": 1.0, "b": 2}])

    assert candidates == [{"a": 0.5, "b": 1}, {"a": 1.5, "b": 3}]
    assert merged == [
        {"a": 0.5, "b": 1},
        {"a": 1.5, "b": 3},
        {"a": 1.0, "b": 2},
    ]


def test_seed_projection_clamps_only_out_of_bounds_values():
    projected, adjustments = _project_candidate(
        {"a": -0.5, "b": 2},
        PARAM_SPACE,
    )
    assert projected == {"a": 0.0, "b": 2}
    assert [item["parameter"] for item in adjustments] == ["a"]


def test_final_campaign_is_full_model_persistent_and_excludes_outliers(tmp_path):
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text("{}", encoding="utf-8")
    variants = _select_variants("final_production")
    runs = _select_runs("final_production")

    assert runs == FINAL_PRODUCTION_RUNS
    assert len(runs) == 40
    assert not set(FINAL_EXCLUDED_RUNS).intersection(runs)
    assert len(variants) == 1
    variant = variants[0]
    assert variant.signal_parameterization == "per-label"
    assert variant.template_registration == "ac14_template"
    assert variant.template_strict
    assert variant.optuna_seed == 17
    assert variant.save_calibration

    args = build_campaign_parser().parse_args(
        [
            "launch",
            "--variant",
            "final_production",
            "--run-set",
            "final_production",
            "--max-iters",
            "1600",
            "--warmup-iters",
            "150",
            "--seed-params-file",
            str(seed_path),
        ]
    )
    command = _common_master_args(
        args,
        variant,
        "queue",
        "logs",
        "control",
    )
    assert "--no-save-calibration" not in command
    assert command[command.index("--seed-params-file") + 1] == f"'{seed_path}'"
    assert command[command.index("--max-iters") + 1] == "1600"
