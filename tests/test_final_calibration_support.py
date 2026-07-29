from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.multiphase.flash import SimpleFlash, TitrationFlash
from scripts.ac_final_calibration_prepare import (
    FINAL_CALIBRATION_POINT_COUNT,
    REQUIRED_REDISTRIBUTION_TIMES_H,
    _configured_calibration_times,
    _project_candidate,
)
from scripts.ac_production_campaign import (
    FINAL_EXCLUDED_RUNS,
    FINAL_MAX_ACTIVE_RUNS,
    FINAL_MAX_IN_FLIGHT_PER_RUN,
    FINAL_PRODUCTION_RUNS,
    FINAL_VARIANT_NAME,
    _common_master_args,
    _select_runs,
    _select_variants,
    _validate_final_campaign,
    build_parser as build_campaign_parser,
)
from scripts import auto_calibrate_color_to_mass
from scripts.auto_calibrate_color_to_mass import evaluate_run, save_best_calibration
from scripts.distributed_auto_calibration_queue import (
    _call_with_transient_retries,
    _load_seed_params_file,
    _merge_unique_params,
    _remaining_local_runs,
    _seed_params_for_run,
    _select_initial_local_run,
)
from scripts.stop_ac_calibration_campaign import (
    ProcessRecord,
    _expand_target_pids,
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
    assert variant.name == FINAL_VARIANT_NAME
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
            "--max-active-runs",
            str(FINAL_MAX_ACTIVE_RUNS),
            "--max-in-flight-per-run",
            str(FINAL_MAX_IN_FLIGHT_PER_RUN),
        ]
    )
    _validate_final_campaign(args, variants, runs)
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
    assert (
        command[command.index("--max-active-runs") + 1]
        == str(FINAL_MAX_ACTIVE_RUNS)
    )
    assert (
        command[command.index("--max-in-flight-per-run") + 1]
        == str(FINAL_MAX_IN_FLIGHT_PER_RUN)
    )


def test_final_mass_calibration_schedule_contains_16_equal_weight_frames():
    repo = Path(__file__).resolve().parents[1]
    times = _configured_calibration_times(repo)

    assert len(times) == FINAL_CALIBRATION_POINT_COUNT
    for expected in REQUIRED_REDISTRIBUTION_TIMES_H:
        assert any(actual == pytest.approx(expected) for actual in times)


def test_pointwise_l1_uses_equal_absolute_weight_per_frame(monkeypatch):
    monkeypatch.setattr(
        auto_calibrate_color_to_mass,
        "apply_params",
        lambda *args, **kwargs: None,
    )
    calibration = lambda detected: SimpleNamespace(mass=detected)
    context = SimpleNamespace(
        _loaded=[
            (1.0, 0.0, 1.0),
            (5.0, 3.0, 2.0),
            (7.0, 10.0, 3.0),
        ],
        calibration=calibration,
        geometry=SimpleNamespace(integrate=float),
        enforce_lower=False,
        objective_integral="off",
        signal_labels=[],
    )

    result = evaluate_run(context, {})

    assert result.status == "ok"
    assert result.objective == pytest.approx(1.0 + 2.0 + 3.0)


def test_campaign_stop_expands_tree_and_recorded_orphan_spawn():
    marker = "Kalibrering_AC_final_test"
    records = {
        10: ProcessRecord(10, 1, "powershell.exe", marker, 1.0),
        11: ProcessRecord(11, 10, "python.exe", "master", 2.0),
        12: ProcessRecord(
            12,
            11,
            "python.exe",
            "spawn_main(parent_pid=11) --multiprocessing-fork",
            3.0,
        ),
        20: ProcessRecord(
            20,
            1,
            "python.exe",
            "spawn_main(parent_pid=99) --multiprocessing-fork",
            4.0,
        ),
        30: ProcessRecord(30, 1, "python.exe", "unrelated.py", 5.0),
    }

    targets = _expand_target_pids(
        records,
        marker,
        recorded_watchdog_pids={99},
        excluded_pids=set(),
    )

    assert targets == {10, 11, 12, 20}


def test_context_build_retries_only_transient_io():
    calls = 0
    messages = []

    def succeeds_after_network_errors():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("network file temporarily unavailable")
        return "ready"

    assert (
        _call_with_transient_retries(
            succeeds_after_network_errors,
            label="[ac44] context build",
            attempts=4,
            delay=0,
            log=messages.append,
        )
        == "ready"
    )
    assert calls == 3
    assert len(messages) == 2

    with pytest.raises(AssertionError, match="invalid geometry"):
        _call_with_transient_retries(
            lambda: (_ for _ in ()).throw(AssertionError("invalid geometry")),
            label="[ac44] context build",
            attempts=4,
            delay=0,
        )


def test_local_gpu_resume_skips_completed_requested_run(tmp_path):
    run_complete = tmp_path / "run_complete"
    run_complete.mkdir()
    (run_complete / "ac17.json").write_text("{}", encoding="utf-8")
    dirs = {"run_complete": run_complete}

    assert (
        _select_initial_local_run(
            dirs,
            ("ac17", "ac19", "ac23"),
            "ac17",
        )
        == "ac19"
    )
    assert (
        _select_initial_local_run(
            dirs,
            ("ac17", "ac19", "ac23"),
            "ac23",
        )
        == "ac23"
    )
    assert _remaining_local_runs(
        dirs,
        ("ac17", "ac19", "ac23"),
    ) == ["ac19", "ac23"]


def test_final_campaign_rejects_concurrent_trials_for_the_same_run(tmp_path):
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text("{}", encoding="utf-8")
    variants = _select_variants("final_production")
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
            "--max-active-runs",
            "12",
            "--max-in-flight-per-run",
            "2",
            "--seed-params-file",
            str(seed_path),
        ]
    )

    with pytest.raises(SystemExit, match="max-active-runs 24"):
        _validate_final_campaign(
            args,
            variants,
            _select_runs("final_production"),
        )
