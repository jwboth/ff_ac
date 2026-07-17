from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pytest

from scripts.auto_calibrate_color_to_mass import (
    _compute_anchor_photometric_gain,
    _porous_baseline_median,
    sample_params,
)
from scripts.distributed_auto_calibration_queue import (
    _clear_master_complete,
    _fixed_params_for_run,
    _load_fixed_params_file,
    _mark_master_complete,
    _queue_is_complete,
    _worker_context_is_idle,
    _derived_run_seed,
    watchdog_main,
    build_parser as build_queue_parser,
)
from scripts.ac_production_campaign import (
    FINAL_GEOMETRY_RUNS,
    _common_master_args,
    _select_runs,
    _select_variants,
    _watchdog_args,
    build_parser as build_campaign_parser,
)


PARAM_SPACE = [
    {"name": "a", "bounds": (0.0, 2.0), "type": "float"},
    {"name": "b", "bounds": (1, 3), "type": "int"},
]


def test_loads_final_payload_as_shared_fixed_params(tmp_path):
    path = tmp_path / "final.json"
    path.write_text(json.dumps({"run": "ac53", "params": {"a": 1.25, "b": 2}}))

    loaded = _load_fixed_params_file(path)
    assert loaded == {"*": {"a": 1.25, "b": 2}}
    assert _fixed_params_for_run(loaded, "ac24", PARAM_SPACE) == {"a": 1.25, "b": 2}


def test_loads_explicit_per_run_fixed_params(tmp_path):
    path = tmp_path / "fixed.json"
    path.write_text(
        json.dumps(
            {
                "params_by_run": {
                    "AC24": {"a": 0.5, "b": 1},
                    "ac40": {"a": 1.5, "b": 3},
                }
            }
        )
    )

    loaded = _load_fixed_params_file(path)
    assert _fixed_params_for_run(loaded, "ac24", PARAM_SPACE) == {"a": 0.5, "b": 1}
    assert _fixed_params_for_run(loaded, "ac40", PARAM_SPACE) == {"a": 1.5, "b": 3}


def test_fixed_params_reject_missing_and_out_of_bounds():
    with pytest.raises(ValueError, match="miss"):
        _fixed_params_for_run({"*": {"a": 1.0}}, "ac24", PARAM_SPACE)
    with pytest.raises(ValueError, match="outside"):
        _fixed_params_for_run({"*": {"a": 2.5, "b": 2}}, "ac24", PARAM_SPACE)


def test_porous_baseline_median_uses_only_active_labels():
    rgb = np.zeros((50, 50, 3), dtype=np.float32)
    labels = np.zeros((50, 50), dtype=np.int16)
    labels[:, :25] = 1
    labels[:, 25:] = 9
    rgb[:, :25] = (0.2, 0.4, 0.8)
    rgb[:, 25:] = (0.9, 0.1, 0.1)

    median = _porous_baseline_median(rgb, labels, [1], stride=1)
    np.testing.assert_allclose(median, [0.2, 0.4, 0.8], atol=1e-6)


def test_anchor_gain_is_diagonal_and_clipped():
    raw, gain = _compute_anchor_photometric_gain(
        np.array([0.1, 0.2, 0.4]),
        np.array([0.3, 0.1, 0.4]),
        gain_low=0.6,
        gain_high=2.0,
    )
    np.testing.assert_allclose(raw, [3.0, 0.5, 1.0], atol=1e-6)
    np.testing.assert_allclose(gain, [2.0, 0.6, 1.0], atol=1e-6)


def test_master_completion_marker_round_trip(tmp_path):
    assert not _queue_is_complete(tmp_path)
    assert _mark_master_complete(tmp_path, ["ac20", "ac24"])
    assert _queue_is_complete(tmp_path)

    payload = json.loads((tmp_path / "master_complete.json").read_text())
    assert payload["status"] == "complete"
    assert payload["runs"] == ["ac20", "ac24"]

    _clear_master_complete(tmp_path)
    assert not _queue_is_complete(tmp_path)


def test_loaded_worker_context_expires_only_after_idle_limit():
    cache = {("ac20",): object()}
    assert not _worker_context_is_idle(cache, 100.0, 300.0, now=399.9)
    assert _worker_context_is_idle(cache, 100.0, 300.0, now=400.0)
    assert not _worker_context_is_idle({}, 100.0, 300.0, now=1000.0)
    assert not _worker_context_is_idle(cache, None, 300.0, now=1000.0)
    assert not _worker_context_is_idle(cache, 100.0, 0.0, now=1000.0)


def test_watchdog_does_not_spawn_workers_for_completed_queue(tmp_path):
    queue = tmp_path / "queue"
    control = tmp_path / "control"
    assert _mark_master_complete(queue, ["ac20"])
    args = argparse.Namespace(
        queue=str(queue),
        workers=2,
        worker_id_prefix="test-host",
        worker_id=None,
        control_dir=str(control),
        worker_log_dir=None,
        thread_limit=1,
        worker_stall_seconds=600.0,
    )

    watchdog_main(args)

    assert not list((queue / "heartbeats").glob("*.json"))
    states = list(control.glob("*.watchdog.json"))
    assert len(states) == 1
    state = json.loads(states[0].read_text())
    assert state["desired_workers"] == 0
    assert state["workers_running"] == 0


def test_queue_master_accepts_optuna_seed():
    args = build_queue_parser().parse_args(
        [
            "master",
            "--queue",
            "queue",
            "--runs",
            "ac20",
            "--optuna-seed",
            "73",
        ]
    )
    assert args.optuna_seed == 73


def test_paired_seed_is_stable_per_run_and_controls_random_warmups():
    ac20_seed = _derived_run_seed(17, "ac20")
    assert ac20_seed == _derived_run_seed(17, "AC20")
    assert ac20_seed != _derived_run_seed(17, "ac24")
    assert ac20_seed != _derived_run_seed(73, "ac20")

    first = sample_params(PARAM_SPACE, rng=random.Random(ac20_seed))
    second = sample_params(PARAM_SPACE, rng=random.Random(ac20_seed))
    assert first == second


def test_final_geometry_campaign_is_paired_and_memory_bounded():
    variants = _select_variants("final_geometry")
    assert _select_runs("final_geometry") == FINAL_GEOMETRY_RUNS
    assert [(variant.template_registration, variant.optuna_seed) for variant in variants] == [
        ("off", 17),
        ("ac14_template", 17),
        ("off", 73),
        ("ac14_template", 73),
    ]
    assert [variant.template_strict for variant in variants] == [False, True, False, True]

    args = build_campaign_parser().parse_args(
        [
            "launch",
            "--variant",
            "final_geometry",
            "--run-set",
            "final_geometry",
            "--max-active-runs",
            "2",
            "--max-in-flight-per-run",
            "3",
            "--idle-exit-seconds",
            "120",
            "--threads-per-worker",
            "1",
        ]
    )
    master = _common_master_args(args, variants[0], "queue", "logs", "control")
    watchdog = _watchdog_args(args, variants[0], "queue", "control", workers=3)
    assert master[master.index("--optuna-seed") + 1] == "17"
    assert master[master.index("--max-active-runs") + 1] == "2"
    assert master[master.index("--max-in-flight-per-run") + 1] == "3"
    assert watchdog[watchdog.index("--idle-exit-seconds") + 1] == "120.0"
    assert watchdog[watchdog.index("--threads-per-worker") + 1] == "1"
