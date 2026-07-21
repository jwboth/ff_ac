from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ac_production_campaign import (
    FINAL_GEOMETRY_RUNS,
    _env_lines,
    _select_runs,
    _select_variants,
)
from scripts.auto_calibrate_color_to_mass import (
    _normalise_signal_parameterization,
    apply_params,
    build_param_space,
    load_bounds_map,
)
from scripts.distributed_auto_calibration_queue import _generate_warmup_params


ROOT = Path(__file__).resolve().parents[1]
BOUNDS = ROOT / "config" / "bounds_seg6_reduced_shared_shape.json"
ACTIVE_LABELS = [1, 2, 5, 7, 8]


class _SignalFunction:
    def __init__(self, size: int):
        self.values = np.zeros(size, dtype=float)

    def update(self, *, values):
        self.values = np.asarray(values, dtype=float)


def _calibration(labels: list[int], size: int):
    functions = {label: _SignalFunction(size) for label in labels}
    signal_model = SimpleNamespace(model={1: functions})
    return SimpleNamespace(signal_model=signal_model, flash=None), functions


def test_shared_shape_space_is_identifiable_and_reduced():
    space = build_param_space(
        "ac20",
        load_bounds_map(BOUNDS),
        signal_labels=ACTIVE_LABELS,
        per_label_params=True,
        n_free_values=6,
        signal_parameterization="shared-shape",
    )
    entries = {entry["name"]: entry for entry in space}

    assert entries["signal.shared.value0"]["bounds"] == (0.0, 0.0)
    assert entries["signal.shared.value6"]["bounds"] == (1.0, 1.0)
    assert {name for name in entries if name.endswith(".gain")} == {
        f"signal.label{label}.gain" for label in ACTIVE_LABELS
    }
    assert not any(name.startswith("signal.label1.value") for name in entries)

    free = [entry for entry in space if entry["bounds"][0] != entry["bounds"][1]]
    assert len(free) == 12  # five shape fractions, five gains, two flash values


def test_shared_shape_applies_one_shape_with_per_label_amplitudes():
    calibration, functions = _calibration([1, 5], 4)
    params = {
        "signal.shared.value0": 0.0,
        "signal.shared.value1": 0.25,
        "signal.shared.value2": 0.5,
        "signal.shared.value3": 1.0,
        "signal.label1.gain": 2.0,
        "signal.label5.gain": 1.0,
    }

    updated = apply_params(calibration, params, labels=[1, 5], np_module=np)

    assert updated == 2
    assert functions[1].values == pytest.approx([0.0, 0.5, 1.0, 2.0])
    assert functions[5].values == pytest.approx([0.0, 0.25, 0.5, 1.0])


def test_reduced_warmups_include_predefined_and_random_candidates():
    space = build_param_space(
        "ac20",
        load_bounds_map(BOUNDS),
        signal_labels=ACTIVE_LABELS,
        per_label_params=True,
        n_free_values=6,
        signal_parameterization="shared-shape",
    )
    context = SimpleNamespace(
        param_space=space,
        signal_label=None,
        per_label_params=True,
    )

    warmups = _generate_warmup_params(
        context,
        warmup_iters=150,
        warmup_levels=None,
        warmup_levels_by_idx=None,
        warmup_levels_default=None,
        warmup_high=None,
        warmup_mode="single",
    )

    assert len(warmups) > 150
    assert all(candidate["signal.shared.value6"] == 1.0 for candidate in warmups)


def test_reduced_campaign_is_paired_with_ac14_template():
    variants = _select_variants("reduced_model")

    assert _select_runs("reduced_model") == FINAL_GEOMETRY_RUNS
    assert [variant.optuna_seed for variant in variants] == [17, 73]
    assert all(variant.template_registration == "ac14_template" for variant in variants)
    assert all(variant.template_strict for variant in variants)
    assert all(variant.signal_parameterization == "shared-shape" for variant in variants)
    assert all(str(BOUNDS.name) in variant.bounds_file for variant in variants)
    assert all(
        "$env:FFAC_SIGNAL_PARAMETERIZATION = 'shared-shape'" in _env_lines(variant, 6.0)
        for variant in variants
    )


def test_unknown_signal_parameterization_is_rejected():
    with pytest.raises(ValueError, match="Unknown signal parameterization"):
        _normalise_signal_parameterization("free-form")
