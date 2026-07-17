from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.auto_calibrate_color_to_mass import (
    _compute_anchor_photometric_gain,
    _porous_baseline_median,
)
from scripts.distributed_auto_calibration_queue import (
    _fixed_params_for_run,
    _load_fixed_params_file,
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
