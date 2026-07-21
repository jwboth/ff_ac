from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from scripts.analyze_temporal_image_stability import (
    Frame,
    _calibration_targets,
    _fit_spatial_gain,
    _mark_calibration_frames,
    _parse_duration_seconds,
    _sample_indices,
)


def test_parse_duration_supports_hours_beyond_one_day():
    assert _parse_duration_seconds("48:00:00") == 48 * 3600


def test_sample_indices_always_keeps_required_calibration_frames():
    selected = _sample_indices(
        1000,
        step=10,
        max_frames=5,
        required={123, 456, 789},
    )
    assert {0, 123, 456, 789, 999}.issubset(selected)
    assert len(selected) == 8


def test_marks_thirteen_mass_calibration_times():
    cfg = {
        "calibration": {"mass": {"data": ["calibration1", "calibration2"]}},
        "data": {
            "interval": {
                "calibration1": {
                    "start": "00:10:00",
                    "end": "02:30:00",
                    "num": 5,
                    "tol": "00:01:00",
                },
                "calibration2": {
                    "start": "03:00:00",
                    "end": "48:00:00",
                    "num": 8,
                    "tol": "00:05:00",
                },
            }
        },
    }
    start = datetime(2026, 1, 1, 12, 0)
    targets = _calibration_targets(cfg)
    frames = [
        Frame(start + timedelta(seconds=seconds), f"f{index}.jpg", Path(f"f{index}.jpg"))
        for index, (seconds, _tolerance) in enumerate(targets)
    ]
    marked, indices = _mark_calibration_frames(frames, start, targets)
    assert len(targets) == 13
    assert len(indices) == 13
    assert all(frame.is_calibration for frame in marked)


def test_spatial_gain_fit_recovers_horizontal_and_vertical_gradient():
    height, width = 180, 300
    yy, xx = np.mgrid[0:height, 0:width]
    x = (xx + 0.5) / width - 0.5
    y = (yy + 0.5) / height - 0.5
    gain = 1.0 + 0.10 * x + 0.06 * y
    reference = np.full((height, width, 3), 100, dtype=np.uint8)
    current = np.clip(reference.astype(float) / gain[..., None], 0, 255).astype(np.uint8)
    mask = np.ones((height, width), dtype=bool)

    global_change, gradient_x, gradient_y, residual = _fit_spatial_gain(
        reference,
        current,
        mask,
    )

    assert abs(global_change) < 1.0
    assert np.isclose(gradient_x, 10.0, atol=1.0)
    assert np.isclose(gradient_y, 6.0, atol=1.0)
    assert residual < 1.0
