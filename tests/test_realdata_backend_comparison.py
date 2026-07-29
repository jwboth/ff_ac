from __future__ import annotations

from types import SimpleNamespace

from scripts.auto_calibrate_color_to_mass import EvalResult, Metrics
from scripts.compare_realdata_evaluation_backends import (
    compare_results,
    select_parameter_records,
)


def _record(record_id, value, objective):
    return {
        "id": record_id,
        "params": {"value": value},
        "objective": objective,
        "source": "test",
    }


def test_parameter_selection_includes_final_extrema_and_endpoints():
    final = _record("final", 99, 0.5)
    history = [_record(str(index), index, float(index)) for index in range(10)]

    selected = select_parameter_records(final, history, 5)

    assert [item["params"]["value"] for item in selected] == [99, 0, 9, 3, 6]


def test_backend_comparison_accepts_combined_tolerance():
    cpu = EvalResult(
        objective=1.0,
        feasible=True,
        status="ok",
        params={},
        metrics={"1.000h": Metrics(1.0, 2.0, 0.0, 2.0)},
    )
    gpu = EvalResult(
        objective=1.0 + 5e-8,
        feasible=True,
        status="ok",
        params={},
        metrics={"1.000h": Metrics(1.0, 2.0 + 1e-8, 5e-11, 2.0)},
    )

    comparison = compare_results(cpu, gpu, rtol=1e-6, atol=1e-10)

    assert comparison["passed"]
    assert comparison["failed_fields"] == []


def test_backend_comparison_reports_numerical_and_metadata_failures():
    cpu = EvalResult(
        objective=1.0,
        feasible=True,
        status="ok",
        params={},
        metrics={"1.000h": Metrics(1.0, 2.0)},
    )
    gpu = EvalResult(
        objective=1.01,
        feasible=False,
        status="non-finite-mass",
        params={},
        metrics={"1.000h": Metrics(1.0, 2.0)},
    )

    comparison = compare_results(cpu, gpu, rtol=1e-6, atol=1e-10)

    assert not comparison["passed"]
    assert not comparison["metadata_match"]
    assert "objective" in comparison["failed_fields"]
