from __future__ import annotations

import copy
from types import SimpleNamespace

import darsia
import numpy as np
import pytest

from scripts import auto_calibrate_color_to_mass
from scripts.ac_production_campaign import (
    PHASE_SCREEN_RUNS,
    _env_lines,
    _select_runs,
    _select_variants,
    _validate_final_campaign,
    build_parser as build_campaign_parser,
)
from scripts.auto_calibrate_color_to_mass import (
    _CudaIntegratedEvaluator,
    _mass_result_for_evaluation,
    _normalise_evaluation_backend,
    _normalise_phase_separation,
    _regularize_color_paths,
    evaluate_run,
    prepare_evaluation_context,
)
from scripts.distributed_auto_calibration_queue import (
    _generate_warmup_params,
    _task_payload,
)


class _ArrayImage:
    def __init__(self, values):
        self.img = np.asarray(values, dtype=np.float32)


class _Calibration:
    def __init__(self, result):
        self.result = result
        self.co2_mass_analysis = SimpleNamespace(
            density_gaseous_co2=np.array([[10.0, 10.0]]),
            solubility_co2=np.array([[2.0, 2.0]]),
        )

    def __call__(self, image):
        return self.result


class _PreparedCalibration:
    def __init__(self):
        self.color_calls = 0
        self.full_calls = 0
        self.downstream_calls = 0

    def call_color_interpretation(self, image):
        self.color_calls += 1
        return _ArrayImage(image.img + 1.0)

    def call_pH_analysis(self, color):
        self.downstream_calls += 1
        return _ArrayImage(color.img * 2.0)

    def call_flash_and_mass_analysis(self, signal):
        return SimpleNamespace(mass=_ArrayImage(signal.img))

    def __call__(self, image):
        self.full_calls += 1
        return SimpleNamespace(mass=_ArrayImage(image.img * 2.0))


def test_prepared_evaluation_reuses_color_projection_and_releases_rgb():
    calibration = _PreparedCalibration()
    context = SimpleNamespace(
        run="ac20",
        calibration=calibration,
        _loaded=[
            (_ArrayImage([[1.0, 2.0]]), 1.0, 0.5),
            (_ArrayImage([[3.0, 4.0]]), 2.0, 1.0),
        ],
        _prepared_colors=[],
        _evaluation_backend="legacy",
        phase_separation="shared-signal",
    )

    prepare_evaluation_context(context, backend="prepared", release_images=True)

    assert calibration.color_calls == 2
    assert context._evaluation_backend == "prepared"
    assert all(image is None for image, _injected, _time in context._loaded)

    first = _mass_result_for_evaluation(context, None, 0, {})
    second = _mass_result_for_evaluation(context, None, 0, {})

    assert calibration.full_calls == 0
    assert calibration.downstream_calls == 2
    np.testing.assert_allclose(first.mass.img, [[4.0, 6.0]])
    np.testing.assert_allclose(second.mass.img, first.mass.img)


def test_evaluation_backend_aliases_and_validation():
    assert _normalise_evaluation_backend("cpu") == "prepared"
    assert _normalise_evaluation_backend("fast") == "prepared"
    assert _normalise_evaluation_backend("off") == "legacy"
    assert _normalise_evaluation_backend("cuda") == "cuda"
    with pytest.raises(ValueError, match="Unknown evaluation backend"):
        _normalise_evaluation_backend("invalid")


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA/CuPy is not available")
@pytest.mark.parametrize(
    ("phase_separation", "gas_scores", "expected"),
    [
        ("shared-signal", [], (2.6875, 1.0, 1.6875)),
        (
            "residual-gas",
            [np.array([[0, 255]], dtype=np.uint8)],
            (3.125, 2.0, 1.125),
        ),
    ],
)
def test_cuda_integrated_evaluator_matches_phase_algebra(
    phase_separation,
    gas_scores,
    expected,
):
    class _Geometry:
        cached_voxel_volume = np.ones((1, 2), dtype=np.float64)

        def _prepare_cached_voxel_volume(self, _shape):
            return None

    class _ExpertAdapter:
        def mask_for(self, _image, mode):
            if mode == "saturation_g":
                return np.array([[True, False]])
            return np.array([[True, True]])

    signal_model = SimpleNamespace(
        supports=np.array([0.0, 1.0]),
        values=np.array([0.0, 1.0]),
    )
    calibration = SimpleNamespace(
        labels=_ArrayImage([[1, 1]]),
        signal_model=SimpleNamespace(model=[None, {1: signal_model}]),
        flash=SimpleNamespace(
            min_value_aq=0.0,
            max_value_aq=1.0,
            min_value_g=0.5,
            max_value_g=1.0,
            restoration=None,
        ),
        co2_mass_analysis=SimpleNamespace(
            density_gaseous_co2=np.full((1, 2), 2.0),
            solubility_co2=np.full((1, 2), 1.5),
        ),
        expert_knowledge_adapter=_ExpertAdapter(),
    )
    context = SimpleNamespace(
        run="test",
        calibration=calibration,
        geometry=_Geometry(),
        signal_labels=[1],
        phase_separation=phase_separation,
        _prepared_colors=[_ArrayImage([[0.75, 0.75]])],
        _gas_scores=gas_scores,
    )

    evaluator = _CudaIntegratedEvaluator(context)
    try:
        result = evaluator.evaluate(
            {"gas.residual_onset": 0.2, "gas.residual_width": 0.4}
        )
    finally:
        evaluator.close()

    np.testing.assert_allclose(result[0], expected, rtol=1e-7, atol=1e-12)


def test_window_balanced_objective_gives_each_operational_window_equal_weight(
    monkeypatch,
):
    monkeypatch.setattr(
        auto_calibrate_color_to_mass,
        "apply_params",
        lambda *args, **kwargs: None,
    )
    calibration = lambda detected: SimpleNamespace(mass=detected)
    context = SimpleNamespace(
        run="ac20",
        _loaded=[
            (1.0, 0.0, 0.5),   # I1 error 1
            (3.0, 0.0, 0.8),   # I1 error 3 -> mean 2
            (4.0, 0.0, 2.0),   # I2 mean 4
            (2.0, 0.0, 5.0),   # early error 2
            (6.0, 0.0, 7.0),   # early error 6 -> mean 4
            (1.0, 0.0, 12.0),  # late mean 1
        ],
        calibration=calibration,
        geometry=SimpleNamespace(integrate=float),
        enforce_lower=False,
        objective_integral="window-balanced",
        phase_separation="shared-signal",
        signal_labels=[],
    )

    result = evaluate_run(context, {})

    assert result.status == "ok"
    assert result.objective == pytest.approx(2.0 + 4.0 + 4.0 + 1.0)


def test_residual_gas_replaces_only_the_gas_saturation_branch():
    result = SimpleNamespace(
        concentration_aq=_ArrayImage([[1.0, 1.0]]),
        saturation_g=_ArrayImage([[0.0, 0.0]]),
        mass_g=_ArrayImage([[0.0, 0.0]]),
        mass_aq=_ArrayImage([[0.0, 0.0]]),
        mass=_ArrayImage([[0.0, 0.0]]),
    )
    calibration = _Calibration(result)
    context = SimpleNamespace(
        run="ac20",
        calibration=calibration,
        phase_separation="residual-gas",
        _gas_scores=[np.array([[0, 255]], dtype=np.uint8)],
    )

    output = _mass_result_for_evaluation(
        context,
        image=None,
        image_index=0,
        params={
            "gas.residual_onset": 0.0,
            "gas.residual_width": 2.0,
        },
    )

    np.testing.assert_allclose(output.saturation_g.img, [[0.0, 1.0]])
    np.testing.assert_allclose(output.mass_g.img, [[0.0, 10.0]])
    np.testing.assert_allclose(output.mass_aq.img, [[2.0, 0.0]])
    np.testing.assert_allclose(output.mass.img, [[2.0, 10.0]])


def test_color_path_regularization_updates_the_executable_heterogeneous_copy(
    tmp_path,
):
    results = tmp_path / "results" / "ac20"
    calibration_folder = (
        results
        / "calibration"
        / "color"
        / "relative_colorpath"
        / "color_to_mass"
        / "from_facies"
    )
    relative_folder = calibration_folder.relative_to(results)
    anchor_folder = (
        results.parent
        / "ac60"
        / relative_folder
        / "color_path_interpretation"
    )
    target_path = darsia.ColorPath(
        base_color=np.zeros(3),
        relative_colors=[np.zeros(3), np.array([1.0, 0.0, 0.0])],
    )
    anchor_path = darsia.ColorPath(
        base_color=np.zeros(3),
        relative_colors=[np.zeros(3), np.array([0.0, 1.0, 0.0])],
    )
    public_interpolation = darsia.ColorPathInterpolation(
        target_path,
        darsia.ColorMode.RELATIVE,
    )
    executable_interpolation = copy.copy(public_interpolation)
    darsia.ColorPathInterpolation(
        anchor_path,
        darsia.ColorMode.RELATIVE,
    ).save(anchor_folder / "color_path_interpretation_1")
    calibration = SimpleNamespace(
        color_path_interpretation={1: public_interpolation},
        color_analysis=SimpleNamespace(model=[{1: executable_interpolation}]),
    )

    _regularize_color_paths(
        run="ac20",
        config=SimpleNamespace(data=SimpleNamespace(results=results)),
        calibration=calibration,
        calibration_folder=calibration_folder,
        labels=[1],
        anchor_run="ac60",
        anchor_weight=1.0,
        anchor_strict=True,
    )

    assert public_interpolation.color_path is executable_interpolation.color_path
    np.testing.assert_allclose(
        executable_interpolation.color_path.relative_colors[-1],
        [0.0, 1.0, 0.0],
    )


def test_phase_screen_is_paired_and_physically_stepwise(tmp_path):
    variants = _select_variants("phase_screen")

    assert _select_runs("phase_screen") == PHASE_SCREEN_RUNS
    assert len(variants) == 3
    assert [variant.objective_integral for variant in variants] == [
        "off",
        "window-balanced",
        "window-balanced",
    ]
    assert variants[0].color_path_anchor == "off"
    assert variants[1].color_path_anchor == "ac60"
    assert variants[2].phase_separation == "residual-gas"
    assert all(variant.optuna_seed == 17 for variant in variants)
    assert all(variant.template_registration == "ac14_template" for variant in variants)

    residual_env = _env_lines(variants[2], 6.0)
    assert "$env:FFAC_COLOR_PATH_ANCHOR = 'ac60'" in residual_env
    assert "$env:FFAC_PHASE_SEPARATION = 'residual-gas'" in residual_env

    seeds = tmp_path / "seeds.json"
    seeds.write_text("{}", encoding="utf-8")
    args = build_campaign_parser().parse_args(
        [
            "launch",
            "--variant",
            "phase_screen",
            "--run-set",
            "phase_screen",
            "--max-iters",
            "800",
            "--warmup-iters",
            "150",
            "--max-active-runs",
            "6",
            "--max-in-flight-per-run",
            "1",
            "--seed-params-file",
            str(seeds),
        ]
    )
    _validate_final_campaign(args, variants, PHASE_SCREEN_RUNS)


def test_residual_gas_warmup_includes_a_targeted_threshold_grid():
    context = SimpleNamespace(
        param_space=[
            {
                "name": "gas.residual_onset",
                "bounds": (0.01, 0.8),
                "type": "float",
            },
            {
                "name": "gas.residual_width",
                "bounds": (0.02, 1.5),
                "type": "float",
            },
        ],
        signal_label=None,
        per_label_params=True,
    )

    warmups = _generate_warmup_params(
        context,
        warmup_iters=0,
        warmup_levels=None,
        warmup_levels_by_idx=None,
        warmup_levels_default=None,
        warmup_high=None,
        warmup_mode="prefix",
    )
    pairs = {
        (
            candidate["gas.residual_onset"],
            candidate["gas.residual_width"],
        )
        for candidate in warmups
    }

    assert (0.02, 0.05) in pairs
    assert (0.4, 0.8) in pairs
    assert len(pairs) >= 25


def test_queue_payload_freezes_phase_and_anchor_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("FFAC_PHASE_SEPARATION", "residual-gas")
    monkeypatch.setenv("FFAC_COLOR_PATH_ANCHOR", "ac60")
    monkeypatch.setenv("FFAC_COLOR_PATH_ANCHOR_WEIGHT", "0.75")
    monkeypatch.setenv("FFAC_COLOR_PATH_ANCHOR_STRICT", "on")

    payload = _task_payload(
        "task",
        "ac20",
        "optuna",
        {"a": 1.0},
        1,
        tmp_path,
        True,
        True,
        False,
        None,
        False,
        "window-balanced",
        None,
    )

    assert payload["phase_separation"] == "residual-gas"
    assert payload["color_path_anchor"] == "ac60"
    assert payload["color_path_anchor_weight"] == pytest.approx(0.75)
    assert payload["color_path_anchor_strict"] is True


def test_unknown_phase_separation_is_rejected():
    assert _normalise_phase_separation("residual") == "residual-gas"
    with pytest.raises(ValueError, match="Unknown phase separation"):
        _normalise_phase_separation("unphysical")
