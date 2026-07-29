from __future__ import annotations

import os
from pathlib import Path

from scripts.ac_production_campaign import (
    EXPECTED_RUNTIME_DEPTH_SHA256,
    FINAL_PRODUCTION_RUNS,
)
from scripts.launch_final_five_campaigns import (
    CAMPAIGN_ID,
    VARIANTS,
    _active_campaign_processes,
    _variant_env,
    build_process_specs,
    cpu_runs,
    gpu_lanes,
    gpu_runs,
    validate_plan,
)


def test_final_five_plan_covers_every_run_once_per_variant():
    plan = validate_plan()

    assert plan["campaign_id"] == CAMPAIGN_ID
    assert plan["run_count"] == 40
    assert plan["variant_count"] == 5
    assert plan["calibration_count"] == 200
    assert plan["cpu_assignments"] == 24
    assert plan["gpu_assignments"] == 176
    assert plan["cpu_workers_per_host"] == 12
    assert plan["gpu_workers"] == {"moderskipet": 3, "olav": 3}

    assignments = {
        (row["variant"], row["run"]) for row in plan["assignments"]
    }
    assert assignments == {
        (variant.name, run)
        for variant in VARIANTS
        for run in FINAL_PRODUCTION_RUNS
    }


def test_cpu_and_gpu_run_lists_are_disjoint_and_complete():
    lanes = gpu_lanes()
    for variant in VARIANTS:
        cpu = set(cpu_runs(variant))
        gpu = set(gpu_runs(variant))
        lane_runs = {
            run
            for lane in lanes
            if lane.variant == variant.name
            for run in lane.runs
        }
        assert not cpu.intersection(gpu)
        assert cpu | gpu == set(FINAL_PRODUCTION_RUNS)
        assert lane_runs == gpu


def test_process_specs_enforce_full_budget_depth_and_sequential_optuna(
    tmp_path: Path,
):
    seed = tmp_path / "seeds.json"
    moderskipet = build_process_specs(
        role="moderskipet",
        logs_root=tmp_path / "logs",
        queue_root=r"\\server\queue\campaign",
        seed_path=seed,
    )
    olav = build_process_specs(
        role="olav",
        logs_root=tmp_path / "logs",
        queue_root=r"\\server\queue\campaign",
        seed_path=seed,
    )

    assert sum(spec.kind == "watchdog" for spec in moderskipet) == 5
    assert sum(spec.kind == "watchdog" for spec in olav) == 5
    assert sum(spec.kind == "gpu" for spec in moderskipet) == 4
    assert sum(spec.kind == "gpu" for spec in olav) == 3
    deferred = [spec for spec in moderskipet if spec.start_after]
    assert len(deferred) == 1
    assert deferred[0].name == "gpu_cuda_sharedpath_s73"
    assert deferred[0].start_after == "gpu_cuda_control_s17"

    all_specs = (*moderskipet, *olav)
    for spec in all_specs:
        if spec.kind in {"master", "gpu"}:
            command = list(spec.command)
            assert command[command.index("--max-iters") + 1] == "1600"
            assert command[command.index("--warmup-iters") + 1] == "150"
            assert command[command.index("--max-in-flight-per-run") + 1] == "1"
            assert "--no-save-calibration" in command
            if spec.kind == "gpu":
                assert "--optuna-persist" not in command
            else:
                assert command[command.index("--optuna-persist") + 1] == "true"
        assert spec.env["FFAC_REQUIRE_VARYING_DEPTH"] == "on"
        assert (
            spec.env["FFAC_EXPECTED_DEPTH_SHA256"]
            == EXPECTED_RUNTIME_DEPTH_SHA256
        )


def test_only_sharedpath_variants_enable_ac60_anchor():
    for variant in VARIANTS:
        env = _variant_env(variant, master=True)
        if variant.color_path_anchor:
            assert env["FFAC_COLOR_PATH_ANCHOR"] == "ac60"
            assert env["FFAC_COLOR_PATH_ANCHOR_STRICT"] == "on"
        else:
            assert "FFAC_COLOR_PATH_ANCHOR" not in env


def test_active_campaign_scan_ignores_its_own_process_tree():
    assert all(
        process["pid"] not in {os.getpid(), os.getppid()}
        for process in _active_campaign_processes()
    )
