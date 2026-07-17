"""Summarise the paired full-budget AC14 template calibration campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median


DEFAULT_ROOT = Path(r"Z:\Albus\Autokalibrering_log\final_geometry_20260717")
EXPECTED = {
    "final_baseline_seed17": ("baseline", 17),
    "final_template_ac14_seed17": ("template", 17),
    "final_baseline_seed73": ("baseline", 73),
    "final_template_ac14_seed73": ("template", 73),
}
EXPECTED_RUNS = ("ac20", "ac24", "ac40", "ac44", "ac52", "ac60")


@dataclass(frozen=True)
class Result:
    variant: str
    method: str
    seed: int
    run: str
    objective: float
    plateau_mae: float
    end_ratio: float
    late_ratio: float
    final_ratio: float
    plateau_tv: float
    source: str


def _ratio_rows(payload: dict) -> list[tuple[float, float, float, float]]:
    rows = []
    for label, values in payload["metrics"].items():
        time_h = float(str(label).removesuffix("h"))
        injected = float(values["injected_full"])
        detected = float(values["total_full"])
        rows.append((time_h, injected, detected, detected / injected))
    return sorted(rows)


def _read_result(path: Path, variant: str, method: str, seed: int) -> Result:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _ratio_rows(payload)
    plateau = [row for row in rows if row[0] >= 2.497 - 1e-3]
    late = [row for row in rows if row[0] >= 9.417 - 1e-3]
    injected_final = plateau[0][1]
    plateau_tv = sum(
        abs(current[2] - previous[2])
        for previous, current in zip(plateau, plateau[1:])
    ) / injected_final
    return Result(
        variant=variant,
        method=method,
        seed=seed,
        run=str(payload["run"]).lower(),
        objective=float(payload["objective_full_scale"]),
        plateau_mae=mean(abs(row[3] - 1.0) for row in plateau),
        end_ratio=min(rows, key=lambda row: abs(row[0] - 2.497))[3],
        late_ratio=mean(row[3] for row in late),
        final_ratio=rows[-1][3],
        plateau_tv=plateau_tv,
        source=str(path),
    )


def collect(root: Path) -> list[Result]:
    results = []
    for variant, (method, seed) in EXPECTED.items():
        variant_root = root / variant
        newest_by_run: dict[str, Path] = {}
        if variant_root.exists():
            for path in variant_root.rglob("final_full_scale_ac*.json"):
                run = path.stem.removeprefix("final_full_scale_").lower()
                current = newest_by_run.get(run)
                if current is None or path.stat().st_mtime > current.stat().st_mtime:
                    newest_by_run[run] = path
        for run, path in newest_by_run.items():
            results.append(_read_result(path, variant, method, seed))
    return results


def _fmt_pct(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def print_report(results: list[Result]) -> None:
    expected_count = len(EXPECTED) * len(EXPECTED_RUNS)
    print(f"Completed: {len(results)}/{expected_count}")
    lookup = {(row.seed, row.method, row.run): row for row in results}
    print("\nPaired effect (negative delta means template is better):")
    for seed in sorted({seed for _, seed in EXPECTED.values()}):
        pairs = [
            (lookup[(seed, "baseline", run)], lookup[(seed, "template", run)])
            for run in EXPECTED_RUNS
            if (seed, "baseline", run) in lookup and (seed, "template", run) in lookup
        ]
        deltas = [template.plateau_mae - baseline.plateau_mae for baseline, template in pairs]
        if not deltas:
            print(f"  seed {seed}: no complete pairs")
            continue
        print(
            f"  seed {seed}: template improved {sum(delta < 0 for delta in deltas)}/{len(deltas)}, "
            f"median delta {100.0 * median(deltas):+.1f} percentage points"
        )

    print("\nMean across seeds:")
    print("run   baseline  template  delta")
    for run in EXPECTED_RUNS:
        baseline = [row.plateau_mae for row in results if row.run == run and row.method == "baseline"]
        template = [row.plateau_mae for row in results if row.run == run and row.method == "template"]
        if not baseline or not template:
            print(f"{run:<5} incomplete")
            continue
        base_mean = mean(baseline)
        template_mean = mean(template)
        print(
            f"{run:<5} {_fmt_pct(base_mean):>8} {_fmt_pct(template_mean):>9} "
            f"{100.0 * (template_mean - base_mean):+6.1f} pp"
        )


def write_outputs(root: Path, results: list[Result]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "final_geometry_summary.json"
    csv_path = root / "final_geometry_summary.csv"
    json_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    fields = list(Result.__dataclass_fields__)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    results = collect(args.root)
    print_report(results)
    if args.write:
        write_outputs(args.root, results)
    if args.require_complete and len(results) != len(EXPECTED) * len(EXPECTED_RUNS):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
