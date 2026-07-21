"""Compare the reduced shared-shape campaign with the paired full model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median


DEFAULT_FULL_ROOT = Path(r"Z:\Albus\Autokalibrering_log\final_geometry_20260717")
DEFAULT_REDUCED_ROOT = Path(r"Z:\Albus\Autokalibrering_log\reduced_model_20260721")
RUNS = ("ac20", "ac24", "ac40", "ac44", "ac52", "ac60")
FULL_VARIANTS = {
    "final_template_ac14_seed17": 17,
    "final_template_ac14_seed73": 73,
}
REDUCED_VARIANTS = {
    "reduced_template_ac14_seed17": 17,
    "reduced_template_ac14_seed73": 73,
}


@dataclass(frozen=True)
class Result:
    model: str
    variant: str
    seed: int
    run: str
    objective: float
    plateau_mae: float
    late_ratio: float
    final_ratio: float
    plateau_tv: float
    source: str


def _read_result(path: Path, model: str, variant: str, seed: int) -> Result:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = []
    for label, values in payload["metrics"].items():
        time_h = float(str(label).removesuffix("h"))
        injected = float(values["injected_full"])
        detected = float(values["total_full"])
        rows.append((time_h, injected, detected, detected / injected))
    rows.sort()
    plateau = [row for row in rows if row[0] >= 2.497 - 1e-3]
    late = [row for row in rows if row[0] >= 9.417 - 1e-3]
    injected_final = plateau[0][1]
    plateau_tv = sum(
        abs(current[2] - previous[2])
        for previous, current in zip(plateau, plateau[1:])
    ) / injected_final
    return Result(
        model=model,
        variant=variant,
        seed=seed,
        run=str(payload["run"]).lower(),
        objective=float(payload["objective_full_scale"]),
        plateau_mae=mean(abs(row[3] - 1.0) for row in plateau),
        late_ratio=mean(row[3] for row in late),
        final_ratio=rows[-1][3],
        plateau_tv=plateau_tv,
        source=str(path),
    )


def _collect_family(root: Path, model: str, variants: dict[str, int]) -> list[Result]:
    results = []
    for variant, seed in variants.items():
        newest_by_run: dict[str, Path] = {}
        variant_root = root / variant
        if variant_root.exists():
            for path in variant_root.rglob("final_full_scale_ac*.json"):
                run = path.stem.removeprefix("final_full_scale_").lower()
                current = newest_by_run.get(run)
                if current is None or path.stat().st_mtime > current.stat().st_mtime:
                    newest_by_run[run] = path
        for run, path in newest_by_run.items():
            if run in RUNS:
                results.append(_read_result(path, model, variant, seed))
    return results


def collect(full_root: Path, reduced_root: Path) -> list[Result]:
    return [
        *_collect_family(full_root, "full", FULL_VARIANTS),
        *_collect_family(reduced_root, "reduced", REDUCED_VARIANTS),
    ]


def _fmt_pct(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def print_report(results: list[Result]) -> None:
    reduced = [row for row in results if row.model == "reduced"]
    print(f"Reduced completed: {len(reduced)}/{len(RUNS) * len(REDUCED_VARIANTS)}")
    lookup = {(row.seed, row.model, row.run): row for row in results}
    print("\nPaired effect (negative delta means reduced is better):")
    for seed in sorted(REDUCED_VARIANTS.values()):
        pairs = [
            (lookup[(seed, "full", run)], lookup[(seed, "reduced", run)])
            for run in RUNS
            if (seed, "full", run) in lookup and (seed, "reduced", run) in lookup
        ]
        deltas = [reduced_row.plateau_mae - full.plateau_mae for full, reduced_row in pairs]
        if not deltas:
            print(f"  seed {seed}: no complete pairs")
            continue
        print(
            f"  seed {seed}: reduced improved {sum(delta < 0 for delta in deltas)}/{len(deltas)}, "
            f"median delta {100.0 * median(deltas):+.1f} percentage points"
        )

    print("\nMean across seeds:")
    print("run     full  reduced   delta  full-seed  reduced-seed")
    for run in RUNS:
        full = [row.plateau_mae for row in results if row.run == run and row.model == "full"]
        reduced_run = [
            row.plateau_mae for row in results if row.run == run and row.model == "reduced"
        ]
        if not full or not reduced_run:
            print(f"{run:<5} incomplete")
            continue
        full_mean = mean(full)
        reduced_mean = mean(reduced_run)
        full_spread = abs(full[0] - full[1]) if len(full) == 2 else math.nan
        reduced_spread = (
            abs(reduced_run[0] - reduced_run[1]) if len(reduced_run) == 2 else math.nan
        )
        print(
            f"{run:<5} {_fmt_pct(full_mean):>7} {_fmt_pct(reduced_mean):>8} "
            f"{100.0 * (reduced_mean - full_mean):+7.1f} pp "
            f"{_fmt_pct(full_spread):>10} {_fmt_pct(reduced_spread):>13}"
        )


def write_outputs(root: Path, results: list[Result]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (root / "reduced_model_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (root / "reduced_model_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Result.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--reduced-root", type=Path, default=DEFAULT_REDUCED_ROOT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    results = collect(args.full_root, args.reduced_root)
    print_report(results)
    if args.write:
        write_outputs(args.reduced_root, results)
    completed = sum(row.model == "reduced" for row in results)
    if args.require_complete and completed != len(RUNS) * len(REDUCED_VARIANTS):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
