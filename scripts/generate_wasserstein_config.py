"""Generate the multi-run Wasserstein config for the new DarSIA comparison.

The new DarSIA already performs the cross-run W1 computation, config-driven, via
``comparison_wasserstein`` (``comparison.py --wasserstein-compute/-assemble``).
This script emits the ``[run.*]`` / ``[roi.*]`` / ``[wasserstein]`` multi-config
it consumes, listing every AC run, the ROIs, the resize factor, and the shared
time grid (dense in injection, exponentially coarser in rest - the same grid
written into each per-run config, so all experiments are compared at the same
time-since-injection).

``distributed_wasserstein_queue.py`` then parallelises the (otherwise sequential)
DarSIA compute by running several workers with ``skip_existing`` against this one
config.

ROI geometry: ``box1``/``box2`` split at the midpoint between the two injection
ports (port1 x=0.45 / I1, port2 x=0.72 / I2), so box1 is the I1 region and box2
the I2 region. Refine corner_1/corner_2 if exact layer boundaries are needed.

Usage
-----
    python scripts/generate_wasserstein_config.py \
        --runs ac17 ac18 ... ac61 --out config/wasserstein_ac.toml \
        --results-root "E:\\ff_ml4gcs" --resize 0.10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Rig extent [m] (from common.toml [rig]); full-domain ROI.
RIG_W, RIG_H = 0.90, 0.49
# Injection port x-coordinates [m] (ac14 protocol): port1/I1, port2/I2.
PORT1_X, PORT2_X = 0.45, 0.72

# Shared time grid (must match TIME_GRID in generate_configs.py).
TIME_GRID = [
    # Dense through BOTH injections (I1 ~0-0.93 h, port switch, I2 ~1.0-2.4 h),
    # then exponentially coarser during rest. t=0 is EXACT: DarSIA sets
    # experiment_start = injection_protocol["start"].min() = start of ramp-up
    # (first gas, 0.1 ml/min), so "5 min" = 5 min after ramp-up start.
    ("inj1_fine",  "00:00:00", "00:10:00", 21, "00:00:15"),  # I1 onset, ~30 s
    ("inj1_body",  "00:10:00", "00:55:00", 16, "00:01:30"),  # I1 body, ~3 min
    ("inj2_onset", "00:55:00", "01:15:00", 11, "00:01:00"),  # port switch + I2 onset, ~2 min
    ("inj2_body",  "01:15:00", "02:30:00", 16, "00:02:30"),  # I2 body, ~5 min
    ("rest_early", "02:30:00", "06:00:00", 15, "00:07:00"),  # post-injection, ~15 min
    ("rest_mid",   "06:00:00", "24:00:00", 13, "01:30:00"),  # ~1.5 h
    ("rest_late",  "24:00:00", "120:00:00", 13, "08:00:00"),  # ~8 h
]


def tomlp(p: str) -> str:
    return p.replace("\\", "\\\\")


def build(runs: list, results_root: str, resize: float, rois: list,
          run_dir: str, common: str) -> str:
    res = f"{results_root}\\wasserstein"
    L = []
    L.append("##################################################")
    L.append("# AC cross-run Wasserstein multi-config (generated)")
    L.append("##################################################")
    L.append("[data]")
    L.append(f'results = "{tomlp(res)}"')
    L.append("")
    L.append("# Common config added to every run, then each per-run config.")
    L.append("[run.common]")
    L.append(f'config = ["{common}"]')
    L.append("")
    for r in runs:
        L.append(f"[run.{r}]")
        L.append(f'config = "{run_dir}/{r}.toml"')
    L.append("")
    # Boxes split at the midpoint between the two injection ports
    # (port1 x=0.45 / I1, port2 x=0.72 / I2 -> midpoint ~0.585), so box1 covers
    # the first-injection (I1) region and box2 the second-injection (I2) region.
    split = round((PORT1_X + PORT2_X) / 2, 3)
    L.append("# ROIs. full = whole domain; box1 = I1 region, box2 = I2 region.")
    L.append("[roi.full]")
    L.append('name = "Full Domain"')
    L.append("corner_1 = [0.0, 0.0]")
    L.append(f"corner_2 = [{RIG_W}, {RIG_H}]")
    L.append("[roi.box1]")
    L.append('name = "Box 1 (I1 / port 1 region)"')
    L.append("corner_1 = [0.0, 0.0]")
    L.append(f"corner_2 = [{split}, {RIG_H}]")
    L.append("[roi.box2]")
    L.append('name = "Box 2 (I2 / port 2 region)"')
    L.append(f"corner_1 = [{split}, 0.0]")
    L.append(f"corner_2 = [{RIG_W}, {RIG_H}]")
    L.append("")
    L.append("[wasserstein]")
    L.append("runs = [" + ", ".join(f'"{r}"' for r in runs) + "]")
    L.append("roi = [" + ", ".join(f'"{r}"' for r in rois) + "]")
    L.append(f"resize = {resize}")
    L.append(f'results = "{tomlp(res)}"')
    L.append("")
    L.append("# Shared time grid (hours since experiment start; HH:MM:SS).")
    for name, start, end, num, tol in TIME_GRID:
        L.append(f"[wasserstein.data.interval.{name}]")
        L.append(f'start = "{start}"')
        L.append(f'end = "{end}"')
        L.append(f"num = {num}")
        L.append(f'tol = "{tol}"')
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=[],
                    help="run ids, e.g. ac17 ac18 ... (default: all in --run-dir)")
    ap.add_argument("--run-dir", default="run_ac",
                    help="per-run config dir relative to the multi-config")
    ap.add_argument("--run-config-dir", default="config/run_ac",
                    help="actual dir to scan when --runs is omitted")
    ap.add_argument("--common", default="common.toml",
                    help="common config path relative to the multi-config")
    ap.add_argument("--out", default="config/wasserstein_ac.toml", type=Path)
    ap.add_argument("--results-root", default="E:\\ff_ml4gcs")
    ap.add_argument("--resize", type=float, default=0.10)
    ap.add_argument("--rois", nargs="+", default=["full", "box1", "box2"])
    args = ap.parse_args()

    runs = args.runs
    if not runs:
        d = Path(args.run_config_dir)
        runs = sorted([p.stem for p in d.glob("ac*.toml")],
                      key=lambda s: int(re.sub(r"\D", "", s)))
    text = build(runs, args.results_root, args.resize, args.rois,
                 args.run_dir, args.common)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out} with {len(runs)} runs, rois={args.rois}, resize={args.resize}")


if __name__ == "__main__":
    main()
