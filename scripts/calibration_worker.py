"""Per-trial calibration worker - wired to the DarSIA mass-balance objective.

Invoked once per calibration trial by ``distributed_auto_calibration_queue.py``.
Applies the trial ``--params`` to the color-to-mass signal functions, runs the
mass analysis over the calibration images and prints the mass-balance objective
(lower = better) as the LAST line of stdout.

Requires DarSIA (``uv sync``); the heavy logic lives in ``calibration_objective``.

    python scripts/calibration_worker.py \
        --config config/common.toml config/run_ac/ac53.toml \
        --params "value1=0.3;value2=0.8;value3=1.1;value4=1.4;value5=1.7;value6=2.0"
"""

from __future__ import annotations

import argparse
import logging
import sys

import calibration_objective as objective


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", required=True,
                    help="config files (common.toml run_ac/acNN.toml)")
    ap.add_argument("--params", default="", help='e.g. "value1=0.3;value2=1.1"')
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap images per evaluation (speed during tuning)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stderr)

    params = objective.parse_params(a.params)
    value = objective.evaluate(a.config, params, max_images=a.max_images)
    # LAST line of stdout = the objective (parsed by the queue worker)
    print(f"{value:.8f}")


if __name__ == "__main__":
    main()
