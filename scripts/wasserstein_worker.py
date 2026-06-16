"""Per-task Wasserstein worker (STUB - science not yet wired).

Invoked once per (pair, time-index, ROI) task by
``distributed_wasserstein_queue.py``. It must:

  1. Resolve the two images for run_a / run_b at the shared time-grid index
     ``time_index`` (same time-since-injection, within the phase tolerance),
     using each run's imaging + injection protocol.
  2. Load the corresponding *mass* fields (analysis output), apply ``resize``
     and the ROI mask, and compute the W1 distance via
     ``darsia.wasserstein_distance(..., method="bregman")``.
  3. Print the distance as the LAST line of stdout (the queue parses it).

Until step 1-2 are wired to the new DarSIA preset, this stub prints a
deterministic placeholder and warns on stderr, so the full queue pipeline is
runnable for testing.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--time-index", type=int, required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--resize", type=float, default=0.10)
    a = ap.parse_args()

    # TODO: wire to DarSIA — load mass fields for (run_a@t, run_b@t), mask ROI,
    #       resize, then darsia.wasserstein_distance(m1, m2, method="bregman").
    sys.stderr.write(
        f"[STUB] wasserstein {a.run_a}-{a.run_b} t={a.time_index} roi={a.roi}: "
        "returning placeholder distance (science not yet wired)\n")

    placeholder = abs(hash((a.run_a, a.run_b, a.time_index, a.roi))) % 1000 / 1000.0
    print(f"{placeholder:.6f}")


if __name__ == "__main__":
    main()
