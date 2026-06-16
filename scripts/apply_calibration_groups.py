"""Share calibrations between AC experiments using the group structure.

Two modes:

* ``--seed-from <exp>`` (run BEFORE auto-calibration): copy one calibrated run's
  artifacts (e.g. ``ac60``) into every group REPRESENTATIVE, to serve as the
  structural template + Optuna starting point. The auto-calibration then loads
  this and optimises the signal values per representative.

* default (run AFTER auto-calibration): copy each optimised REPRESENTATIVE's
  calibration into its group MEMBERS, so all 43 experiments reuse a calibration
  without calibrating each one.

DarSIA loads a run's calibration from that run's results folder, so this simply
copies the calibration sub-folders between results folders.

Usage
-----
    # seed all 10 representatives from AC60 (template + start point)
    python scripts/apply_calibration_groups.py \
        --groups-file config/calibration_groups/groups.json \
        --results-root "E:\\ff_ml4gcs" --seed-from ac60

    # after Optuna: push each representative to its members
    python scripts/apply_calibration_groups.py \
        --groups-file config/calibration_groups/groups.json \
        --results-root "E:\\ff_ml4gcs"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

LOG = logging.getLogger("apply_calibration_groups")

# Common calibration artifact folder names written under a run's results dir.
DEFAULT_SUBDIRS = ["calibration", "color_paths", "color_to_mass",
                   "color_embedding", "flash"]


def _run_results_dir(results_root: Path, run: str) -> Path:
    return results_root / run.lower()


def copy_calibration(src_dir: Path, dst_dir: Path, subdirs: list,
                     dry_run: bool = False) -> list:
    """Copy existing calibration subdirs from src_dir into dst_dir. Returns names."""
    copied = []
    for name in subdirs:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        LOG.info("%s %s -> %s", "DRY" if dry_run else "copy", src, dst)
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        copied.append(name)
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups-file", required=True, type=Path)
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--calib-subdirs", nargs="+", default=DEFAULT_SUBDIRS)
    ap.add_argument("--seed-from", default=None,
                    help="experiment (e.g. ac60) whose calibration seeds every "
                         "representative as template + Optuna start point")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    groups = json.loads(a.groups_file.read_text())

    if a.seed_from:
        src_dir = _run_results_dir(a.results_root, a.seed_from)
        reps = [info["representative"] for info in groups.values()]
        seeded = 0
        for rep in reps:
            if rep.lower() == a.seed_from.lower():
                continue
            copied = copy_calibration(src_dir, _run_results_dir(a.results_root, rep),
                                      a.calib_subdirs, a.dry_run)
            if not copied:
                LOG.warning("No calibration artifacts in %s (calibrate %s first)",
                            src_dir, a.seed_from)
                break
            seeded += 1
        LOG.info("Seeded %d representative(s) from %s", seeded, a.seed_from)
        return

    total_members = 0
    for gid, info in groups.items():
        rep = info["representative"]
        rep_dir = _run_results_dir(a.results_root, rep)
        for member in info["members"]:
            if member.lower() == rep.lower():
                continue
            member_dir = _run_results_dir(a.results_root, member)
            copied = copy_calibration(rep_dir, member_dir, a.calib_subdirs, a.dry_run)
            if not copied:
                LOG.warning("Group %s: no calibration artifacts in %s "
                            "(calibrate the representative first)", gid, rep_dir)
            total_members += 1
    LOG.info("Applied representative calibration to %d member experiment(s)", total_members)


if __name__ == "__main__":
    main()
