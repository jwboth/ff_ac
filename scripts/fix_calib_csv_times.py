"""Re-key the `metrics` column of auto-calibration CSVs from image INDEX to real
time-since-start (hours), in place. For CSVs produced before the time-keying fix
(metrics keyed "0","1",...), so the calibration viewer plots a real time axis.

No DarSIA / no re-run needed: it recomputes each run's calibration image times by
replicating the calibration1/2 selection from the run's protocols.

    python scripts/fix_calib_csv_times.py --logs-dir logs --protocols-root protocols
"""
from __future__ import annotations
import argparse, ast, csv, glob, re, shutil
from datetime import datetime, timedelta
from pathlib import Path

# calibration1/2 intervals from common.toml [data.interval.calibration1/2]
CAL_INTERVALS = [  # (start_h, end_h, num, tol_seconds)
    (10/60, 2.5, 5, 60),      # calibration1: 10 min .. 2.5 h, tol 1 min
    (3.0, 48.0, 5, 300),      # calibration2: 3 h .. 48 h,  tol 5 min
]


def _dt(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def calib_times_for_run(run: str, protocols_root: Path):
    """Ordered list of distinct calibration-image times (hours since t=0)."""
    base = protocols_root / run
    inj = base / "injection_protocol.csv"
    rows = list(csv.DictReader(inj.open(encoding="utf-8"))) if inj.exists() else []
    if not rows:
        return None
    t0 = _dt(rows[0]["start"])
    if t0 is None:
        return None
    # all imaging datetimes
    imgs = []
    for f in sorted(base.glob("imaging_protocol_*.csv")):
        for r in csv.DictReader(f.open(encoding="utf-8")):
            d = _dt(r.get("datetime", ""))
            if d is not None:
                imgs.append(d)
    if not imgs:
        return None
    imgs.sort()
    matched = []  # (datetime) of selected images, in request order
    seen = set()
    for (s_h, e_h, num, tol_s) in CAL_INTERVALS:
        for k in range(num):
            t_req = s_h + (e_h - s_h) * (k / (num - 1) if num > 1 else 0)
            target = t0 + timedelta(hours=t_req)
            best = min(imgs, key=lambda d: abs((d - target).total_seconds()))
            if abs((best - target).total_seconds()) <= tol_s:
                if best not in seen:
                    seen.add(best); matched.append(best)
    matched.sort()
    return [round((d - t0).total_seconds() / 3600.0, 3) for d in matched]


def _run_from_name(name: str):
    m = re.match(r"(?:tmp_)?auto_calibration_(.+)\.csv$", name)
    return m.group(1) if m else None


def fix_csv(path: Path, times, backup: bool):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return 0
    changed = 0
    for row in rows:
        raw = row.get("metrics", "")
        if not raw:
            continue
        try:
            metrics = ast.literal_eval(raw)
        except Exception:
            continue
        if not isinstance(metrics, dict) or not metrics:
            continue
        # only re-key if keys look like plain integer indices
        keys = list(metrics.keys())
        if not all(re.fullmatch(r"\d+", str(k)) for k in keys):
            continue
        ordered = sorted(metrics.items(), key=lambda kv: int(kv[0]))
        new = {}
        for i, (_, v) in enumerate(ordered):
            label = f"{times[i]:.3f}h" if i < len(times) else f"{i}"
            new[label] = v
        row["metrics"] = repr(new)
        changed += 1
    if changed:
        if backup:
            shutil.copy2(path, path.with_suffix(".csv.bak"))
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--protocols-root", type=Path, default=Path("protocols"))
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(str(args.logs_dir / "auto_calibration_*.csv")) +
                   glob.glob(str(args.logs_dir / "tmp_auto_calibration_*.csv")))
    files = [Path(f) for f in files if "sanity" not in Path(f).stem]
    for f in files:
        run = _run_from_name(f.name)
        if not run:
            continue
        times = calib_times_for_run(run, args.protocols_root)
        if not times:
            print(f"{f.name}: could not resolve calib times for {run}; skipped"); continue
        n = fix_csv(f, times, backup=not args.no_backup)
        print(f"{f.name}: re-keyed {n} row(s) -> times {times}")


if __name__ == "__main__":
    main()
