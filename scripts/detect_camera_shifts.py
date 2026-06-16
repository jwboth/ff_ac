"""Detect camera shifts / bumps across the raw image series per experiment.

For each experiment, walks the images in TIME order (from the imaging-protocol
CSVs referenced by the run config), estimates the translation between each
frame and the previous one via phase cross-correlation on a downscaled
grayscale image, and flags frames where the inter-frame jump exceeds a
threshold (a "bump"). Also tracks the running (cumulative) displacement so you
can see whether the camera returned after a bump or stayed shifted.

This reads the RAW images from wherever the config points ([protocols.imaging]
folder keys -> UNC/server), so run it on a machine that can reach them.

    python scripts/detect_camera_shifts.py --config-dir config/run_ac --all `
        --common config/common.toml --threshold-px 12 --out qa_shifts

Outputs (under --out):
  * shifts_<exp>.csv     per-frame: datetime, file, dx, dy, jump_px, cum_x, cum_y, bump
  * bumps.csv            one row per detected bump across all experiments
  * shifts_<exp>.png     (unless --no-plots) displacement-over-time with bumps marked
"""
from __future__ import annotations

import argparse, csv, math, re, sys
try:
    import tomllib  # Python >= 3.11 (your venv is 3.13)
except ModuleNotFoundError:  # older Python
    import tomli as tomllib
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.registration import phase_cross_correlation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_gray(path: Path, max_dim: int) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    w, h = im.size
    s = max_dim / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))))
    return np.asarray(im, dtype=np.float32), (1.0 / s if s < 1.0 else 1.0)


def _imaging_pairs(cfg: dict, repo_root: Path) -> list[tuple[Path, Path]]:
    """Return [(folder, csv_path), ...] in [data].folders order."""
    prot = cfg.get("protocols", {})
    imaging = prot.get("imaging", {})
    folders = cfg.get("data", {}).get("folders", [])
    pairs = []
    if isinstance(imaging, dict):
        for folder in folders:
            csv_rel = imaging.get(folder)
            if csv_rel:
                pairs.append((Path(folder), repo_root / csv_rel))
    elif isinstance(imaging, str) and folders:
        pairs.append((Path(folders[0]), repo_root / imaging))
    return pairs


def _ordered_images(cfg: dict, repo_root: Path) -> list[tuple[str, str, Path]]:
    """[(datetime, filename, full_path)] in time order across all phase folders."""
    rows = []
    for folder, csv_path in _imaging_pairs(cfg, repo_root):
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                name = r.get("path") or r.get("Path")
                dt = r.get("datetime") or r.get("Datetime") or ""
                if name:
                    rows.append((dt, name, folder / name))
    rows.sort(key=lambda x: (x[0] == "", x[0]))  # by datetime, blanks last
    return rows


def process_experiment(name: str, cfg: dict, repo_root: Path, args, out_dir: Path) -> list[dict]:
    imgs = _ordered_images(cfg, repo_root)
    imgs = imgs[:: max(1, args.step)]
    if len(imgs) < 2:
        print(f"  {name}: <2 images resolved, skipping"); return []
    prev = None; cum_x = cum_y = 0.0
    per_frame = []; bumps = []
    for i, (dt, fname, fpath) in enumerate(imgs):
        loaded = _load_gray(fpath, args.max_dim)
        if loaded is None:
            continue
        gray, scale = loaded
        if prev is None or prev.shape != gray.shape:
            prev = gray; per_frame.append((dt, fname, 0.0, 0.0, 0.0, 0.0, 0.0, 0)); continue
        (dy, dx), _, _ = phase_cross_correlation(prev, gray, upsample_factor=10)
        dx *= scale; dy *= scale
        jump = math.hypot(dx, dy)
        cum_x += dx; cum_y += dy
        is_bump = 1 if jump >= args.threshold_px else 0
        per_frame.append((dt, fname, dx, dy, jump, cum_x, cum_y, is_bump))
        if is_bump:
            bumps.append({"experiment": name, "datetime": dt, "file": fname,
                          "jump_px": round(jump, 1), "cum_x": round(cum_x, 1), "cum_y": round(cum_y, 1)})
        prev = gray
    # write per-frame csv
    fp = out_dir / f"shifts_{name}.csv"
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["datetime", "file", "dx_px", "dy_px", "jump_px", "cum_x_px", "cum_y_px", "bump"])
        for row in per_frame:
            w.writerow([row[0], row[1], f"{row[2]:.1f}", f"{row[3]:.1f}", f"{row[4]:.1f}", f"{row[5]:.1f}", f"{row[6]:.1f}", row[7]])
    if not args.no_plots:
        mags = [math.hypot(r[5], r[6]) for r in per_frame]
        plt.figure(figsize=(10, 3))
        plt.plot(mags, lw=0.8, label="cumulative |displacement| (px)")
        bidx = [k for k, r in enumerate(per_frame) if r[7]]
        if bidx:
            plt.scatter(bidx, [mags[k] for k in bidx], c="red", s=14, zorder=3, label=f"bumps (>{args.threshold_px}px)")
        plt.title(f"{name}: camera displacement over time ({len(imgs)} frames)")
        plt.xlabel("frame # (time order)"); plt.ylabel("px"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out_dir / f"shifts_{name}.png", dpi=90); plt.close()
    print(f"  {name}: {len(per_frame)} frames, {len(bumps)} bump(s)"
          f"{' -> ' + ', '.join(b['datetime'][:16]+' ('+str(b['jump_px'])+'px)' for b in bumps[:6]) if bumps else ''}")
    return bumps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, default=Path("config/run_ac"))
    ap.add_argument("--common", type=Path, default=Path("config/common.toml"))
    ap.add_argument("--experiments", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--max-dim", type=int, default=500, help="downscale longest side to this many px")
    ap.add_argument("--step", type=int, default=1, help="process every Nth image (fast first pass)")
    ap.add_argument("--threshold-px", type=float, default=12.0, help="flag inter-frame jump >= this (original px)")
    ap.add_argument("--out", type=Path, default=Path("qa_shifts"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.all:
        configs = sorted(args.config_dir.glob("ac*.toml"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    else:
        configs = [args.config_dir / f"{e.lower()}.toml" for e in args.experiments]

    all_bumps = []
    for cpath in configs:
        if not cpath.is_file():
            print("missing", cpath); continue
        cfg = tomllib.loads(cpath.read_text(encoding="utf-8"))
        all_bumps += process_experiment(cpath.stem, cfg, args.repo_root, args, args.out)

    bp = args.out / "bumps.csv"
    with bp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment", "datetime", "file", "jump_px", "cum_x", "cum_y"])
        w.writeheader(); w.writerows(all_bumps)
    print(f"\nTotal bumps: {len(all_bumps)} across {len(configs)} experiment(s). Details -> {bp}")


if __name__ == "__main__":
    main()
