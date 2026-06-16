"""Group AC experiments into calibration groups by colour / lighting.

Calibrating all 43 experiments is expensive. Instead we cluster experiments that
*look* alike (same colour cast and lighting in their first few, pre-injection
frames), pick one representative per group, calibrate only the representatives,
and let every group member reuse its representative's calibration.

Method
------
1. For each experiment, read the first ``--num-images`` JPGs of its injection
   folder (these are pre-injection / baseline frames) and extract a colour/light
   feature vector: per-channel mean and std + overall brightness, averaged over
   the frames, on a downscaled image.
2. Standardise features across experiments (z-score).
3. k-means (numpy, multi-restart) into ``--groups`` clusters.
4. Representative = the experiment closest to its cluster centroid.

Outputs ``groups.csv`` (experiment, group, representative flag, features) and
``groups.json`` ({group: {representative, members}}).

Usage
-----
    python scripts/group_calibration.py --albus-root /path/to/Albus \
        --out config/calibration_groups --groups 10 --num-images 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import numpy as np

LOG = logging.getLogger("group_calibration")
FEATURES = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "brightness"]


def _image_features(path: Path, size: int = 96) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow required: pip install pillow")
    try:
        im = Image.open(path)
        im.draft("RGB", (size, size))          # fast partial decode
        im = im.convert("RGB")
        im.thumbnail((size, size))
        a = np.asarray(im, dtype=np.float64) / 255.0
    except Exception as exc:  # noqa: BLE001
        LOG.warning("cannot read %s: %s", path.name, exc)
        return None
    mean = a.reshape(-1, 3).mean(axis=0)
    std = a.reshape(-1, 3).std(axis=0)
    brightness = a.mean()
    return np.array([mean[0], mean[1], mean[2], std[0], std[1], std[2], brightness])


def _injection_folder(exp_dir: Path) -> Path | None:
    for sub in exp_dir.iterdir():
        if sub.is_dir() and "injection" in sub.name:
            return sub
    return None


def experiment_feature(exp_dir: Path, num_images: int) -> np.ndarray | None:
    folder = _injection_folder(exp_dir)
    if folder is None:
        LOG.warning("%s: no injection folder", exp_dir.name)
        return None
    jpgs = sorted(folder.glob("*.JPG"))[:num_images]
    feats = [f for f in (_image_features(p) for p in jpgs) if f is not None]
    if not feats:
        return None
    return np.mean(feats, axis=0)


# --- lightweight k-means (numpy) -----------------------------------------
def kmeans(X: np.ndarray, k: int, restarts: int = 25, iters: int = 100,
           seed: int = 0):
    rng = np.random.default_rng(seed)
    best_labels, best_centroids, best_inertia = None, None, np.inf
    n = len(X)
    k = min(k, n)
    for _ in range(restarts):
        centroids = X[rng.choice(n, k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = d.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                pts = X[labels == j]
                if len(pts):
                    centroids[j] = pts.mean(axis=0)
                else:  # re-seed an empty cluster
                    centroids[j] = X[rng.integers(n)]
        inertia = ((X - centroids[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_labels, best_centroids, best_inertia = labels.copy(), centroids.copy(), inertia
    return best_labels, best_centroids, best_inertia


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--albus-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output dir")
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--num-images", type=int, default=5)
    ap.add_argument("--experiments", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = args.albus_root
    if args.experiments:
        exps = [root / e for e in args.experiments]
    else:
        exps = sorted([d for d in root.iterdir()
                       if d.is_dir() and re.fullmatch(r"AC\d+", d.name)],
                      key=lambda d: int(d.name[2:]))

    names, feats = [], []
    for exp in exps:
        f = experiment_feature(exp, args.num_images)
        if f is None:
            continue
        names.append(exp.name)
        feats.append(f)
        LOG.info("%s features: %s", exp.name,
                 ", ".join(f"{k}={v:.3f}" for k, v in zip(FEATURES, f)))

    X = np.array(feats)
    # standardise
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xz = (X - mu) / sigma

    labels, centroids, inertia = kmeans(Xz, args.groups, seed=args.seed)
    LOG.info("k-means: %d groups, inertia=%.3f", len(set(labels)), inertia)

    # representative = nearest experiment to its centroid
    reps = {}
    for g in sorted(set(labels)):
        idx = np.where(labels == g)[0]
        d = ((Xz[idx] - centroids[g]) ** 2).sum(axis=1)
        reps[g] = names[idx[d.argmin()]]

    args.out.mkdir(parents=True, exist_ok=True)
    # CSV
    with (args.out / "groups.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "group", "is_representative", *FEATURES])
        for name, g, f in zip(names, labels, X):
            w.writerow([name, int(g), int(name == reps[g]),
                        *[f"{v:.4f}" for v in f]])
    # JSON
    groups = {}
    for g in sorted(set(labels)):
        members = [names[i] for i in range(len(names)) if labels[i] == g]
        groups[str(int(g))] = {"representative": reps[g], "members": members}
    (args.out / "groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")

    LOG.info("Wrote %s/groups.csv and groups.json", args.out)
    print("\n=== Calibration groups ===")
    for g, info in groups.items():
        print(f"Group {g}: rep={info['representative']:6s}  "
              f"({len(info['members'])}) {', '.join(info['members'])}")
    print(f"\nCalibrate only {len(groups)} representatives instead of {len(names)} experiments.")


if __name__ == "__main__":
    main()
