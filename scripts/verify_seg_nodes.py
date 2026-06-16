"""Verifiser at den LASTEDE signal-modellen (seed-et auto-kalibreringen faktisk ser)
har forventet antall noder. Bygger konteksten nøyaktig som auto_calibrate gjør, og
teller value-nodene per facies.

Bruk:
    python scripts/verify_seg_nodes.py <run> [--config-dir config_seg6/run_ac] [--expect 7]

Exit-kode 0 = OK (riktig antall noder), 1 = feil antall (seed ikke regenerert), 2 = lastefeil.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--config-dir", default="config_seg6/run_ac")
    ap.add_argument("--expect", type=int, default=7, help="forventet antall noder (seg6 -> 7)")
    args = ap.parse_args()

    try:
        from auto_calibrate_color_to_mass import build_context
        from darsia.presets.workflows.rig import Rig
        ctx = build_context(run=args.run, config_dir=args.config_dir, rig_cls=Rig, use_facies=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{args.run}] LASTEFEIL: {exc!r}")
        return 2

    hetero = ctx.calibration.signal_model.model[1]
    labels = list(hetero.keys()) if hasattr(hetero, "keys") else []
    if not labels:
        print(f"[{args.run}] fant ingen facies i signal-modellen")
        return 2
    counts = {int(l): len(list(hetero[l].values)) for l in labels}
    n = counts[labels[0]]
    same = len(set(counts.values())) == 1
    seg = n - 1
    flag = "OK" if n == args.expect else "FEIL"
    print(f"[{args.run}] {flag}: {n} noder (value0..value{seg}) = seg{seg}, "
          f"{len(labels)} facies {sorted(counts.keys())}"
          + ("" if same else f"  (ULIKT antall per facies: {counts})"))
    return 0 if (n == args.expect and same) else 1


if __name__ == "__main__":
    raise SystemExit(main())
