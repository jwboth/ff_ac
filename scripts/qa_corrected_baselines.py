"""QA contact sheet of per-experiment corrected baselines.

Camera movement BETWEEN experiments shows up as a mis-placed crop: each
experiment uses the SAME curvature `pts_src`, so if the camera sat at a
different position, the rig will look shifted/cut in that experiment's
corrected baseline. This tiles every experiment's
``<results>/acNN/setup/rig/log/corrected_baseline.png`` into one labelled
grid so mis-crops jump out at a glance. Run AFTER setup.

    python scripts/qa_corrected_baselines.py --results-root "Z:\\Albus\\Results" --out qa_baselines.png
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("qa_corrected_baselines.png"))
    ap.add_argument("--thumb-width", type=int, default=320)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--rel", default="setup/rig/log/corrected_baseline.png",
                    help="path to the corrected baseline within each acNN folder")
    args = ap.parse_args()

    dirs = sorted([d for d in args.results_root.iterdir()
                   if d.is_dir() and re.fullmatch(r"ac\d+", d.name.lower())],
                  key=lambda d: int(re.sub(r"\D", "", d.name) or 0))
    thumbs = []
    missing = []
    for d in dirs:
        p = d / args.rel
        if not p.is_file():
            missing.append(d.name); continue
        try:
            im = Image.open(p).convert("RGB")
            w = args.thumb_width; h = round(im.height * w / im.width)
            thumbs.append((d.name, im.resize((w, h))))
        except Exception as e:  # noqa: BLE001
            missing.append(f"{d.name}({e})")

    if not thumbs:
        print("No corrected_baseline.png found under", args.results_root); 
        if missing: print("missing:", ", ".join(missing))
        return

    tw = args.thumb_width
    th = max(t.height for _, t in thumbs)
    label_h = 26
    cols = args.cols
    rows = (len(thumbs) + cols - 1) // cols
    cell_w, cell_h = tw + 8, th + label_h + 8
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, (name, im) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = c * cell_w + 4, r * cell_h + 4
        draw.rectangle([x-2, y-2, x+tw+2, y+label_h+th+2], outline=(180, 180, 180))
        draw.text((x + 4, y + 4), name.upper(), fill=(0, 0, 0), font=font)
        sheet.paste(im, (x, y + label_h))
    sheet.save(args.out)
    print(f"Wrote {args.out}  ({len(thumbs)} experiments, {rows}x{cols} grid)")
    if missing:
        print("No corrected baseline for:", ", ".join(missing))


if __name__ == "__main__":
    main()
