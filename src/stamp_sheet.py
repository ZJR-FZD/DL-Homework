"""Build a contact sheet of every mined tick/cross crop so bad crops can be spotted.

    python tools/stamp_sheet.py --dataset qichen --classes 2 --out runs/stamps.jpg

Prints an index for each cell; feed the good indices back into enhance.py via
--whitelist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

import enhance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/hy-tmp/cd"))
    parser.add_argument("--dataset", default="qichen")
    parser.add_argument("--classes", nargs="+", type=int, default=[2])
    parser.add_argument("--min-px", type=int, default=24)
    parser.add_argument("--cell", type=int, default=110)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("/hy-tmp/cd/runs/stamps.jpg"))
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--clean", type=int, default=1)
    args = parser.parse_args()

    src = args.root / "split" / args.dataset
    marks = enhance.mine_marks(
        src / "images" / "train", src / "labels" / "train", set(args.classes), args.min_px,
        clean_only=bool(args.clean),
    )
    print(f"{len(marks)} stamps")
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)

    cell = args.cell
    cols = args.cols
    rows = (len(marks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, stamp in enumerate(marks):
        if args.export:
            stamp.save(args.export / f"{index:04d}.png")
        thumb = stamp.copy()
        thumb.thumbnail((cell - 18, cell - 18), Image.LANCZOS)
        x = (index % cols) * cell + (cell - thumb.width) // 2
        y = (index // cols) * cell + 14
        # composite the multiplicative stamp over white to preview it
        sheet.paste(thumb, (x, y))
        draw.text(((index % cols) * cell + 3, (index // cols) * cell + 2), str(index), fill=(200, 0, 0))
        draw.rectangle(
            [(index % cols) * cell, (index // cols) * cell,
             (index % cols) * cell + cell - 1, (index // cols) * cell + cell - 1],
            outline=(220, 220, 220),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=90)
    print(f"sheet -> {args.out} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
