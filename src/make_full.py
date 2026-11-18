"""Build a full-data split (train + local val merged) for final retraining.

The official val set ships without labels, so the only local validation is our
own split. Once the configuration is fixed on that split, the standard move is to
retrain on every labelled image (1790 for qichen, 6343 for qinsi) to squeeze out
the extra ~11% of data. `val` points at the same folder purely so Ultralytics can
run its per-epoch validation loop; the numbers it prints are train metrics and are
only a "is it learning" sanity check.

Usage:
    python3 tools/make_full.py --dataset qichen
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path("/hy-tmp/cd")


def link_all(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.iterdir():
        if not p.is_file():
            continue
        target = dst / p.name
        if not target.exists():
            os.link(p, target)
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["qichen", "qinsi"])
    p.add_argument("--out", default="split_all")
    args = p.parse_args()

    out = ROOT / args.out / args.dataset
    if out.exists():
        shutil.rmtree(out)

    total = 0
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src = ROOT / "split" / args.dataset / kind / split
            n = link_all(src, out / kind / "train")
            if kind == "images":
                total += n
    # val mirrors train: it only exists so the validation loop can run
    for kind in ("images", "labels"):
        link_all(out / kind / "train", out / kind / "val")

    shutil.copy2(ROOT / "split" / args.dataset / "class.txt", out / "class.txt")
    print(f"[full] {args.dataset}: train={total} val={total} -> {out}")


if __name__ == "__main__":
    main()
