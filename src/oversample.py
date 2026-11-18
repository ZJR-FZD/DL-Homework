"""Oversample images that contain the rare tick / cross classes.

The safest fix for class imbalance: no synthetic pixels, just repeated sampling
of the pages that actually contain the rare marks. Hard links keep the disk cost
at zero.

    python tools/oversample.py --dataset qichen --repeat 2=4,1=2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def parse_repeat(text: str) -> dict[int, int]:
    rules: dict[int, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        cls, times = item.split("=")
        rules[int(cls)] = int(times)
    return rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/hy-tmp/cd"))
    parser.add_argument("--dataset", default="qichen")
    parser.add_argument("--source", default="split")
    parser.add_argument("--out", default="split_os")
    parser.add_argument("--repeat", default="2=4,1=2")
    args = parser.parse_args()

    rules = parse_repeat(args.repeat)
    src = args.root / args.source / args.dataset
    dst = args.root / args.out / args.dataset
    img_train, lbl_train = src / "images" / "train", src / "labels" / "train"

    for p in (src / "images" / "val").iterdir():
        link_or_copy(p, dst / "images" / "val" / p.name)
    for p in (src / "labels" / "val").iterdir():
        link_or_copy(p, dst / "labels" / "val" / p.name)
    for extra in ("class.txt",):
        if (src / extra).is_file():
            link_or_copy(src / extra, dst / extra)

    counts: dict[str, int] = {}
    total = 0
    for label_path in sorted(lbl_train.glob("*.txt")):
        image = None
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = img_train / f"{label_path.stem}{suffix}"
            if candidate.is_file():
                image = candidate
                break
        if image is None:
            continue
        classes = {int(float(line.split()[0])) for line in label_path.read_text().splitlines() if line.strip()}
        times = max([rules.get(c, 1) for c in classes] + [1])
        for copy_index in range(times):
            suffix = "" if copy_index == 0 else f"_os{copy_index}"
            link_or_copy(image, dst / "images" / "train" / f"{image.stem}{suffix}{image.suffix}")
            link_or_copy(label_path, dst / "labels" / "train" / f"{label_path.stem}{suffix}.txt")
            total += 1
        counts[str(sorted(classes))] = counts.get(str(sorted(classes)), 0) + 1

    report = {"dataset": args.dataset, "rules": rules, "train_images": total, "source_groups": counts}
    (dst / "oversample_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
