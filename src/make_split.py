"""Build a leak-free, class-stratified train/val split for the course-design datasets.

Run from the directory that contains ``datasets/``:

    python make_split.py --root . --out split

* Identical images (md5) are kept inside the same split, so duplicates cannot
  inflate validation scores.
* Splits are stratified on which of tick / cross appear in the image, because
  cross is extremely rare (242 instances over 1790 images).
* Files are hard-linked when possible, so the split costs almost no extra disk.

The resulting layout matches what ``src/object_detection/yolo_training.py``
expects: ``<out>/<dataset>/images/{train,val}`` and ``.../labels/{train,val}``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
from pathlib import Path


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_classes(label_path: Path) -> set[int]:
    if not label_path.is_file():
        return set()
    classes: set[int] = set()
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if fields:
            classes.add(int(float(fields[0])))
    return classes


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(root: Path, out: Path, dataset: str, val_fraction: float, seed: int) -> dict:
    image_dir = root / "datasets" / dataset / "images" / "train"
    label_dir = root / "datasets" / dataset / "labels" / "train"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    images = sorted(p for p in image_dir.iterdir() if p.is_file())
    by_hash: dict[str, list[Path]] = {}
    for path in images:
        by_hash.setdefault(md5_of(path), []).append(path)
    groups = list(by_hash.values())

    buckets: dict[str, list[list[Path]]] = collections.defaultdict(list)
    for group in groups:
        seen: set[int] = set()
        for path in group:
            seen |= label_classes(label_dir / f"{path.stem}.txt")
        key = ("t" if 1 in seen else "") + ("x" if 2 in seen else "") or "none"
        buckets[key].append(group)

    rng = random.Random(seed)
    val_group_ids: set[int] = set()
    for key in sorted(buckets):
        bucket = buckets[key]
        rng.shuffle(bucket)
        take = max(1, round(len(bucket) * val_fraction)) if len(bucket) > 1 else 0
        val_group_ids |= {id(group) for group in bucket[:take]}

    stats = {"dataset": dataset, "buckets": {k: len(v) for k, v in buckets.items()}, "splits": {}}
    for split_name, selected in (
        ("val", [g for g in groups if id(g) in val_group_ids]),
        ("train", [g for g in groups if id(g) not in val_group_ids]),
    ):
        class_count: collections.Counter[int] = collections.Counter()
        image_count = 0
        for group in selected:
            for path in group:
                link_or_copy(path, out / dataset / "images" / split_name / path.name)
                label_path = label_dir / f"{path.stem}.txt"
                if label_path.is_file():
                    link_or_copy(label_path, out / dataset / "labels" / split_name / label_path.name)
                image_count += 1
                class_count.update(label_classes(label_path))
        stats["splits"][split_name] = {
            "images": image_count,
            "class_instances": {str(k): v for k, v in sorted(class_count.items())},
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("split"))
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--datasets", nargs="+", default=["qichen", "qinsi"])
    args = parser.parse_args()

    report = [build(args.root, args.out, name, args.val_fraction, args.seed) for name in args.datasets]
    for item in report:
        print(json.dumps(item, ensure_ascii=False))
    (args.out / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
