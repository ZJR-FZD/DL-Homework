"""Copy-paste augmentation for the rare tick / cross classes (qichen only).

Motivation: the training split holds ~1600 handwriting boxes but only ~240 tick
and ~100 cross boxes, so the network barely sees the two classes that carry two
thirds of the mAP weight.

Rather than drawing synthetic symbols in a paint program (which teaches the
network a suspiciously clean white background), this script mines *real*
tick/cross crops from labelled training pages and re-composites them elsewhere.

Compositing is multiplicative, not alpha: a crop is divided by its own paper
whiteness and multiplied into the destination page. Paper areas become a factor
of ~1 and are therefore untouched, while ink darkens the page exactly as much as
it darkened its source page. This avoids the grey rectangle that any alpha matte
leaves behind when the source scan is darker than the destination.

The validation split is hard-linked unchanged, so the metric stays honest.

    python tools/enhance.py --dataset qichen --aug-frac 0.6 --copies 1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_BOX_RATIO = 0.14   # a tick/cross never covers more than 14% of the page
MAX_PASTE_RATIO = 0.10  # nor more than 10% of the destination page


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def read_boxes(label_path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 5:
                rows.append([float(v) for v in fields[:5]])
    return rows


def mine_marks(images_dir: Path, labels_dir: Path, classes: set[int], min_px: int, clean_only: bool = True) -> list[Image.Image]:
    """Extract real tick/cross crops, already normalised by their own paper level."""
    crops: list[Image.Image] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        boxes = [b for b in read_boxes(label_path) if int(b[0]) in classes]
        if not boxes:
            continue
        with Image.open(image_path) as handle:
            page = handle.convert("RGB")
            width, height = page.size
            for _, cx, cy, bw, bh in boxes:
                px, py = bw * width, bh * height
                if px < min_px or py < min_px:
                    continue
                if px > width * MAX_BOX_RATIO or py > height * MAX_BOX_RATIO:
                    continue  # oversized box: almost certainly mixed with printed content
                if not 0.35 <= px / max(py, 1.0) <= 2.8:
                    continue
                x1 = max(0, int((cx - bw / 2) * width) - 1)
                y1 = max(0, int((cy - bh / 2) * height) - 1)
                x2 = min(width, int((cx + bw / 2) * width) + 2)
                y2 = min(height, int((cy + bh / 2) * height) + 2)
                if x2 - x1 < min_px or y2 - y1 < min_px:
                    continue
                patch = page.crop((x1, y1, x2, y2)).convert("RGB")
                normalised = normalise_ink(patch)
                if normalised is None:
                    continue
                if clean_only and not is_clean_mark(normalised):
                    continue
                crops.append(normalised)
    return crops


def normalise_ink(patch: Image.Image) -> Image.Image | None:
    """Divide the crop by its own paper level so paper becomes exactly white.

    Returns an 8-bit RGB image that behaves like a multiplicative ink stamp.
    """
    arr = np.asarray(patch, dtype=np.float32) + 1.0
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    paper = float(np.percentile(lum, 92))
    ink = float(np.percentile(lum, 5))
    if paper < 90 or paper - ink < 30:      # blank or uniformly dark patch
        return None
    factor = arr / paper
    if float(factor.min()) > 0.72:          # no real contrast -> no stroke
        return None
    factor = np.clip(factor, 0.0, 1.0)      # nothing may brighten the page
    stamp = np.clip(factor * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(stamp, mode="RGB")


def count_ink_components(mask: np.ndarray, min_area: int) -> tuple[int, float, float]:
    """Flood-fill the ink mask; return (n_components, largest_share, bbox_fill).

    Pure Python BFS is fine here: crops are tiny and this runs a few hundred
    times. It exists because a clean cross is one or two strokes, whereas a crop
    that accidentally includes printed text explodes into a dozen components.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []
    for y0 in range(height):
        for x0 in range(width):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                areas.append(area)
    if not areas:
        return 0, 0.0, 0.0
    total = int(mask.sum())
    largest = max(areas)
    ys, xs = np.where(mask)
    bbox_fill = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / float(height * width)
    return len(areas), largest / max(total, 1), bbox_fill


def is_clean_mark(stamp: Image.Image) -> bool:
    """Reject crops that carry printed text, table rules or barcodes."""
    small = stamp.copy()
    small.thumbnail((128, 128), Image.BILINEAR)
    grey = np.asarray(small.convert("L"), dtype=np.uint8)
    ink = grey < 150
    ratio = float(ink.mean())
    if not 0.02 <= ratio <= 0.30:
        return False
    count, largest_share, bbox_fill = count_ink_components(ink, max(6, int(ink.size * 0.004)))
    if count == 0 or count > 4:
        return False
    if largest_share < 0.45 or bbox_fill < 0.35:
        return False
    return True


def jitter_ink(stamp: Image.Image, rng: random.Random) -> Image.Image:
    """Shift the multiplicative stamp towards black / red / blue pen ink."""
    arr = np.asarray(stamp, dtype=np.float32)
    tint = rng.choice(
        [
            np.array([1.0, 1.0, 1.0]),    # keep the original pen colour
            np.array([1.0, 0.80, 0.80]),  # red pen
            np.array([0.85, 0.90, 1.0]),  # blue pen
            np.array([0.95, 0.95, 0.95]), # faded black
        ]
    )
    arr = np.clip(arr * tint * rng.uniform(0.9, 1.05), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def paste_mark(page: Image.Image, stamp: Image.Image, rng: random.Random) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    """Multiply one stamp into the page and return its normalised YOLO box."""
    width, height = page.size
    scale = rng.uniform(0.65, 1.15)
    mark_w = max(12, int(stamp.width * scale))
    mark_h = max(12, int(stamp.height * scale))
    limit = min(width, height) * MAX_PASTE_RATIO
    if max(mark_w, mark_h) > limit:
        ratio = limit / max(mark_w, mark_h)
        mark_w, mark_h = max(12, int(mark_w * ratio)), max(12, int(mark_h * ratio))
    if mark_w >= width or mark_h >= height:
        return None

    stamp = stamp.resize((mark_w, mark_h), Image.BILINEAR)
    stamp = stamp.rotate(rng.uniform(-22, 22), expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    if rng.random() < 0.5:
        stamp = stamp.transpose(Image.FLIP_LEFT_RIGHT)
    mark_w, mark_h = stamp.size
    if mark_w >= width or mark_h >= height:
        return None
    x = rng.randint(0, width - mark_w)
    y = rng.randint(0, height - mark_h)

    canvas = np.asarray(page, dtype=np.float32)
    region = canvas[y:y + mark_h, x:x + mark_w]
    factor = np.asarray(stamp, dtype=np.float32) / 255.0
    canvas[y:y + mark_h, x:x + mark_w] = np.clip(region * factor, 0, 255)
    out = Image.fromarray(canvas.astype(np.uint8), mode="RGB")

    cx = (x + mark_w / 2) / width
    cy = (y + mark_h / 2) / height
    return out, (cx, cy, mark_w / width, mark_h / height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/hy-tmp/cd"))
    parser.add_argument("--dataset", default="qichen")
    parser.add_argument("--source", default="split")
    parser.add_argument("--out", default="split_aug")
    parser.add_argument("--classes", nargs="+", type=int, default=[2], help="class ids to synthesise (2=cross)")
    parser.add_argument("--aug-frac", type=float, default=0.6)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--min-px", type=int, default=24)
    parser.add_argument("--keep-original", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src = args.root / args.source / args.dataset
    dst = args.root / args.out / args.dataset
    img_train, lbl_train = src / "images" / "train", src / "labels" / "train"
    labels_out = dst / "labels" / "train"
    images_out = dst / "images" / "train"
    labels_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    # validation split stays byte-identical
    for p in (src / "images" / "val").iterdir():
        link_or_copy(p, dst / "images" / "val" / p.name)
    for p in (src / "labels" / "val").iterdir():
        link_or_copy(p, dst / "labels" / "val" / p.name)
    for extra in ("class.txt",):
        if (src / extra).is_file():
            link_or_copy(src / extra, dst / extra)

    classes = set(args.classes)
    marks = mine_marks(img_train, lbl_train, classes, args.min_px)
    if not marks:
        raise SystemExit("no usable tick/cross crops were mined")
    print(f"mined {len(marks)} real marks for classes {sorted(classes)}")

    added = 0
    augmented_images = 0
    for label_path in sorted(lbl_train.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image(img_train, stem)
        if image_path is None:
            continue
        link_or_copy(image_path, images_out / image_path.name)
        original_label = label_path.read_text(encoding="utf-8")
        link_or_copy(label_path, labels_out / label_path.name)

        if rng.random() > args.aug_frac:
            continue
        with Image.open(image_path) as handle:
            page = handle.convert("RGB")
        if max(page.size) > 2200:   # keep paste cost bounded on huge scans
            ratio = 2200 / max(page.size)
            page = page.resize((int(page.width * ratio), int(page.height * ratio)), Image.LANCZOS)
        new_lines = [original_label.strip()] if original_label.strip() else []
        pasted = 0
        for _ in range(args.copies):
            stamp = jitter_ink(rng.choice(marks), rng)
            result = paste_mark(page, stamp, rng)
            if result is None:
                continue
            page, (cx, cy, bw, bh) = result
            new_lines.append(f"{args.classes[0]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            pasted += 1
        if pasted == 0:
            continue
        out_name = f"aug_{stem}.jpg"
        page.save(images_out / out_name, quality=args.jpeg_quality)
        (labels_out / f"aug_{stem}.txt").write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        augmented_images += 1
        added += pasted

    report = {
        "dataset": args.dataset,
        "marks_mined": len(marks),
        "images_augmented": augmented_images,
        "marks_pasted": added,
        "originals": len(list(images_out.glob("*.jpg"))) - augmented_images,
        "out": str(dst),
    }
    (dst / "augment_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
