"""Sweep a global box rescaling factor to detect systematic box bias.

If the model predicts boxes that are consistently a bit too tight or too
loose, multiplying every box around its own centre by a constant can lift
mAP50-95 without any retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wbf_lib  # noqa: E402


def build_gt(root: Path, dataset: str, split: str, names: list[str], sizes: dict[str, tuple[int, int]]) -> dict:
    lbl_dir = root / "split" / dataset / "labels" / split
    gt = {"info": {}, "licenses": [], "images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    ann_id = 1
    for img_id, name in enumerate(sorted(sizes), start=1):
        width, height = sizes[name]
        gt["images"].append({"id": img_id, "file_name": name, "width": int(width), "height": int(height)})
        for c, x, y, w, h in wbf_lib.load_gt(lbl_dir, Path(name).stem, width, height):
            gt["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": c + 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h), "iscrowd": 0,
            })
            ann_id += 1
    return gt


def evaluate(coco_gt, per_image: dict, names: list[str], sx: float, sy: float) -> tuple[float, float]:
    from pycocotools.cocoeval import COCOeval

    dets = []
    for img_id, (name, (boxes, scores, labels, _)) in enumerate(sorted(per_image.items()), start=1):
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = (x2 - x1) * sx, (y2 - y1) * sy
            dets.append({"image_id": img_id, "category_id": int(c) + 1,
                         "bbox": [float(cx - w / 2), float(cy - h / 2), float(w), float(h)],
                         "score": float(s)})
    if not dets:
        return 0.0, 0.0
    ev = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
    ev.params.maxDets = [1, 10, 300]
    ev.evaluate()
    ev.accumulate()
    prec = ev.eval["precision"]
    per_class, per_class_50 = {}, {}
    for k, cname in enumerate(names):
        curve = np.where(prec[:, :, k, 0, -1] > -1, prec[:, :, k, 0, -1], np.nan)
        per_class[cname] = float(np.nanmean(curve))
        per_class_50[cname] = float(np.nanmean(curve[0]))
    return float(np.mean(list(per_class.values()))), float(np.mean(list(per_class_50.values())))


def main() -> None:
    from pycocotools.coco import COCO

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/hy-tmp/cd"))
    p.add_argument("--dataset", default="qinsi")
    p.add_argument("--split", default="val")
    p.add_argument("--members", required=True)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--scales", default="0.94,0.96,0.98,1.00,1.02,1.04,1.06")
    p.add_argument("--xy-grid", type=int, default=0, help="1 = also sweep separate x/y scales")
    args = p.parse_args()

    root: Path = args.root
    data_root = root / "split" / args.dataset
    img_dir = data_root / "images" / args.split
    names = [ln.split(maxsplit=1)[1].strip()
             for ln in (data_root / "class.txt").read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

    specs = []
    for item in args.members.split(","):
        path, _, size = item.partition(":")
        specs.append((path, int(size or 640)))

    per_image: dict[str, tuple] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for path, size in specs:
        pred = wbf_lib.predict_member(path, size, img_dir, args.conf, bool(args.tta), args.batch)
        for name, (boxes, scores, labels, (w, h)) in pred.items():
            sizes[name] = (w, h)
        if not per_image:
            for name, (boxes, scores, labels, shape) in pred.items():
                per_image[name] = (boxes, scores, labels, shape)
        else:
            fused = wbf_lib.fuse([{n: per_image[n] for n in per_image}, pred],
                                 [1.0, 1.0], sorted(per_image))
            for name, (boxes, scores, labels) in fused.items():
                per_image[name] = (boxes, scores, labels, sizes[name])

    gt = build_gt(root, args.dataset, args.split, names, sizes)
    gt_path = root / "runs" / f"_gt_{args.dataset}_{args.split}.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    coco_gt = COCO(str(gt_path))

    scales = [float(x) for x in args.scales.split(",")]
    rows = []
    for s in scales:
        m, m50 = evaluate(coco_gt, per_image, names, s, s)
        rows.append({"sx": s, "sy": s, "map50_95": round(m, 4), "map50": round(m50, 4)})
        print(json.dumps(rows[-1]), flush=True)

    if args.xy_grid:
        best = max(rows, key=lambda r: r["map50_95"])
        for sx in scales:
            for sy in scales:
                m, m50 = evaluate(coco_gt, per_image, names, sx, sy)
                rows.append({"sx": sx, "sy": sy, "map50_95": round(m, 4), "map50": round(m50, 4)})
        rows.sort(key=lambda r: -r["map50_95"])
        print("--- top 5 (x/y grid) ---", flush=True)
        for r in rows[:5]:
            print(json.dumps(r), flush=True)

    print(json.dumps({"best": max(rows, key=lambda r: r["map50_95"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
