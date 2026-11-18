"""Evaluate single models or a WBF ensemble on the local validation split.

Uses pycocotools so the numbers are directly comparable across configurations and
against the ensemble (Ultralytics' own mAP is also 101-point, so single-model
results land within a rounding error).

Usage:
    # single model reference
    python3 tools/eval_ensemble.py --members "runs/q_hsv/weights/best.pt:1024"
    # ensemble of two
    python3 tools/eval_ensemble.py --members "runs/q_hsv/weights/best.pt:1024,runs/q_mos05/weights/best.pt:1024"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wbf_lib

ROOT = Path("/hy-tmp/cd")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--members", required=True, help="comma list of path:imgsz")
    p.add_argument("--weights-list", default="", help="comma list of fusion weights, default equal")
    p.add_argument("--dataset", default="qichen")
    p.add_argument("--split", default="val")
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--iou-thr", type=float, default=0.55)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    root = ROOT / "split" / args.dataset
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    names = [ln.split(maxsplit=1)[1].strip() for ln in (root / "class.txt").read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

    specs = []
    for item in args.members.split(","):
        path, _, size = item.partition(":")
        specs.append((path, int(size or 1024)))
    weights = [float(x) for x in args.weights_list.split(",")] if args.weights_list else [1.0] * len(specs)

    print(f"[eval] {len(specs)} member(s), tta={bool(args.tta)}, conf={args.conf}", flush=True)
    members = []
    for path, size in specs:
        print(f"[eval] predicting {path} @ {size}", flush=True)
        members.append(wbf_lib.predict_member(path, size, img_dir, args.conf, bool(args.tta), args.batch))

    image_names = sorted(p.name for p in img_dir.iterdir() if p.is_file())
    fused = wbf_lib.fuse(members, weights, image_names, iou_thr=args.iou_thr, skip_box_thr=args.conf / 2)

    gt = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
    }
    dets = []
    ann_id = 1
    for img_id, name in enumerate(image_names, start=1):
        _, _, _, (width, height) = members[0][name]
        gt["images"].append({"id": img_id, "file_name": name, "width": int(width), "height": int(height)})
        for c, x, y, w, h in wbf_lib.load_gt(lbl_dir, Path(name).stem, width, height):
            gt["annotations"].append(
                {"id": ann_id, "image_id": img_id, "category_id": c + 1,
                 "bbox": [float(x), float(y), float(w), float(h)], "area": float(w * h), "iscrowd": 0}
            )
            ann_id += 1
        boxes, scores, labels = fused[name]
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels):
            dets.append(
                {"image_id": img_id, "category_id": int(c) + 1,
                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)}
            )

    gt_path = ROOT / "runs" / f"_gt_{args.dataset}_{args.split}.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    coco_gt = COCO(str(gt_path))
    if not dets:
        print("[eval] no detections")
        return
    coco_dt = coco_gt.loadRes(dets)

    for iou_type in ("bbox",):
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
        precisions = ev.eval["precision"]  # [T, R, K, A, M]
        # COCOeval.summarize() reports -1 for this setup, so aggregate by hand:
        # AP per class = mean over recall points of the precision envelope at maxDets=300.
        per_class = {}
        per_class_50 = {}
        for k, cname in enumerate(names):
            curve = precisions[:, :, k, 0, -1]  # [T, R]
            curve = np.where(curve > -1, curve, np.nan)
            per_class[cname] = round(float(np.nanmean(curve)), 4)
            per_class_50[cname] = round(float(np.nanmean(curve[0])), 4)
        map50_95 = float(np.mean(list(per_class.values())))
        map50 = float(np.mean(list(per_class_50.values())))
        print(json.dumps({
            "members": args.members,
            "map50_95": round(map50_95, 4),
            "map50": round(map50, 4),
            "per_class_map50_95": per_class,
            "per_class_map50": per_class_50,
            "n_det": len(dets),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
