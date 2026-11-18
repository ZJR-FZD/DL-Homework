"""Per-class IoU diagnostics: is the bottleneck localisation or detection?

For each class, greedily matches predictions to ground truth at IoU >= 0.3 and
reports the matched IoU distribution plus missed-GT / false-positive counts.
If matched IoUs cluster just above 0.5, the boxes are "roughly right but loose"
and mAP50-95 is capped by box regression; if GT go unmatched, it is a recall
problem.

Usage:
    python3 tools/diagnose_iou.py --weights runs/q_s1024_s/weights/best.pt \
        --dataset qichen --imgsz 1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/hy-tmp/cd")


def load_gt(label_path: Path, width: int, height: int) -> np.ndarray:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) != 5:
            continue
        c, x, y, w, h = int(float(f[0])), *map(float, f[1:])
        rows.append(
            [
                c,
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", default="qichen")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--split", default="val")
    args = p.parse_args()

    from ultralytics import YOLO

    root = ROOT / "split" / args.dataset
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    names = [ln.split(maxsplit=1)[1].strip() for ln in (root / "class.txt").read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

    model = YOLO(args.weights, task="detect")
    results = model.predict(
        source=str(img_dir), imgsz=args.imgsz, conf=args.conf, iou=0.7, batch=4,
        device="0", stream=True, verbose=False,
    )

    stats: dict[int, dict[str, list[float] | int]] = {
        c: {"ious": [], "gt": 0, "matched": 0, "pred": 0, "fp": 0} for c in range(len(names))
    }
    for res in results:
        gt = load_gt(lbl_dir / (Path(res.path).stem + ".txt"), *res.orig_shape[::-1])
        w, h = res.orig_shape[1], res.orig_shape[0]
        boxes = res.boxes
        pred = (
            np.column_stack([boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy()[:, None]])
            if boxes is not None and len(boxes)
            else np.zeros((0, 5), dtype=np.float32)
        )
        for c in range(len(names)):
            g = gt[gt[:, 0] == c][:, 1:]
            pr = pred[pred[:, 4] == c][:, :4]
            s = stats[c]
            s["gt"] += len(g)
            s["pred"] += len(pr)
            m = iou_matrix(pr, g)
            used_g: set[int] = set()
            used_p: set[int] = set()
            while m.size:
                i, j = np.unravel_index(np.argmax(m), m.shape)
                if m[i, j] < 0.3:
                    break
                s["ious"].append(float(m[i, j]))  # type: ignore[union-attr]
                used_g.add(int(j))
                used_p.add(int(i))
                m[i, :] = -1
                m[:, j] = -1
            s["matched"] += len(used_g)  # type: ignore[operator]
            s["fp"] += len(pr) - len(used_p)  # type: ignore[operator]

    out = {}
    for c, s in stats.items():
        ious = np.asarray(sorted(s["ious"]), dtype=np.float32)  # type: ignore[arg-type]
        if ious.size == 0:
            out[names[c]] = {"gt": s["gt"], "matched": 0}
            continue
        out[names[c]] = {
            "gt": s["gt"],
            "pred": s["pred"],
            "matched_at_iou0.3": int(ious.size),
            "recall_at_iou0.3": round(float(ious.size / max(1, s["gt"])), 3),  # type: ignore[operator]
            "fp": s["fp"],
            "iou_mean": round(float(ious.mean()), 3),
            "iou_p10": round(float(np.percentile(ious, 10)), 3),
            "iou_p50": round(float(np.percentile(ious, 50)), 3),
            "iou_p90": round(float(np.percentile(ious, 90)), 3),
            "frac_above_0.5": round(float((ious >= 0.5).mean()), 3),
            "frac_above_0.75": round(float((ious >= 0.75).mean()), 3),
            "frac_above_0.9": round(float((ious >= 0.9).mean()), 3),
            "gt_area_frac_median": None,
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
