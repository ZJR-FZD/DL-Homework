"""Weighted-boxes-fusion helpers shared by the eval and submission scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path("/hy-tmp/cd")


def predict_member(weights: str, imgsz: int, image_dir: Path, conf: float, tta: bool,
                   batch: int = 4, max_det: int = 300) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]]:
    """Run one model over a folder; returns {filename: (xyxy, conf, cls, (w, h))}."""
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    stream = model.predict(
        source=str(image_dir), imgsz=imgsz, conf=conf, iou=0.7, batch=batch,
        max_det=max_det, augment=tta, device="0", stream=True, verbose=False,
    )
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for r in stream:
        name = Path(r.path).name
        h, w = r.orig_shape
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            out[name] = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32), (w, h))
        else:
            out[name] = (
                boxes.xyxy.cpu().numpy().astype(np.float32),
                boxes.conf.cpu().numpy().astype(np.float32),
                boxes.cls.cpu().numpy().astype(np.int32),
                (w, h),
            )
    return out


def fuse(members: list[dict], weights: list[float], image_names: list[str],
         iou_thr: float = 0.55, skip_box_thr: float = 0.005) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fuse per-image predictions from several members with weighted box fusion.

    A single member is passed through untouched: WBF would additionally merge
    overlapping boxes inside one model's own output, which is not what NMS already
    tuned for.
    """
    if len(members) == 1:
        only = members[0]
        return {name: (only[name][0], only[name][1], only[name][2]) for name in image_names}

    from ensemble_boxes import weighted_boxes_fusion

    fused: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in image_names:
        boxes_list, scores_list, labels_list = [], [], []
        width = height = 1
        for member in members:
            if name not in member:
                continue
            xyxy, conf, cls, (width, height) = member[name]
            if len(xyxy) == 0:
                boxes_list.append(np.zeros((0, 4), np.float32))
                scores_list.append(np.zeros(0, np.float32))
                labels_list.append(np.zeros(0, np.int32))
                continue
            norm = xyxy.copy()
            norm[:, [0, 2]] /= width
            norm[:, [1, 3]] /= height
            boxes_list.append(np.clip(norm, 0.0, 1.0).astype(np.float32))
            scores_list.append(conf.astype(np.float32))
            labels_list.append(cls.astype(np.int32))
        if not boxes_list or all(len(b) == 0 for b in boxes_list):
            fused[name] = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))
            continue
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        )
        boxes = boxes.astype(np.float32)
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        fused[name] = (boxes, scores.astype(np.float32), labels.astype(np.int32))
    return fused


def load_gt(labels_dir: Path, stem: str, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    """COCO-style xywh ground truth rows for one image."""
    path = labels_dir / f"{stem}.txt"
    rows: list[tuple[int, float, float, float, float]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) != 5:
            continue
        c, x, y, w, h = int(float(f[0])), *map(float, f[1:])
        bw, bh = w * width, h * height
        rows.append((c, (x - w / 2) * width, (y - h / 2) * height, bw, bh))
    return rows
