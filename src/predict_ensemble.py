"""WBF-ensemble inference that produces a schema-valid result.json.

Fusing several differently-regularised checkpoints gave +1.2 mAP50-95 over the
best single model on the local split, so the final submission uses this instead
of a single checkpoint. The output reuses the official Pydantic schema.

Usage:
    python3 tools/predict_ensemble.py \
        --members "runs/q_hsv/weights/best.pt:1024,runs/q_mos05/weights/best.pt:1024" \
        --image-dir datasets/qichen/images/val --out runs/infer_v2/qichen/result.json \
        --group-id 0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/hy-tmp/cd")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import wbf_lib  # noqa: E402
from src.object_detection.yolo_inference import (  # noqa: E402
    ImagePrediction,
    InferenceResult,
    ObjectPrediction,
    ResultSummary,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--members", required=True, help="comma list of path:imgsz")
    p.add_argument("--weights-list", default="")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--iou-thr", type=float, default=0.55)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--group-id", type=int, default=0)
    p.add_argument("--class-names", default="handwriting,tick,cross")
    args = p.parse_args()

    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    names = [n.strip() for n in args.class_names.split(",") if n.strip()]

    specs = []
    for item in args.members.split(","):
        path, _, size = item.partition(":")
        specs.append((path, int(size or 1024)))
    weights = [float(x) for x in args.weights_list.split(",")] if args.weights_list else [1.0] * len(specs)

    members = []
    for path, size in specs:
        print(f"[ens] predicting {path} @ {size}", flush=True)
        members.append(wbf_lib.predict_member(path, size, image_dir, args.conf, bool(args.tta), args.batch))

    image_names = sorted(p.name for p in image_dir.iterdir() if p.is_file())
    fused = wbf_lib.fuse(members, weights, image_names, iou_thr=args.iou_thr, skip_box_thr=args.conf / 2)

    images: list[ImagePrediction] = []
    for name in image_names:
        _, _, _, (width, height) = members[0][name]
        boxes, scores, labels = fused[name]
        boxes = boxes.copy()
        # clamp to the image and drop degenerate boxes, so every bbox satisfies 0 <= xmin < xmax <= width
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        keep = (boxes[:, 2] - boxes[:, 0] > 1.0) & (boxes[:, 3] - boxes[:, 1] > 1.0)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        order = scores.argsort()[::-1]
        preds = [
            ObjectPrediction(
                prediction_id=i,
                class_id=int(labels[j]),
                class_name=names[int(labels[j])] if int(labels[j]) < len(names) else str(int(labels[j])),
                confidence_score=float(scores[j]),
                bbox_xyxy=(float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3])),
            )
            for i, j in enumerate(order)
        ]
        images.append(ImagePrediction(image_path=str(image_dir / name), width=int(width), height=int(height), predictions=preds))

    payload = InferenceResult(
        group_id=args.group_id,
        generated_at=datetime.now().astimezone(),
        model="+".join(str(Path(m).resolve()) for m, _ in specs),
        summary=ResultSummary(image_count=len(images), prediction_count=sum(len(x.predictions) for x in images)),
        images=images,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    InferenceResult.model_validate_json(out.read_text(encoding="utf-8"))
    print(f"[ens] wrote {out} images={payload.summary.image_count} boxes={payload.summary.prediction_count} "
          f"size={out.stat().st_size / 1e6:.2f}MB", flush=True)


if __name__ == "__main__":
    main()
