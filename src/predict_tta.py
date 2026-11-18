"""TTA-capable inference that reuses the official result schema.

The official ``yolo_inference.py`` hard-codes ``model.predict(...)`` without
``augment``. Test-time augmentation gave +1.5 mAP50-95 on the qichen validation
split (mostly on the rare ``cross`` class), so this script runs the same
predictions with ``augment=True`` but reuses the official ``convert_result`` and
Pydantic models, guaranteeing the JSON stays schema-compatible.

Usage:
    python3 tools/predict_tta.py --weights runs/q_hsv/weights/best.pt \
        --image-dir datasets/qichen/images/val --imgsz 1024 \
        --out runs/infer_v1/qichen/result.json --group-id 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/hy-tmp/cd")
sys.path.insert(0, str(ROOT))

from src.object_detection.yolo_inference import (  # noqa: E402
    ImagePrediction,
    InferenceResult,
    ResultSummary,
    convert_result,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out", required=True, help="path of result.json to write")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--group-id", type=int, default=0)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights).resolve()
    model = YOLO(str(weights), task="detect")
    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    workspace = Path.cwd()
    print(f"[tta] {image_dir} imgsz={args.imgsz} conf={args.conf} tta={bool(args.tta)}", flush=True)
    stream = model.predict(
        source=str(image_dir),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        max_det=args.max_det,
        augment=bool(args.tta),
        device=args.device,
        stream=True,
        verbose=False,
    )
    images: list[ImagePrediction] = [convert_result(r, workspace) for r in stream]

    payload = InferenceResult(
        group_id=args.group_id,
        generated_at=datetime.now().astimezone(),
        model=str(weights),
        summary=ResultSummary(
            image_count=len(images),
            prediction_count=sum(len(img.predictions) for img in images),
        ),
        images=images,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # round-trip through the official validator so a bad file fails here, not at grading
    InferenceResult.model_validate_json(out.read_text(encoding="utf-8"))
    print(f"[tta] wrote {out} images={payload.summary.image_count} boxes={payload.summary.prediction_count} "
          f"size={out.stat().st_size / 1e6:.2f}MB", flush=True)


if __name__ == "__main__":
    main()
