"""Evaluate one checkpoint on an arbitrary split (train / val) to find the fitting ceiling."""

import argparse
import json

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="dataset yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = p.parse_args()

    model = YOLO(args.weights, task="detect")
    kwargs = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
        workers=8,
    )
    if args.max_images:
        kwargs["fraction"] = min(1.0, args.max_images / 2000)
    res = model.val(**kwargs)
    names = res.names
    per_class = {}
    for i, name in names.items():
        try:
            per_class[name] = {
                "map50_95": round(float(res.box.maps[i]), 4),
                "map50": round(float(res.box.ap50[i]), 4),
            }
        except Exception:
            pass
    print(
        json.dumps(
            {
                "weights": args.weights,
                "split": args.split,
                "map50_95": round(float(res.box.map), 4),
                "map50": round(float(res.box.mp), 4),
                "per_class": per_class,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
