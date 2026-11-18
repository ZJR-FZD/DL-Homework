"""Sweep confidence threshold (and TTA) on the local validation split.

The submitted result.json only contains predictions above a confidence
threshold, and the grader computes AP from that file. So the threshold used at
inference time directly changes the score -- this measures how much.

Usage:
    python3 tools/eval_sweep.py --weights runs/q_s1024_s/weights/best.pt \
        --dataset qichen --imgsz 1024 --confs 0.001 0.01 0.05 0.25 --tta 0 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path("/hy-tmp/cd")
CLASS_FILES = {"qichen": "class.txt", "qinsi": "class.txt"}


def read_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            names.append(fields[1].strip())
    return names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", required=True, choices=["qichen", "qinsi"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--confs", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.25])
    p.add_argument("--tta", type=int, nargs="+", default=[0, 1])
    args = p.parse_args()

    root = ROOT / "split" / args.dataset
    names = read_names(root / CLASS_FILES[args.dataset])
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    yaml_path = ROOT / "runs" / f"sweep_{args.dataset}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    from ultralytics import YOLO

    model = YOLO(args.weights, task="detect")
    rows = []
    for tta in args.tta:
        for conf in args.confs:
            m = model.val(
                data=str(yaml_path),
                imgsz=args.imgsz,
                batch=args.batch,
                conf=conf,
                iou=0.7,
                augment=bool(tta),
                device="0",
                plots=False,
                verbose=False,
                split="val",
            )
            box = m.box
            rows.append(
                {
                    "tta": tta,
                    "conf": conf,
                    "map50_95": round(float(box.map), 4),
                    "map50": round(float(box.map50), 4),
                    "precision": round(float(box.mp), 4),
                    "recall": round(float(box.mr), 4),
                    "per_class_map50_95": [round(float(x), 4) for x in box.maps],
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    out = ROOT / "runs" / f"sweep_{args.dataset}_{Path(args.weights).parent.parent.name}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sweep] saved {out}")


if __name__ == "__main__":
    main()
