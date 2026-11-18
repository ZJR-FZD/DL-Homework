"""Generate submission result.json for both tasks and pack the zip.

Wraps the official src/object_detection/yolo_inference.py so the output is
guaranteed to pass the Pydantic schema check.

Usage:
    python3 tools/predict_submit.py --qichen runs/q_s1024_s/weights/best.pt \
        --qinsi runs/n_s640_s/weights/best.pt --tag v1 --group-id 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/hy-tmp/cd")
INFERENCE = ROOT / "src" / "object_detection" / "yolo_inference.py"

DATASETS = {
    "qichen": {
        "image_dir": ROOT / "datasets" / "qichen" / "images" / "val",
        "imgsz": 1024,
        "folder": "qichen_val_labels",
    },
    "qinsi": {
        "image_dir": ROOT / "datasets" / "qinsi" / "images" / "val",
        "imgsz": 640,
        "folder": "qinsi_val_labels",
    },
}


def run_inference(dataset: str, members: str, tag: str, group_id: int, conf: float, tta: int) -> Path:
    spec = DATASETS[dataset]
    out_dir = ROOT / "runs" / f"infer_{tag}" / dataset
    result = out_dir / "result.json"
    class_names = "handwriting,tick,cross" if dataset == "qichen" else "person"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "predict_ensemble.py"),
        "--members",
        members,
        "--image-dir",
        str(spec["image_dir"]),
        "--out",
        str(result),
        "--conf",
        str(conf),
        "--batch",
        "4",
        "--group-id",
        str(group_id),
        "--tta",
        str(tta),
        "--class-names",
        class_names,
    ]
    print(f"[submit] {dataset}: members={members} TTA={bool(tta)} conf={conf}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qichen", required=True,
                   help="qichen members, comma list of path:imgsz (one or several for WBF)")
    p.add_argument("--qinsi", required=True, help="qinsi members, comma list of path:imgsz")
    p.add_argument("--tag", default="v1", help="version tag used in output folder and zip name")
    p.add_argument("--group-id", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.01, help="confidence threshold written into result.json")
    p.add_argument("--tta", type=int, default=1, help="1 = test-time augmentation")
    p.add_argument("--name-prefix", default="48-23网络工程-202330452242-曾婧茹", help="zip name prefix")
    args = p.parse_args()

    stage = ROOT / "runs" / f"submit_{args.tag}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    members = {"qichen": args.qichen, "qinsi": args.qinsi}
    for dataset, folder in (("qichen", "qichen_val_labels"), ("qinsi", "qinsi_val_labels")):
        result = run_inference(dataset, members[dataset], args.tag, args.group_id, args.conf, args.tta)
        target_dir = stage / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, target_dir / "result.json")
        data = json.loads(result.read_text(encoding="utf-8"))
        print(f"[submit] {dataset}: images={data['summary']['image_count']} "
              f"boxes={data['summary']['prediction_count']} "
              f"size={result.stat().st_size/1e6:.2f}MB")

    zip_path = ROOT / "runs" / f"{args.name_prefix}-inference_result-{args.tag}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("qichen_val_labels", "qinsi_val_labels"):
            for f in sorted((stage / folder).rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(stage))
    print(f"[submit] zip={zip_path} size={zip_path.stat().st_size/1e6:.2f}MB")


if __name__ == "__main__":
    main()
