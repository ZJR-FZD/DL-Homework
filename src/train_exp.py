"""Course-design experiment runner: thin, explicit wrapper around Ultralytics training.

The official script ``src/object_detection/yolo_training.py`` only exposes a few
hyper-parameters. Experiments on class imbalance and resolution need more knobs,
so this wrapper keeps the same dataset layout / YAML convention and passes the
augmentation settings through explicitly.

Every run writes ``<project>/<name>/config.json`` so the experiment is
reproducible from the artefacts alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_FILES = {"qichen": "class.txt", "qinsi": "class.txt"}


def read_class_names(path: Path) -> list[str]:
    names: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            names.append((int(fields[0]), fields[1].strip()))
    names.sort()
    assert [i for i, _ in names] == list(range(len(names))), f"non-contiguous ids in {path}"
    return [n for _, n in names]


def write_dataset_yaml(dataset_dir: Path, names: list[str]) -> Path:
    yaml_path = dataset_dir.parent / f"{dataset_dir.name}.yaml"
    lines = [f"path: {json.dumps(str(dataset_dir.resolve()), ensure_ascii=False)}",
             "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {json.dumps(n, ensure_ascii=False)}" for i, n in enumerate(names)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/hy-tmp/cd"))
    p.add_argument("--dataset", required=True, choices=sorted(CLASS_FILES))
    p.add_argument("--split-dir", default="split", help="dataset root folder name under --root")
    p.add_argument("--model", default="yolo26s.pt")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--name", required=True)
    p.add_argument("--project", type=Path, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--cache", default="ram")
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--close-mosaic", type=int, default=20)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy-paste", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--extra", default="{}", help="extra ultralytics kwargs as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    dataset_dir = args.root / args.split_dir / args.dataset
    class_file = dataset_dir / CLASS_FILES[args.dataset]
    if not class_file.is_file():
        class_file = args.root / "datasets" / args.dataset / CLASS_FILES[args.dataset]
    names = read_class_names(class_file)
    data_yaml = write_dataset_yaml(dataset_dir, names)

    project = args.project or (args.root / "runs")
    run_dir = project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["classes"] = names
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cache=(False if args.cache in ("none", "false") else args.cache),
        amp=bool(args.amp),
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        scale=args.scale,
        degrees=args.degrees,
        translate=args.translate,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        fliplr=args.fliplr,
        flipud=args.flipud,
        plots=True,
        val=True,
        task="detect",
    )
    train_kwargs.update(json.loads(args.extra))

    print(f"[train_exp] dataset={args.dataset} classes={names}")
    print(f"[train_exp] yaml={data_yaml}")
    print("[train_exp] " + json.dumps({k: v for k, v in train_kwargs.items() if k != "data"}, ensure_ascii=False))

    model = YOLO(args.model, task="detect")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
