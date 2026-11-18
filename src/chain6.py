"""After the qichen long run ends, train the qinsi model with regularisation aimed at its generalisation gap."""

import subprocess
import time
from pathlib import Path

LOG = Path("/hy-tmp/cd/runs/queue.log")


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[chain6] {msg}\n")


def done(marker: str) -> bool:
    return LOG.is_file() and marker in LOG.read_text(encoding="utf-8", errors="ignore")


deadline = time.time() + 2 * 3600
while not done("DONE q_long") and time.time() < deadline:
    time.sleep(20)

log("q_long finished, starting qinsi generalisation run")
with LOG.open("a", encoding="utf-8") as fh:
    fh.write("[chain6] START n_m_aug\n")

# qinsi: train mAP 0.870 vs val 0.726 -> the model memorises the train scenes.
# Counter it with a bigger backbone and heavier photometric / scale augmentation.
subprocess.run(
    [
        "python3", "tools/train_exp.py",
        "--name", "n_m_aug",
        "--dataset", "qinsi",
        "--model", "yolo26m.pt",
        "--imgsz", "640",
        "--batch", "32",
        "--epochs", "90",
        "--patience", "60",
        "--workers", "12",
        "--mosaic", "1.0",
        "--close-mosaic", "20",
        "--mixup", "0.10",
        "--scale", "0.9",
        "--hsv-h", "0.015",
        "--hsv-s", "0.7",
        "--hsv-v", "0.5",
        "--fliplr", "0.5",
    ],
    cwd="/hy-tmp/cd",
    stdout=(Path("/hy-tmp/cd/runs/n_m_aug.log")).open("w"),
    stderr=subprocess.STDOUT,
)
log("n_m_aug finished")
