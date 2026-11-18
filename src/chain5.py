"""After the long yolo26s run finishes, train the yolo26m counterpart with the same recipe."""

import subprocess
import time
from pathlib import Path

LOG = Path("/hy-tmp/cd/runs/queue.log")


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[chain5] {msg}\n")


def done(marker: str) -> bool:
    return LOG.is_file() and marker in LOG.read_text(encoding="utf-8", errors="ignore")


deadline = time.time() + 3 * 3600
while not done("DONE q_long") and time.time() < deadline:
    time.sleep(20)

log("q_long finished, launching q_long_m")
with LOG.open("a", encoding="utf-8") as fh:
    fh.write("[chain5] START q_long_m\n")

subprocess.run(
    [
        "python3", "tools/train_exp.py",
        "--name", "q_long_m",
        "--dataset", "qichen",
        "--model", "yolo26m.pt",
        "--imgsz", "1024",
        "--batch", "8",
        "--epochs", "100",
        "--patience", "80",
        "--workers", "12",
        "--hsv-h", "0.005",
        "--hsv-s", "0.25",
        "--hsv-v", "0.30",
        "--fliplr", "0.0",
        "--mosaic", "0.5",
        "--close-mosaic", "20",
    ],
    cwd="/hy-tmp/cd",
    stdout=(Path("/hy-tmp/cd/runs/q_long_m.log")).open("w"),
    stderr=subprocess.STDOUT,
)
log("q_long_m finished")
