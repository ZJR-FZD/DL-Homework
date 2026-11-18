"""Wait for the in-flight training job to finish, then run the phase2 queue."""

import subprocess
import time
from pathlib import Path

LOG = Path("/hy-tmp/cd/runs/queue.log")


def training_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "tools/train_exp.py"], capture_output=True, text=True)
    return bool(out.stdout.strip())


with LOG.open("a", encoding="utf-8") as fh:
    fh.write("[chain2] started, waiting for current job to finish\n")

deadline = time.time() + 12 * 3600
while training_running() and time.time() < deadline:
    time.sleep(20)

with LOG.open("a", encoding="utf-8") as fh:
    fh.write("[chain2] training idle, launching phase2\n")

subprocess.run(["bash", "/hy-tmp/cd/tools/run_queue.sh", "phase2"], cwd="/hy-tmp/cd")
