"""Run the phase3 queue once the running phase2 queue reports it is finished."""

import subprocess
import time
from pathlib import Path

LOG = Path("/hy-tmp/cd/runs/queue.log")


def done(marker: str) -> bool:
    return LOG.is_file() and marker in LOG.read_text(encoding="utf-8", errors="ignore")


deadline = time.time() + 24 * 3600
while not done("QUEUE phase2 FINISHED") and time.time() < deadline:
    time.sleep(30)

with LOG.open("a", encoding="utf-8") as fh:
    fh.write("[chain3] phase2 finished, launching phase3\n")

subprocess.run(["bash", "/hy-tmp/cd/tools/run_phase3.sh"], cwd="/hy-tmp/cd")
