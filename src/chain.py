"""Wait for the phase-1 queue to finish, then start phase 2 automatically."""

import pathlib
import subprocess
import time

LOG = pathlib.Path("/hy-tmp/cd/runs/queue.log")
SCRIPT = "/hy-tmp/cd/tools/run_queue.sh"


def finished(marker: str) -> bool:
    return LOG.is_file() and marker in LOG.read_text(errors="ignore")


deadline = time.time() + 6 * 3600
while not finished("QUEUE phase1 FINISHED") and time.time() < deadline:
    time.sleep(30)

with LOG.open("a", encoding="utf-8") as handle:
    handle.write(f"[chain] phase1 marker seen, starting phase2\n")
subprocess.run(["bash", SCRIPT, "phase2"], cwd="/hy-tmp/cd")
