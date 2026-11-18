"""Wait for the current job to finish, then replace the long phase2 queue with the short final one."""

import os
import signal
import subprocess
import time
from pathlib import Path

LOG = Path("/hy-tmp/cd/runs/queue.log")


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[chain4] {msg}\n")


def pids_matching(needle: str) -> list[int]:
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        if needle in cmdline and int(entry.name) != os.getpid():
            out.append(int(entry.name))
    return out


def training_alive() -> bool:
    return bool(pids_matching("tools/train_exp.py"))


def done(marker: str) -> bool:
    return LOG.is_file() and marker in LOG.read_text(encoding="utf-8", errors="ignore")


deadline = time.time() + 6 * 3600
while training_alive() and time.time() < deadline:
    time.sleep(20)

# The old queue shell would go on to launch the long tail of low-value runs.
for pid in pids_matching("tools/run_queue.sh"):
    try:
        os.kill(pid, signal.SIGTERM)
        log(f"killed stale queue shell {pid}")
    except OSError:
        pass
for pid in pids_matching("tools/chain2.py") + pids_matching("tools/chain3.py"):
    try:
        os.kill(pid, signal.SIGTERM)
        log(f"killed stale chain {pid}")
    except OSError:
        pass

time.sleep(3)
if done("QUEUE final STARTED"):
    log("final queue already running, exit")
    raise SystemExit

log("launching final queue")
with LOG.open("a", encoding="utf-8") as fh:
    fh.write("QUEUE final STARTED\n")
subprocess.run(["bash", "/hy-tmp/cd/tools/run_final.sh"], cwd="/hy-tmp/cd")
