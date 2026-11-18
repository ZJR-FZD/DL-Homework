"""Print a markdown table summarising every training run in runs/."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/hy-tmp/cd")
RUNS = ROOT / "runs"


def main() -> None:
    rows = []
    for cfg_path in sorted(RUNS.glob("*/config.json")):
        name = cfg_path.parent.name
        if name.startswith(("sweep", "infer", "test_infer", "submit")):
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        res = cfg_path.parent / "results.csv"
        if not res.is_file():
            continue
        recs = list(csv.DictReader(res.open(encoding="utf-8")))
        if not recs:
            continue
        key95 = [k for k in recs[0] if "mAP50-95" in k][0]
        key50 = [k for k in recs[0] if "mAP50" in k and "95" not in k][0]
        best = max(recs, key=lambda d: float(d[key95] or 0))
        last = recs[-1]
        rows.append(
            {
                "name": name,
                "data": cfg.get("dataset", ""),
                "model": Path(cfg.get("model", "")).stem,
                "imgsz": cfg.get("imgsz", ""),
                "ep_stop": len(recs),
                "ep_best": int(float(best["epoch"])),
                "map50_95": round(float(best[key95]), 4),
                "map50": round(float(best[key50]), 4),
                "last": round(float(last[key95]), 4),
                "min": round(float(last.get("time", 0)) / 60, 1),
                "hsv_s": cfg.get("hsv_s", ""),
                "fliplr": cfg.get("fliplr", ""),
                "split": cfg.get("split_dir", ""),
            }
        )

    if not rows:
        print("no runs yet")
        return
    cols = ["name", "data", "model", "imgsz", "ep_best", "ep_stop", "map50_95", "map50", "last", "min", "hsv_s", "fliplr", "split"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in sorted(rows, key=lambda x: -x["map50_95"]):
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
