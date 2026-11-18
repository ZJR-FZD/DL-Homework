#!/bin/bash
# Final queue: only the experiments that still have a plausible payoff.
# Everything below feeds either the report's ablation table or the final ensemble.
set -u
cd /hy-tmp/cd || exit 1
mkdir -p runs
LOG=/hy-tmp/cd/runs/queue.log

run() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name :: $*" >> "$LOG"
  python3 tools/train_exp.py --name "$name" "$@" > "runs/$name.log" 2>&1
  local rc=$?
  local best
  best=$(python3 - "$name" <<'PY'
import sys, csv, pathlib
name = sys.argv[1]
p = pathlib.Path("runs") / name / "results.csv"
if not p.is_file():
    print("no results.csv"); raise SystemExit
rows = list(csv.DictReader(p.open(encoding="utf-8")))
if not rows:
    print("empty"); raise SystemExit
key = [k for k in rows[0] if "mAP50-95" in k][0]
r = max(rows, key=lambda d: float(d[key] or 0))
print(f"epoch={r['epoch']} mAP50-95={float(r[key]):.4f} mAP50={float(r[next(k for k in rows[0] if 'mAP50' in k and '95' not in k)] or 0):.4f}")
PY
)
  echo "[$(date '+%F %T')] DONE  $name rc=$rc $best" >> "$LOG"
}

# Everything that showed a positive effect, stacked at once, on the oversampled split.
COMBO="--split-dir split_os --dataset qichen --model yolo26s.pt --imgsz 1024 --batch 12 \
       --epochs 150 --patience 30 --workers 12 --hsv-h 0.005 --hsv-s 0.25 --hsv-v 0.30 \
       --fliplr 0.0 --mosaic 0.5 --close-mosaic 40"

# A. the stacked config itself
run q_combo    $COMBO
# B. same config, different seed -> ensemble member + seed-variance estimate
run q_combo_s1 $COMBO --seed 1
# C. bigger backbone on the same recipe -> ensemble diversity
run q_combo_m  --split-dir split_os --dataset qichen --model yolo26m.pt --imgsz 1024 --batch 8 \
               --epochs 150 --patience 30 --workers 12 --hsv-h 0.005 --hsv-s 0.25 --hsv-v 0.30 \
               --fliplr 0.0 --mosaic 0.5 --close-mosaic 40
# D. no mosaic at all (all objects are large; tiling may only hurt)
run q_mos0     $COMBO --mosaic 0.0

echo "[$(date '+%F %T')] QUEUE final FINISHED" >> "$LOG"
