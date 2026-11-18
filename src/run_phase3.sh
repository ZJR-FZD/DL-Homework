#!/bin/bash
# Phase 3: stack the priors that worked (low HSV jitter, less mosaic, oversampling, no flip)
# and measure seed variance, since the 180-image val split is noisy.
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

COMMON="--dataset qichen --model yolo26s.pt --imgsz 1024 --batch 12 --epochs 150 --patience 30 --workers 12 --hsv-h 0.005 --hsv-s 0.25 --hsv-v 0.30 --fliplr 0.0"

# A. 叠加版：过采样 + 弱色彩抖动 + 降 mosaic + 不翻转
run q_combo --split-dir split_os $COMMON --mosaic 0.5 --close-mosaic 40
# B. 干脆不用 mosaic（对象都偏大，拼图收益存疑）
run q_mos0  --split-dir split_os $COMMON --mosaic 0.0
# C. 集成用的第二个视角：yolo26m
run q_combo_m --split-dir split_os --dataset qichen --model yolo26m.pt --imgsz 1024 --batch 8 --epochs 150 --patience 30 --workers 12 \
              --hsv-h 0.005 --hsv-s 0.25 --hsv-v 0.30 --fliplr 0.0 --mosaic 0.5 --close-mosaic 40
# D. 种子方差：同一配置换两个种子，用来估计验证集噪声带来的波动
run q_combo_s1 --split-dir split_os $COMMON --mosaic 0.5 --close-mosaic 40 --seed 1
run q_combo_s2 --split-dir split_os $COMMON --mosaic 0.5 --close-mosaic 40 --seed 2

echo "[$(date '+%F %T')] QUEUE phase3 FINISHED" >> "$LOG"
