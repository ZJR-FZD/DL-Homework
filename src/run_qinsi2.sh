#!/bin/bash
set -u
cd /hy-tmp/cd || exit 1
LOG=/hy-tmp/cd/runs/queue.log

run() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name :: $*" >> "$LOG"
  python3 tools/train_exp.py --name "$name" "$@" > "runs/$name.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE  $name rc=$rc" >> "$LOG"
}

run n_s960 --dataset qinsi --model yolo26s.pt --imgsz 960 --batch 16 --epochs 120 --patience 100 --mixup 0.05 --scale 0.7
run n_m_long --dataset qinsi --model yolo26m.pt --imgsz 640 --batch 32 --epochs 180 --patience 120 --mixup 0.10 --scale 0.7 --seed 0
run n_s_seed --dataset qinsi --model yolo26s.pt --imgsz 640 --batch 32 --epochs 120 --patience 100 --seed 7 --mixup 0.20 --scale 0.9 --hsv-s 0.9

echo "[$(date '+%F %T')] QUEUE qinsi2 FINISHED" >> "$LOG"
