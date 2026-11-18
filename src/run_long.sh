#!/bin/bash
set -u
cd /hy-tmp/cd || exit 1
LOG=/hy-tmp/cd/runs/queue.log
echo "[$(date '+%F %T')] START q_long :: full 120 epochs, late close_mosaic" >> "$LOG"
python3 tools/train_exp.py --name q_long --dataset qichen --model yolo26s.pt --imgsz 1024 --batch 12   --epochs 120 --patience 100 --workers 12 --hsv-h 0.005 --hsv-s 0.25 --hsv-v 0.30   --fliplr 0.0 --mosaic 0.5 --close-mosaic 20 > runs/q_long.log 2>&1
echo "[$(date '+%F %T')] DONE q_long rc=$?" >> "$LOG"
