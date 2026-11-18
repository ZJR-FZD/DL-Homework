#!/bin/bash
# Sequential experiment queue for the course design (single 4090, one job at a time).
# Usage:  bash tools/run_queue.sh phase1|phase2|phase3
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

case "${1:-phase1}" in
  phase1)
    # 主线：启辰高分辨率 + 勤思基线
    run q_s1024_s --dataset qichen --model yolo26s.pt --imgsz 1024 --batch 12 --epochs 150 --patience 30 --workers 12
    run n_s640_s  --dataset qinsi  --model yolo26s.pt --imgsz 640  --batch 32 --epochs 150 --patience 30 --workers 12
    # 消融：启辰 640 对照（回答“为什么要高分辨率”）
    run q_s640_s  --dataset qichen --model yolo26s.pt --imgsz 640  --batch 32 --epochs 120 --patience 30 --workers 12
    # 消融：勤思 960 对照
    run n_s960_s  --dataset qinsi  --model yolo26s.pt --imgsz 960  --batch 16 --epochs 100 --patience 30 --workers 12
    ;;
  phase2)
    # 诊断结论：tick/cross 的瓶颈是“召回 0.67/0.64 + 误检多”，不是框精度
    # 因此优先做：色彩先验（让红笔更显著）、降 mosaic（保住大勾的尺度）、过采样
    # A. 红笔色彩先验：削弱 HSV 抖动，保留“红墨 vs 黑字”这一最强线索
    run q_hsv     --dataset qichen --model yolo26s.pt --imgsz 1024 --batch 12 --epochs 150 --patience 30 --workers 12 \
                  --hsv-h 0.005 --hsv-s 0.25 --hsv-v 0.30
    # B. 降低 mosaic：拼图会把大勾缩小到 1/2，可能压低 tick 召回
    run q_mos05   --dataset qichen --model yolo26s.pt --imgsz 1024 --batch 12 --epochs 150 --patience 30 --workers 12 \
                  --mosaic 0.5 --close-mosaic 30
    # C. 更高分辨率
    run q_s1280_s --dataset qichen --model yolo26s.pt --imgsz 1280 --batch 8 --epochs 150 --patience 30 --workers 12
    # D. 组合：过采样（cross×4 tick×2）+ 低色彩抖动
    run q_os_hsv  --split-dir split_os --dataset qichen --model yolo26s.pt --imgsz 1024 --batch 12 --epochs 150 --patience 30 --workers 12 \
                  --hsv-h 0.005 --hsv-s 0.25 --hsv-v 0.30
    # E. 取消水平翻转：镜像的中文字与镜像的勾都是不存在的形态
    run q_noflip  --dataset qichen --model yolo26s.pt --imgsz 1024 --batch 12 --epochs 150 --patience 30 --workers 12 \
                  --fliplr 0.0
    # F. 模型规模（可能同时抬召回与压误检）
    run q_m1024_m --dataset qichen --model yolo26m.pt --imgsz 1024 --batch 8 --epochs 150 --patience 30 --workers 12
    run q_m1280_m --dataset qichen --model yolo26m.pt --imgsz 1280 --batch 4 --epochs 150 --patience 30 --workers 12
    # G. 低价值对照（报告用）
    run q_s640_s  --dataset qichen --model yolo26s.pt --imgsz 640  --batch 32 --epochs 120 --patience 30 --workers 12
    run n_m640_m  --dataset qinsi  --model yolo26m.pt --imgsz 640  --batch 32 --epochs 150 --patience 30 --workers 12
    run n_s960_s  --dataset qinsi  --model yolo26s.pt --imgsz 960  --batch 16 --epochs 100 --patience 30 --workers 12
    ;;
  phase3)
    # 最终配置（按 phase2 结论填，先留占位）
    ;;
esac
echo "[$(date '+%F %T')] QUEUE ${1:-phase1} FINISHED" >> "$LOG"
