import json, statistics
from pathlib import Path
from ultralytics import YOLO

img = Path('split/qinsi/images/val')
files = sorted(p for p in img.iterdir() if p.is_file())[:200]
m = YOLO('runs/n_s640_s/weights/best.pt', task='detect')
res = m.predict([str(p) for p in files], imgsz=640, conf=0.25, verbose=False, batch=16)
sides, per_img = [], []
for r in res:
    h, w = r.orig_shape
    n = 0
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes.xyxy.cpu().numpy():
            bw, bh = (b[2]-b[0])/w, (b[3]-b[1])/h
            sides.append((bw*bh)**0.5); n += 1
    per_img.append(n)
sides.sort()
print(json.dumps({
  'n_img': len(files),
  'pred_boxes': len(sides),
  'median_side_rel': round(float(sides[len(sides)//2]), 4) if sides else None,
  'p10': round(float(sides[len(sides)//10]), 4) if sides else None,
  'p90': round(float(sides[int(len(sides)*0.9)]), 4) if sides else None,
  'objs_per_img_mean': round(statistics.mean(per_img), 2),
  'objs_per_img_median': statistics.median(per_img),
}, ensure_ascii=False), flush=True)
