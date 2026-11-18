import json
from ultralytics import YOLO

out = []
m = YOLO('runs/n_s640_s/weights/best.pt', task='detect')

r = m.val(data='split/qinsi.yaml', split='train', imgsz=640, batch=32,
          conf=0.001, iou=0.7, fraction=0.3, plots=False, verbose=False)
out.append({'probe': 'train_fit', 'imgsz': 640, 'map50_95': round(float(r.box.map), 4),
            'map50': round(float(r.box.map50), 4), 'P': round(float(r.box.mp), 4), 'R': round(float(r.box.mr), 4)})

for size in (960, 1280):
    r = m.val(data='split/qinsi.yaml', split='val', imgsz=size, batch=16,
              conf=0.001, iou=0.7, plots=False, verbose=False)
    out.append({'probe': 'val_upscale', 'imgsz': size, 'map50_95': round(float(r.box.map), 4),
                'map50': round(float(r.box.map50), 4), 'P': round(float(r.box.mp), 4), 'R': round(float(r.box.mr), 4)})

print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
