import json
from ultralytics import YOLO
m = YOLO('runs/n_s640_s/weights/best.pt', task='detect')
out = []
for size in (512, 576, 608, 640, 704, 768):
    r = m.val(data='split/qinsi.yaml', split='val', imgsz=size, batch=16,
              conf=0.001, iou=0.7, plots=False, verbose=False)
    out.append({'imgsz': size, 'map50_95': round(float(r.box.map), 4),
                'map50': round(float(r.box.map50), 4),
                'P': round(float(r.box.mp), 4), 'R': round(float(r.box.mr), 4)})
    print(json.dumps(out[-1]), flush=True)
