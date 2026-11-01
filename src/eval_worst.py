import cv2, csv, numpy as np
from pathlib import Path
import classical

rows = sorted(csv.DictReader(open("outputs/eval/per_image.csv")), key=lambda r: float(r["iou"]))[:5]
Path("outputs/eval/worst").mkdir(parents=True, exist_ok=True)
for r in rows:
    bgr = cv2.imread(f"data/all_images/{r['file']}")
    gt = cv2.imread(f"data/masks/{Path(r['file']).stem}.png", 0)
    pred = classical.t12b(bgr, thr=0.65, wa=1.5)
    panel = np.hstack([bgr, cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(f"outputs/eval/worst/{float(r['iou']):.2f}_{Path(r['file']).stem[:20]}.jpg", panel)
    print(f"{float(r['iou']):.2f}", r["file"])