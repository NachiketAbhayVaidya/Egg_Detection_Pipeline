import cv2 , sys ,json ,numpy as np
from pathlib import Path
from metrics import scores
import classical

method = getattr(classical,sys.argv[1])
dev = json.load(open("src/split.json"))["dev"]

res = []

for name in dev:
    bgr = cv2.imread(f"data/all_images/{name}")
    gt = cv2.imread(f"data/masks/{Path(name).stem}.png", 0)
    res.append((scores(gt, method(bgr))["iou"], name))

ious = np.array([r[0] for r in res])
print("wrong blob   (IoU < 0.1):", int((ious < 0.1).sum()))
print("bad boundary (0.1-0.6)  :", int(((ious >= 0.1) & (ious <= 0.6)).sum()))
print("good         (IoU > 0.6):", int((ious > 0.6).sum()))
print("median IoU:", round(float(np.median(ious)), 3))

out = Path("outputs/diagnosis"); out.mkdir(parents=True, exist_ok=True)
for iou, name in sorted(res)[:10]:
    bgr = cv2.imread(f"data/all_images/{name}")
    gt = cv2.imread(f"data/masks/{Path(name).stem}.png", 0)
    pred = method(bgr)
    panel = np.hstack([bgr, cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR),
                       cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(str(out / f"{iou:.2f}_{Path(name).stem[:20]}.jpg"), panel)