import cv2, json, csv, numpy as np
from pathlib import Path
from metrics import scores
import classical

ev = json.load(open("src/split.json"))["eval"]
rows = []
for n in ev:
    bgr = cv2.imread(f"data/all_images/{n}")
    gt = cv2.imread(f"data/masks/{Path(n).stem}.png", 0)
    pred = classical.t12b(bgr, thr=0.65, wa=1.5)
    s = scores(gt, pred)
    rows.append((n, s["iou"], s["dice"], s["precision"], s["recall"]))

Path("outputs/eval").mkdir(parents=True, exist_ok=True)
with open("outputs/eval/per_image.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "iou", "dice", "precision", "recall"]); w.writerows(rows)

iou = np.array([r[1] for r in rows])
print("EVAL n =", len(rows))
print("mean IoU", iou.mean().round(4), "| median", np.median(iou).round(4))
print("mean Dice", np.mean([r[2] for r in rows]).round(4))
print("precision", np.mean([r[3] for r in rows]).round(4), "recall", np.mean([r[4] for r in rows]).round(4))
print("IoU > 0.5:", int((iou > 0.5).sum()), "of", len(iou), "| IoU < 0.1:", int((iou < 0.1).sum()))