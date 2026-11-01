import cv2, json, numpy as np
from pathlib import Path
from metrics import scores
import classical

def candidates(bgr, K=3):
    mask = classical.t5_kmeans(bgr, K)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, _, _ = cv2.connectedComponentsWithStats(mask)
    return [(labels == i).astype(np.uint8) * 255 for i in range(1, n)]

dev = json.load(open("src/split.json"))["dev"]
scoring, generation, other, best_ious = 0, 0, 0, []

for name in dev:
    bgr = cv2.imread(f"data/all_images/{name}")
    gt = cv2.imread(f"data/masks/{Path(name).stem}.png", 0)
    if scores(gt, classical.t9_select(bgr))["iou"] >= 0.1:
        continue
    best = max([scores(gt, c)["iou"] for c in candidates(bgr)], default=0.0)
    best_ious.append(best)
    if best > 0.5:   scoring += 1
    elif best < 0.3: generation += 1
    else:            other += 1

print("scoring failures   (best candidate > 0.5):", scoring)
print("generation failures (best candidate < 0.3):", generation)
print("in between:", other, "| median best-candidate IoU:", round(float(np.median(best_ious)), 3))