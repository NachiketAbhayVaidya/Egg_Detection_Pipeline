import cv2, json, numpy as np
from pathlib import Path
from metrics import scores
import classical

dev = json.load(open("src/split.json"))["dev"]


runs = [
    ("default thr=0.65 wa=1.5", dict()),
    ("thr=0.50",                dict(thr=0.50)),
    ("thr=0.80",                dict(thr=0.80)),
    ("wa=1.0",                  dict(wa=1.0)),
    ("wa=2.0",                  dict(wa=2.0)),
]

imgs = [(cv2.imread(f"data/all_images/{n}"),
         cv2.imread(f"data/masks/{Path(n).stem}.png", 0)) for n in dev]

print(f"{'setting':26} {'IoU':>6} {'Dice':>6} {'Prec':>6} {'Rec':>6} {'median':>7} {'<0.1':>5} {'>0.6':>5}")
for label, kw in runs:
    s = [scores(gt, classical.t12b(bgr, **kw)) for bgr, gt in imgs]
    iou = np.array([x["iou"] for x in s])
    print(f"{label:26} {iou.mean():6.3f} {np.mean([x['dice'] for x in s]):6.3f} "
          f"{np.mean([x['precision'] for x in s]):6.3f} {np.mean([x['recall'] for x in s]):6.3f} "
          f"{np.median(iou):7.3f} {(iou < 0.1).sum():5d} {(iou > 0.6).sum():5d}")