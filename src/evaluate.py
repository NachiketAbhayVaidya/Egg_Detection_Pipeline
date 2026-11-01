import cv2,json,sys,numpy as np
from pathlib import Path
from metrics import scores
import classical

METHOD = getattr(classical,sys.argv[1])
SPLIT = json.load(open("src/split.json"))["dev"]

rows=[]
for name in SPLIT:
    stem = Path(name).stem
    bgr = cv2.imread(f"data/all_images/{name}")
    gt = cv2.imread(f"data/masks/{stem}.png",0)
    pred = METHOD(bgr)
    rows.append((name,scores(gt,pred),bgr,gt,pred))

keys = ["iou","dice","precision","recall"]
means = {k:np.mean([r[1][k] for r in rows])for k in keys}
print(sys.argv[1],{k: round(v,4)for k , v in means.items()})


out = Path(f"outputs/trials/{sys.argv[1]}"); out.mkdir(parents=True, exist_ok=True)
for name, s, bgr, gt, pred in sorted(rows, key=lambda r: r[1]["iou"])[:5]:
    panel = np.hstack([bgr, cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR),
                       cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(str(out / f"{s['iou']:.2f}_{Path(name).stem[:20]}.jpg"), panel)