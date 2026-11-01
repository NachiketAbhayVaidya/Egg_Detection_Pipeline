import cv2
from pathlib import Path

imgs = sorted(Path("D:/nosh_assignment/data/all_images").iterdir())
masks = Path("data/masks")
bad = []
for p in imgs:
    m = cv2.imread(str(masks/f"{p.stem}.png"),0)
    im = cv2.imread(str(p))
    if m is None:
        bad.append((p.name,"missing mask"));continue
    if m.shape != im.shape[:2]:
        bad.append((p.name, f"size mismatch {m.shape} vs {im.shape[:2]}")); continue
    frac = (m>0).mean()
    if frac < 0.02 or frac > 0.60:
        bad.append((p.name, f"suspicious egg area {frac:.1%}"))

print("images:",len(imgs),"|problems:",len(bad))
for b in bad: print(b)