import cv2, numpy as np
from pathlib import Path

flagged = []
for p in sorted(Path("data/all_images").iterdir()):
    m = cv2.imread(f"data/masks/{p.stem}.png", 0) > 0
    m8 = m.astype(np.uint8)
    why = []
    if m.mean() > 0.25: why.append(f"big {m.mean():.0%}")
    edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()
    if edge > 0.10: why.append(f"touches border {edge:.0%}")
    cnts, hier = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None and (hier[0][:, 3] >= 0).any(): why.append("has hole")
    if len(cnts) > 3: why.append(f"{len(cnts)} pieces")
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ha = cv2.contourArea(cv2.convexHull(c))
        if ha and cv2.contourArea(c) / ha < 0.85: why.append("ragged shape")
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) > 0 and max(rw, rh) / min(rw, rh) > 3: why.append("thin strip")
    if why: flagged.append((p.stem, why))

print("flagged:", len(flagged))
for s, w in flagged: print(s, w)
Path("flagged.txt").write_text("\n".join(s for s, _ in flagged))