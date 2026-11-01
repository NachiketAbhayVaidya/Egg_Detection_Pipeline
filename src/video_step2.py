import cv2
from pathlib import Path
import video_mask

out = Path("outputs/video/mask_debug"); out.mkdir(parents=True, exist_ok=True)
for p in sorted(Path("outputs/video/frames").glob("*.jpg")):
    bgr = cv2.imread(str(p))
    view = bgr.copy()
    for i, c in enumerate(video_mask.egg_blobs(bgr)):
        cnts, _ = cv2.findContours(c["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, cnts, -1, (0, 165, 255) if i == 0 else (255, 160, 0), 4)
    raw = cv2.cvtColor(video_mask.white_mask(bgr), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out / p.name), cv2.hconcat([view, raw]))
print("saved to", out)