import cv2
from pathlib import Path

cap = cv2.VideoCapture("D:/nosh_assignment/input.mp4")
if not cap.isOpened():
    raise SystemExit("could not open data/video/input.mp4")

n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"frames: {n} | fps: {fps:.1f} | size: {w}x{h} | length: {n / fps:.1f}s")

out = Path("outputs/video/frames"); out.mkdir(parents=True, exist_ok=True)
for i in range(8):
    idx = int(i * (n - 1) / 7)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(str(out / f"frame_{idx:05d}.jpg"), frame)
cap.release()
print("saved to", out)