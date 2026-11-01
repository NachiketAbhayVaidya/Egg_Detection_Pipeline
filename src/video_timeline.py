import cv2, numpy as np
import video_track

frames, fps, (w, h) = video_track.track_video("D:/nosh_assignment/input.mp4")
n = len(frames)

def ranges(flags):
    out, start = [], None
    for i, ok in enumerate(flags + [True]):
        if not ok and start is None:
            start = i
        if ok and start is not None:
            out.append((start, i - 1)); start = None
    return out

for tid in (1, 2):
    real = [(tid in f and f[tid][1]) for f in frames]
    gaps = [g for g in ranges(real) if g[1] - g[0] >= 15]
    print(f"\nEgg {tid}: {len(gaps)} gaps longer than 15 frames")
    for a, b in gaps:
        print(f"  frames {a}-{b}  ({a / fps:.1f}s to {b / fps:.1f}s, {b - a + 1} frames)")


cap = cv2.VideoCapture("data/video/input.mp4")
longest = []
for tid in (1, 2):
    real = [(tid in f and f[tid][1]) for f in frames]
    longest += sorted(ranges(real), key=lambda g: g[0] - g[1])[:3]
for a, b in longest:
    cap.set(cv2.CAP_PROP_POS_FRAMES, (a + b) // 2)
    ok, fr = cap.read()
    if ok:
        cv2.imwrite(f"outputs/video/gap_{(a + b) // 2:05d}.jpg", fr)