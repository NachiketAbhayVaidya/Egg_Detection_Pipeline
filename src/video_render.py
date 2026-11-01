import cv2
from pathlib import Path
import video_track

SHOW_LINE = False
COLORS = {1: (0, 140, 255), 2: (255, 160, 0)}

frames, fps, (w, h) = video_track.track_video(video_track.VIDEO)

Path("outputs/video").mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(video_track.VIDEO)
out = cv2.VideoWriter("outputs/video/tracked_rebuild.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

for i, state in enumerate(frames):
    ok, img = cap.read()
    if not ok:
        break
    for tid, (cnt, real) in state.items():
        col = COLORS[tid]
        if real:
            shade = img.copy()
            cv2.drawContours(shade, [cnt], -1, col, cv2.FILLED)
            img = cv2.addWeighted(shade, 0.35, img, 0.65, 0)
            cv2.drawContours(img, [cnt], -1, col, 3)
        else:
            cv2.drawContours(img, [cnt], -1, col, 1)
        x, y, bw, bh = cv2.boundingRect(cnt)
        label = f"Egg {tid}" + ("" if real else " (tracked)")
        ly = max(20, y - 8) if tid == 1 else min(h - 10, y + bh + 22)
        cv2.putText(img, label, (x, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, col, 2, cv2.LINE_AA)
    if SHOW_LINE:
        cv2.line(img, (0, video_track.EDGE_Y), (w, video_track.EDGE_Y), (0, 255, 0), 1)
    cv2.putText(img, f"frame {i}  {i / fps:.1f}s", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out.write(img)

cap.release(); out.release()
print("saved outputs/video/tracked_rebuild.mp4")