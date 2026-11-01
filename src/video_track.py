import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
import video_mask

VIDEO = "D:/nosh_assignment/input.mp4"

MAX_JUMP = 1.0
GROW = 0.5
BIG = 1e9
MAX_DRAW = 12
MAX_FOLLOW = 120
BOTTOM_MIN = 140
EDGE_Y = 129
WHITE_MIN = 0.05
WHITE_PIXELS = 80
DRIFT_MAX = 1.5
REJECT = {"csrt_lost": 0, "too_long": 0, "drift": 0, "above_table": 0, "not_white": 0}


def contour_of(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea)

def bottom_of(cnt):
    x, y, bw, bh = cv2.boundingRect(cnt)
    return y + bh

def make_tracker(frame, cnt):
    x, y, bw, bh = cv2.boundingRect(cnt)
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        f = getattr(cv2, name, None) or getattr(getattr(cv2, "legacy", None), name, None)
        if f:
            tr = f()
            tr.init(frame, (int(x), int(y), int(bw), int(bh)))
            return tr
    raise RuntimeError("no OpenCV tracker available")

def assign(tracks, ids, dets):
    if not dets or not ids:
        return {}
    cost = np.full((len(ids), len(dets)), BIG)
    for i, tid in enumerate(ids):
        t = tracks[tid]
        gate = t["width"] * (MAX_JUMP + GROW * t["missed"])
        for j, d in enumerate(dets):
            dist = np.hypot(d["cx"] - t["cx"], d["cy"] - t["cy"])
            if dist <= gate:
                cost[i, j] = dist
    rows, cols = linear_sum_assignment(cost)
    return {ids[r]: dets[c] for r, c in zip(rows, cols) if cost[r, c] < BIG}

def track_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    tracks, frames, prev = {}, [], None

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev is not None and tracks and cv2.absdiff(gray, prev).mean() > 40:
            for t in tracks.values():
                t["missed"] = 10**6
        prev = gray


        dets = [d for d in video_mask.egg_blobs(bgr, k=4)
                if bottom_of(contour_of(d["mask"])) > BOTTOM_MIN][:2]

        if not tracks:
            if len(dets) == 2:
                dets.sort(key=lambda d: d["cx"])
                for tid, d in zip((1, 2), dets):
                    cnt0 = contour_of(d["mask"])
                    tracks[tid] = dict(cx=d["cx"], cy=d["cy"], width=np.sqrt(d["area"]),
                                       cnt=cnt0, missed=0, tracker=make_tracker(bgr, cnt0),
                                       rx=d["cx"], ry=d["cy"])
                frames.append({t: (tracks[t]["cnt"], True) for t in tracks})
            else:
                frames.append({})
            continue

        ids = list(tracks)
        matched = assign(tracks, ids, dets)
        wm = video_mask.white_mask(bgr)

        out = {}
        for tid in ids:
            t = tracks[tid]
            if tid in matched:
                d = matched[tid]
                t.update(cx=d["cx"], cy=d["cy"], width=np.sqrt(d["area"]),
                         cnt=contour_of(d["mask"]), missed=0)
                t["tracker"] = make_tracker(bgr, t["cnt"])
                t["rx"], t["ry"] = d["cx"], d["cy"]
                out[tid] = (t["cnt"], True)
            else:
                t["missed"] += 1
                followed = False
                ok2, box = t["tracker"].update(bgr)
                if not ok2:
                    REJECT["csrt_lost"] += 1
                elif t["missed"] > MAX_FOLLOW:
                    REJECT["too_long"] += 1
                else:
                    bx, by, bw, bh = box
                    x0, y0 = int(max(bx, 0)), int(max(by, 0))
                    x1, y1 = int(min(bx + bw, w)), int(min(by + bh, h))
                    ys, xs = np.nonzero(wm[y0:y1, x0:x1])
                    if len(xs) < WHITE_PIXELS:
                        REJECT["not_white"] += 1
                    else:
                        nx, ny = x0 + xs.mean(), y0 + ys.mean()
                        shift = np.array([[[int(nx - t["cx"]), int(ny - t["cy"])]]], dtype=np.int32)
                        new_cnt = t["cnt"] + shift
                        drift = np.hypot(nx - t["rx"], ny - t["ry"])
                        if drift >= DRIFT_MAX * t["width"]:
                            REJECT["drift"] += 1
                        elif bottom_of(new_cnt) <= BOTTOM_MIN:
                            REJECT["above_table"] += 1
                        else:
                            t["cnt"] = new_cnt
                            t["cx"], t["cy"] = nx, ny
                            followed = True

                if followed or (t["missed"] <= MAX_DRAW and bottom_of(t["cnt"]) > BOTTOM_MIN):
                    out[tid] = (t["cnt"], False)
        frames.append(out)

    cap.release()
    return frames, fps, (w, h)

if __name__ == "__main__":
    frames, fps, size = track_video(VIDEO)
    n = len(frames)
    for tid in (1, 2):
        real = sum(1 for f in frames if tid in f and f[tid][1])
        print(f"Egg {tid}: real detection in {real}/{n} frames ({real / n:.1%})")