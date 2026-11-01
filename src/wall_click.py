import cv2, numpy as np, sys

frame_no = int(sys.argv[1]) if len(sys.argv) > 1 else 500
cap = cv2.VideoCapture("D:/nosh_assignment/input.mp4")
if not cap.isOpened():
    raise SystemExit("could not open the video: check the path")
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ok, img = cap.read()
if not ok:
    raise SystemExit(f"could not read frame {frame_no}")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

def f(e, x, y, fl, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        L, a, b = lab[y, x]
        print(f"L={L:.0f} chroma={np.hypot(a - 128, b - 128):.1f}")

cv2.namedWindow("wall"); cv2.setMouseCallback("wall", f)
cv2.imshow("wall", img); cv2.waitKey(0)