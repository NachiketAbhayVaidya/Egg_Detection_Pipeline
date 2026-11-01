import cv2, numpy as np, sys
img = cv2.imread(sys.argv[1])
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
def on_click(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        L, a, b = lab[y, x]
        print(f"L={L:.0f}  chroma={np.hypot(a - 128, b - 128):.1f}")
cv2.namedWindow("calib"); cv2.setMouseCallback("calib", on_click)
cv2.imshow("calib", img); cv2.waitKey(0)