import cv2,numpy as np,torch
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor









ONLY = Path("flagged.txt").read_text().split()

CKPT = "D:/nosh_assignment/src/checkpoints/sam2.1_hiera_tiny.pt"
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = SAM2ImagePredictor(build_sam2(CFG,CKPT,device=device))

IMG_DIR , MASK_DIR = Path("data/all_images"),Path("data/masks")
pts,labels = [] , []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x,y]);labels.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts.append([x,y]);labels.append(0)


def segment(bgr):
    predictor.set_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    m,_,_ = predictor.predict(point_coords=np.array(pts),
                              point_labels=np.array(labels),
                              multimask_output=False)
    return (m[0]*255).astype(np.uint8)

def show(bgr,mask,name):
    view=bgr.copy()
    if mask is not None:
        view[mask>0] = (0.5*view[mask>0]+0.5*np.array([0,0,255])).astype(np.uint8)
    for(x,y),l in zip(pts,labels):
        cv2.circle(view, (x, y), 6, (0, 255, 0) if l else (255, 0, 0), -1)
    cv2.imshow("sam", view); cv2.setWindowTitle("sam", name)

cv2.namedWindow("sam"); cv2.setMouseCallback("sam", on_mouse)

for p in sorted(IMG_DIR.iterdir()):
    if not any(k in p.name for k in ONLY):
        continue
    bgr = cv2.imread(str(p))
    mask_path = MASK_DIR / f"{p.stem}.png"
    current = cv2.imread(str(mask_path), 0) if mask_path.exists() else None
    pts.clear(); labels.clear()
    while True:
        show(bgr, current, f"{p.name} | click, SPACE=run, A=accept, C=clear, S=keep old, Q=quit")
        k = cv2.waitKey(30) & 0xFF
        if k == ord(" ") and pts:  current = segment(bgr)
        elif k == ord("c"):        pts.clear(); labels.clear(); current = None
        elif k == ord("a") and current is not None:
            cv2.imwrite(str(mask_path), current); break
        elif k == ord("s"):        break
        elif k == ord("q"):        raise SystemExit
        elif k == ord("i") and current is not None:
            current = 255 - current
cv2.destroyAllWindows()