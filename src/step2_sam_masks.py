import numpy as np , cv2 , torch
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CKPT = "D:/nosh_assignment/src/checkpoints/sam2.1_hiera_tiny.pt"
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = SAM2ImagePredictor(build_sam2(CFG,CKPT,device=device))

def sam_mask(path,point=None):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if point is None:
        point = (w//2,h//2)
    predictor.set_image(rgb)
    masks,scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels = np.array([1]),
        multimask_output = True)
    best = masks[int(np.argmax(scores))]
    return bgr,best.astype(np.uint8)*255


Path("data/masks").mkdir(exist_ok = True)
Path("output/overlay").mkdir(parents=True,exist_ok=True)

for p in sorted(Path("data/all_images").iterdir()):
    brg,mask = sam_mask(p)
    cv2.imwrite(f"data/masks/{p.stem}.png",mask)
    over = brg.copy()
    over[mask > 0] = (0.5 * over[mask>0]+0.5*np.array([0,0,225])).astype(np.uint8)
    cv2.imwrite(f"outputs/overlay/{p.stem}.jpg",over)