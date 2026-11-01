import numpy as np
from metrics import scores

gt   = np.zeros((10, 20), np.uint8); gt[:, :10] = 255
pred = np.zeros((10, 20), np.uint8); pred[:8, 2:] = 255
print(scores(gt, pred))