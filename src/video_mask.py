import cv2, numpy as np

L_MIN = 205
CHROMA_MAX = 10

def white_mask(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.sqrt(a * a + b * b)
    m = ((L > L_MIN) & (chroma < CHROMA_MAX)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

def split_blob(comp):
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    seeds = (dist > 0.6 * dist.max()).astype(np.uint8)
    ns, seed_lab = cv2.connectedComponents(seeds)
    if ns <= 2:
        return [comp]
    markers = np.zeros(comp.shape, np.int32)
    markers[comp == 0] = 1
    markers[seeds > 0] = seed_lab[seeds > 0] + 1
    cv2.watershed(cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR), markers)
    return [((markers == k).astype(np.uint8) * 255) for k in range(2, ns + 1)]

def describe(part, h, w):
    cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if len(cnt) < 5 or area / (h * w) < 0.005 or area / (h * w) > 0.25:
        return None
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    solidity = area / hull_area
    (_, _), (ea, eb), _ = cv2.fitEllipse(cnt)
    aspect = max(ea, eb) / max(1.0, min(ea, eb))
    if solidity < 0.85 or aspect > 2.0:
        return None
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [hull], -1, 255, cv2.FILLED)
    M = cv2.moments(hull)
    return dict(score=solidity - 0.5 * abs(aspect - 1.3), mask=filled,
                cx=M["m10"] / M["m00"], cy=M["m01"] / M["m00"],
                area=int(hull_area))

def egg_blobs(bgr, k=2):
    h, w = bgr.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask(bgr))
    cands = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] / (h * w) < 0.005:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        for part in split_blob(comp):
            c = describe(part, h, w)
            if c:
                cands.append(c)
    cands.sort(key=lambda c: -c["score"])
    return cands[:k]