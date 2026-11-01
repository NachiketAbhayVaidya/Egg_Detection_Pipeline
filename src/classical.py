import cv2
import numpy as np

def t1_otsu(bgr):
    gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    _,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return mask

def t5_kmeans(bgr, K=4):
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (160, 160))
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    data = lab.reshape(-1, 3).astype(np.float32)

    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)

    egg = int(np.argmax(centers[:, 0]))
    mask = (labels.reshape(160, 160) == egg).astype(np.uint8) * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

def t9_select(bgr, K=3):
    h, w = bgr.shape[:2]
    mask = t5_kmeans(bgr, K)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    best, best_score, fallback = None, -1e9, (0, None)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > fallback[0]:
            fallback = (area, i)
        frac = area / (h * w)
        if frac < 0.01 or frac > 0.70:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = cv2.contourArea(cnt) / hull_area if hull_area else 0

        cx, cy = cents[i]
        dist = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        touches = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h

        score = solidity - 1.0 * dist - (0.3 if touches else 0.0)
        if score > best_score:
            best, best_score = i, score

    if best is None:
        best = fallback[1]
    out = np.zeros((h, w), np.uint8)
    if best is not None:
        comp = (labels == best).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, 255, thickness=cv2.FILLED)
    return out

def cand_kmeans(bgr, K):
    m = t5_kmeans(bgr, K)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, _, _ = cv2.connectedComponentsWithStats(m)
    return [(labels == i).astype(np.uint8) * 255 for i in range(1, n)]

def cand_grabcut(bgr):
    h, w = bgr.shape[:2]
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    cv2.ellipse(gc, (w // 2, h // 2), (int(w * 0.40), int(h * 0.40)), 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.ellipse(gc, (w // 2, h // 2), (int(w * 0.20), int(h * 0.20)), 0, 0, 360, cv2.GC_FGD, -1)
    bg = np.zeros((1, 65), np.float64); fg = np.zeros((1, 65), np.float64)
    small = cv2.resize(bgr, (160, 160)); gcs = cv2.resize(gc, (160, 160), interpolation=cv2.INTER_NEAREST)
    cv2.grabCut(small, gcs, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    m = ((gcs == cv2.GC_FGD) | (gcs == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    return [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)]

def cand_edge_ellipse(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if len(c) < 20:
            continue
        frac = cv2.contourArea(c) / (h * w)
        if frac < 0.03 or frac > 0.70:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, cv2.fitEllipse(c), 255, -1)
        out.append(m)
    return out

def pick(bgr, comps):
    h, w = bgr.shape[:2]
    best, best_score = None, -1e9
    for m in comps:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        frac = area / (h * w)
        if frac < 0.01 or frac > 0.70:
            continue
        hull = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull if hull else 0
        x, y, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        dist = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        touches = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h
        score = solidity - 1.0 * dist - (0.3 if touches else 0.0)
        if score > best_score:
            best, best_score = m, score
    return best if best is not None else np.zeros((h, w), np.uint8)

def t12_pool(bgr):
    comps = []
    for K in (3, 4, 5):
        comps += cand_kmeans(bgr, K)
    comps += cand_grabcut(bgr)
    comps += cand_edge_ellipse(bgr)
    return pick(bgr, comps)


def _feats(m, grad, h, w):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt); frac = area / (h * w)
    if frac < 0.01 or frac > 0.70 or len(cnt) < 5:
        return None
    hull = cv2.contourArea(cv2.convexHull(cnt))
    (ex, ey), (ea, eb), _ = cv2.fitEllipse(cnt)
    ell_area = np.pi * ea * eb / 4
    M = cv2.moments(cnt); cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    x, y, bw, bh = cv2.boundingRect(cnt)
    pts = cnt[:, 0, :]
    return dict(
        t9=(area / hull if hull else 0)
           - np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
           - (0.3 if (x <= 0 or y <= 0 or x + bw >= w or y + bh >= h) else 0.0),
        bloat=max(0.0, frac - 0.35),
        ell=1 - min(1.0, abs(1 - area / ell_area)) if ell_area else 0,
        edge=float(grad[pts[:, 1], pts[:, 0]].mean()) / 255.0)

def t12b(bgr, thr=0.65, wa=1.5, we=1.5, wg=0.25):
    h, w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    grad = np.clip(grad / grad.max() * 255, 0, 255)

    def best(comps, extended):
        top, top_s = None, -1e9
        for m in comps:
            f = _feats(m, grad, h, w)
            if f is None:
                continue
            s = f["t9"] + (-wa * f["bloat"] + we * f["ell"] + wg * f["edge"] if extended else 0)
            if s > top_s:
                top, top_s = m, s
        return top, top_s

    m9, s9 = best(cand_kmeans(bgr, 3), False)
    if m9 is not None and s9 >= thr:
        return m9
    pool = []
    for K in (3, 4, 5):
        pool += cand_kmeans(bgr, K)
    pool += cand_grabcut(bgr) + cand_edge_ellipse(bgr)
    m, _ = best(pool, True)
    return m if m is not None else np.zeros((h, w), np.uint8)