import numpy as np

def to_bool(mask):
    return mask > 0

def counts(gt, pred):
    gt, pred = to_bool(gt), to_bool(pred)
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    return tp, fp, fn

def scores(gt, pred):
    tp, fp, fn = counts(gt, pred)
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return dict(iou=iou, dice=dice, precision=prec, recall=rec)