"""Metrics for the classical baseline (point-level detection + geometric quantification)."""
import numpy as np


def prf(mask, gt):
    mask = np.asarray(mask, bool); gt = np.asarray(gt, bool)
    tp = int(np.sum(mask & gt)); fp = int(np.sum(mask & ~gt))
    fn = int(np.sum(~mask & gt)); tn = int(np.sum(~mask & ~gt))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return dict(precision=precision, recall=recall, f1=f1, iou=iou, tp=tp, fp=fp, fn=fn, tn=tn)


def geo_err(dev_m, gt_depth_mm, mask):
    """On GT-damaged points, compare recovered |deviation| (mm) vs true depth (mm)."""
    dev_m = np.asarray(dev_m, float); gt = np.asarray(gt_depth_mm, float)
    m = np.asarray(mask, bool) & (gt > 0)
    if m.sum() == 0:
        return dict(mae=np.nan, rmse=np.nan, n=0)
    d = np.abs(dev_m[m]) * 1000.0 - gt[m]
    return dict(mae=float(np.mean(np.abs(d))),
                rmse=float(np.sqrt(np.mean(d ** 2))), n=int(m.sum()))