"""Scene-level engineering metrics: WHERE is the damage, HOW BIG, HOW MUCH.

Point-level P/R/F1 answers "did the points look right". An inspector also
needs: damage area (m2), damage volume (cm3, shallow-defect approximation
V = integral of depth dA), and localization (centroid distance between
predicted and true damage regions).

Area model: the scan samples points uniformly over the full blade surface,
so each surviving point represents area = S_mesh / n_kept (m2).
Volume: V = sum_i depth_i * a_w  (shallow defect: depth x local area).
"""
import numpy as np

LOC_TOL_M = 0.5  # "localized" if predicted centroid within 0.5 m of true one


def mesh_area(V, F):
    """Total surface area (m2) of the triangulated blade mesh."""
    tri = np.asarray(V, dtype=float)[np.asarray(F)]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * float(np.sum(np.linalg.norm(cr, axis=1)))


def scene_damage_metrics(mask, gt, pts, dev_mm, gt_depth_mm, a_w):
    """Engineering quantities for one scene.

    mask/gt: bool (n,)   pts: (n,3) aligned coords (m)
    dev_mm: (n,) measured signed deviation (mm); gt_depth_mm: (n,) true depth (mm)
    a_w: per-point area weight (m2)
    """
    mask = np.asarray(mask, bool); gt = np.asarray(gt, bool)
    area_gt = float(a_w * gt.sum()); area_pred = float(a_w * mask.sum())
    vol_gt = float(a_w * 1000.0 * np.sum(gt_depth_mm[gt]))        # cm3
    vol_pred = float(a_w * 1000.0 * np.sum(np.abs(dev_mm[mask])))  # cm3
    row = dict(n_gt=int(gt.sum()), n_pred=int(mask.sum()),
               area_gt_m2=area_gt, area_pred_m2=area_pred,
               vol_gt_cm3=vol_gt, vol_pred_cm3=vol_pred,
               d_centroid_m=np.nan)
    if gt.any() and mask.any():
        row["d_centroid_m"] = float(np.linalg.norm(pts[mask].mean(0) - pts[gt].mean(0)))
    return row


def aggregate_scene_metrics(rows, types, type_names, type_order):
    """Aggregate per-scene rows into per-type engineering summaries."""
    out = []
    for t in type_order:
        sel = [r for r, ty in zip(rows, types) if ty == t]
        if not sel:
            continue
        agt = np.array([r["area_gt_m2"] for r in sel])
        apd = np.array([r["area_pred_m2"] for r in sel])
        vgt = np.array([r["vol_gt_cm3"] for r in sel])
        vpd = np.array([r["vol_pred_cm3"] for r in sel])
        dc = np.array([r["d_centroid_m"] for r in sel])
        dmg = vgt > 0.0
        valid = ~np.isnan(dc)
        out.append(dict(
            type=int(t), type_name=type_names[t], n_scenes=len(sel),
            area_gt_m2=float(agt.mean()), area_pred_m2=float(apd.mean()),
            area_abs_err_m2=float(np.abs(apd - agt).mean()),
            area_rel_err=(float(np.mean(np.abs(apd[dmg] - agt[dmg])
                                    / np.maximum(agt[dmg], 1e-6))) if dmg.any() else np.nan),
            vol_gt_cm3=float(vgt.mean()), vol_pred_cm3=float(vpd.mean()),
            vol_mae_cm3=(float(np.mean(np.abs(vpd[dmg] - vgt[dmg]))) if dmg.any() else np.nan),
            vol_rel_err=(float(np.mean(np.abs(vpd[dmg] - vgt[dmg])
                                     / np.maximum(vgt[dmg], 1e-6))) if dmg.any() else np.nan),
            centroid_err_m=(float(dc[valid].mean()) if valid.any() else np.nan),
            loc_ok_rate=(float((dc[valid] < LOC_TOL_M).mean()) if valid.any() else np.nan),
            n_pred_points=float(np.mean([r["n_pred"] for r in sel])),
        ))
    return out