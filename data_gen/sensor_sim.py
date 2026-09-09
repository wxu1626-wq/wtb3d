"""Sensor simulation: rigid misregistration + Gaussian noise + dropout + optional occlusion.

Model = a DENSE undamaged CAD reference (nominal, fixed pose, shared across scenes)
plus a SPARSE scan of the DAMAGED surface that carries a small residual registration
offset T (R, t), sensor noise and dropout. This is the realistic CAD-assisted
inspection setting: the as-designed blade is always available; the field scan is
only roughly registered.

The 'scene' dict returned by to_scan() carries:
  nominal / nominal_nrm : dense reference (same object passed in, not copied)
  scanned / scanned_nrm : damaged + posed + noisy + dropout scan
  gt_mask, gt_depth_mm, type : per-scanned-point ground truth
  pose (R, t), n_kept
"""
import numpy as np
from .geometry import sample_surface


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rigid(rng, max_angle_deg, max_trans_mm):
    ang = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg, 3))
    R = _rot_z(ang[2]) @ _rot_y(ang[1]) @ _rot_x(ang[0])
    t = rng.uniform(-max_trans_mm, max_trans_mm, 3) * 1e-3
    return R, t


def build_reference(V, F, N, n_ref, rng):
    """Dense undamaged surface sample -> (nominal, nominal_nrm). Reuse for all scenes."""
    pts, nrm, _fa, _fb, _fc, _b = sample_surface(V, F, N, n_ref, rng)
    return pts, nrm


def to_scan(V, F, N, disp, lab, cfg, rng, nominal, nominal_nrm):
    Vd = V + N * disp[:, None]
    pts, nrm, fa, fb, fc, bary = sample_surface(Vd, F, N, cfg.points, rng)
    dpt = bary[:, 0] * disp[fa] + bary[:, 1] * disp[fb] + bary[:, 2] * disp[fc]
    vert = np.column_stack((fa, fb, fc))
    lab_pt = lab[vert[np.arange(cfg.points), np.argmax(bary, axis=1)]]
    R, t = random_rigid(rng, cfg.max_angle_deg, cfg.max_trans_mm)
    scanned = pts @ R.T + t
    nrm = nrm @ R
    scanned = scanned + rng.normal(0.0, cfg.noise_mm * 1e-3, scanned.shape)
    keep = rng.random(len(scanned)) > cfg.dropout
    if cfg.occlusion > 0.0:                 # single-side view: drop back-facing points
        keep &= (nrm[:, 1] > -0.2)
    idx = np.where(keep)[0]
    return {
        "nominal": nominal,
        "nominal_nrm": nominal_nrm,
        "scanned": scanned[idx],
        "scanned_nrm": nrm[idx],
        "gt_mask": (np.abs(dpt) > cfg.gt_eps)[idx],
        "gt_depth_mm": (np.abs(dpt) * 1000.0)[idx],
        "type": lab_pt[idx],
        "pose": (R, t),
        "n_kept": int(idx.size),
    }