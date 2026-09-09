"""Per-point feature extraction for the learned classifier.

The learned model is trained on the SAME measurement the classical baseline
uses (ICP -> signed point-to-plane deviation -> kNN smoothing), plus local
context features. This keeps the comparison fair: the classifier replaces the
hard global threshold with a location-aware, neighbourhood-aware decision.

Feature vector (12-d) per scanned point:
   0 dev_mm         signed deviation from CAD (mm), kNN-smoothed (k=3)
   1 abs_dev_mm     |dev_mm|
   2 dev_lmean      mean of dev over k_local scanned-neighbours
   3 dev_lstd       std of dev over k_local neighbours
   4 dev_lmax       max |dev| over k_local neighbours
   5 coh_frac       fraction of k_local neighbours with |dev| > 0.5 mm
                    (coherence: real damage is a connected patch, boundary
                     artefacts are isolated -> key FP-suppression signal)
   6 x              chord coordinate (m), ICP-aligned
   7 z              span coordinate (m), ICP-aligned
   8 nrm_x          scanned surface normal x
   9 nrm_z          scanned surface normal z
  10 nrm_disp       ||n_i - mean(n_neighbours)||  (curvature-ish)
  11 edge_dist      min chordwise distance (m) to LE/TE edges of the scan
                    (alignment artefacts concentrate near LE/TE -> FP source)
"""
import numpy as np
from scipy.spatial import cKDTree

from baselines.classical_pipeline import (icp_pp, point_to_plane_distance,
                                          smooth_dev)

FEAT_NAMES = ["dev_mm", "abs_dev_mm", "dev_lmean", "dev_lstd", "dev_lmax",
              "coh_frac", "x", "z", "nrm_x", "nrm_z", "nrm_disp", "edge_dist"]


def deviation_field(scen, nominal, nominal_nrm, tree):
    """Classical pipeline core: ICP align, signed point-to-plane dev, smooth.

    Returns (aligned, dev_mm): ICP-aligned scan points (n,3) and the
    kNN-smoothed signed deviation in mm (n,).
    """
    pts = scen["scanned"]
    G, tG, _cost = icp_pp(pts, nominal, tree=tree)
    aligned = pts @ G + tG
    signed = point_to_plane_distance(aligned, nominal, nominal_nrm, tree=tree)
    dev_s = smooth_dev(aligned, signed, k=3)
    return aligned, dev_s * 1000.0


def extract_scene_features(scen, nominal, nominal_nrm, tree, k_local=12):
    """Run the classical deviation pipeline on one scene and build features.

    Returns (X, y, dev_mm):
      X      (n, 12) float32 feature matrix
      y      (n,)    bool ground-truth damage mask
      dev_mm (n,)    float32 signed deviation (mm)
    """
    aligned, dev_mm = deviation_field(scen, nominal, nominal_nrm, tree)
    pts = scen["scanned"]

    n = len(pts)
    kk = max(2, min(k_local, n))
    tree_l = cKDTree(pts)
    _, nn = tree_l.query(pts, k=kk)
    dv = dev_mm[nn]                                  # (n, kk)
    lmean = dv.mean(axis=1)
    lstd = dv.std(axis=1)
    lmax = np.abs(dv).max(axis=1)
    coh = (np.abs(dv) > 0.5).mean(axis=1)

    nr = scen["scanned_nrm"]
    nb_nrm = nr[nn].mean(axis=1)
    nrm_disp = np.linalg.norm(nr - nb_nrm, axis=1)

    x0, x1 = float(aligned[:, 0].min()), float(aligned[:, 0].max())
    edge = np.minimum(aligned[:, 0] - x0, x1 - aligned[:, 0])

    X = np.column_stack([dev_mm, np.abs(dev_mm).astype(np.float32), lmean,
                         lstd, lmax, coh, aligned[:, 0], aligned[:, 2],
                         nr[:, 0], nr[:, 2], nrm_disp, edge])
    y = scen["gt_mask"].astype(bool)
    return X.astype(np.float32), y, dev_mm.astype(np.float32)


def extract_pointnet_features(scen, nominal, nominal_nrm, tree):
    """Per-point (7-d) features for the PyTorch PointNet model:
    aligned xyz (m), scanned normal xyz, dev_mm.

    Returns (X, y, dev_mm) with X (n, 7) float32.
    """
    aligned, dev_mm = deviation_field(scen, nominal, nominal_nrm, tree)
    nr = scen["scanned_nrm"]
    X = np.column_stack([aligned, nr, dev_mm])
    y = scen["gt_mask"].astype(bool)
    return X.astype(np.float32), y, dev_mm.astype(np.float32)