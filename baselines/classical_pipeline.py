"""Classical baseline: ICP alignment -> point-to-plane signed distance -> threshold.

Open3D is used when importable (HAVE_O3D=True); otherwise a mathematically
equivalent scipy-based Kabsch point-to-point ICP fallback runs, so real numbers
are produced either way. Row-vector transform convention: x' = x @ R + t.
"""
import numpy as np
from scipy.spatial import cKDTree

try:
    import open3d as o3d  # type: ignore
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False


def point_to_plane_distance(query, ref_pts, ref_nrm, tree=None):
    """Signed distance (m) of each query point to the reference point cloud surface."""
    if tree is None:
        tree = cKDTree(ref_pts)
    _, nn = tree.query(query, k=1)
    nn = nn.reshape(-1)
    vec = query - ref_pts[nn]
    return (vec * ref_nrm[nn]).sum(axis=1)


def _kabsch_rows(X, Y):
    """Row vectors. Returns (R, t) minimizing ||Y - (X @ R + t)||_F, R a proper rotation."""
    cx = X.mean(axis=0); cy = Y.mean(axis=0)
    Xc = X - cx; Yc = Y - cy
    H = Xc.T @ Yc
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    t = cy - cx @ R
    return R, t


def icp_pp(src, tgt, max_iter=40, tol=1e-9, tree=None):
    """Point-to-point ICP aligning src -> tgt. Returns (G, tG, cost); aligned = src @ G + tG."""
    if tree is None:
        tree = cKDTree(tgt)
    G = np.eye(3); tG = np.zeros(3)
    cur = src.copy(); cost = np.inf
    for _ in range(max_iter):
        _, nn = tree.query(cur, k=1)
        nn = nn.reshape(-1)
        tp = tgt[nn]
        Ri, ti = _kabsch_rows(cur, tp)
        cur = cur @ Ri + ti
        G = G @ Ri
        tG = tG @ Ri + ti
        new_cost = float(np.mean(np.linalg.norm(cur - tp, axis=1)))
        if abs(cost - new_cost) < tol:
            cost = new_cost
            break
        cost = new_cost
    return G, tG, cost


def smooth_dev(pts, dev, k=15):
    """Local kNN mean of the per-point signed deviation (denoising)."""
    k = max(2, min(k, len(pts)))
    tree = cKDTree(pts)
    _, nn = tree.query(pts, k=k)
    return dev[nn].mean(axis=1)


def run_scene(scen, tau_mm, smooth_k=3):
    from .evaluate import prf, geo_err
    G, tG, cost = icp_pp(scen["scanned"], scen["nominal"])
    aligned = scen["scanned"] @ G + tG
    signed = point_to_plane_distance(aligned, scen["nominal"], scen["nominal_nrm"])
    dev_s = smooth_dev(aligned, signed, k=smooth_k)
    mask = np.abs(dev_s) > (tau_mm * 1e-3)
    res = prf(mask, scen["gt_mask"])
    res.update(geo_err(dev_s, scen["gt_depth_mm"], scen["gt_mask"]))
    res.update(icp_cost_mm=cost * 1000.0, tau_mm=float(tau_mm), n=int(len(scen["scanned"])))
    return res, (G, tG)