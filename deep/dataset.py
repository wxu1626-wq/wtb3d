"""In-memory dataset over generated scenes for the PyTorch models.

Uses the exact same scan generation + ICP deviation pipeline as
run_learned.py (same seeds -> same scenes -> directly comparable numbers).
Also precomputes per-point kNN indices (scene-local) for the
graph / attention backbones (PointNet++-style, DGCNN, PointTransformer).
"""
import os, sys
import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_gen.config import GenConfig
from data_gen.geometry import loft_blade
from data_gen import damage as dm
from data_gen.sensor_sim import to_scan, build_reference
from learned.features import extract_pointnet_features

TYPE_ORDER = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM, dm.HEALTHY]


def stratified_scene_split(types, train_frac, seed):
    """Scene-level stratified split by damage type (matches run_learned.py).

    Returns (train_idx, test_idx).
    """
    rng = np.random.default_rng(seed)
    types = np.asarray(types)
    n = len(types)
    is_train = np.zeros(n, dtype=bool)
    for t in TYPE_ORDER:
        idx = np.where(types == t)[0]
        rng.shuffle(idx)
        k = max(2, int(round(len(idx) * train_frac)))
        is_train[idx[:k]] = True
    return (np.where(is_train)[0].tolist(),
            np.where(~is_train)[0].tolist())


def build_arrays(num, points, n_ref, seed, knn_k=16, verbose=True):
    """Generate all scenes once, return stacked numpy arrays.

    Returns dict with:
      X (N,7) float32  [aligned xyz, scanned normal, dev_mm]
      y (N,) bool      gt damage mask
      dev_mm (N,)      signed deviation (mm)
      gt_depth_mm (N,) true damage depth (mm)
      knn (N,K) int32  scene-local kNN indices (self removed; tiny scenes
                       padded by repeating the last neighbour)
      offsets (S,)     start row of each scene in X
      lens (S,)        points per scene
      types (S,)       damage type per scene
    """
    from scipy.spatial import cKDTree
    import time
    cfg = GenConfig(num_scenes=num, points=points, n_ref=n_ref, seed=seed,
                    out_dir="data")
    rng = np.random.default_rng(cfg.seed)
    rng_ref = np.random.default_rng(cfg.seed + 1000)
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    nominal, nominal_nrm = build_reference(V, F, N, cfg.n_ref, rng_ref)
    tree = cKDTree(nominal)

    Xs, ys, devs, depths, types, knns = [], [], [], [], [], []
    t0 = time.time()
    for i in range(num):
        disp, lab, meta = dm.generate_damage(V, N, cfg, rng)
        scen = to_scan(V, F, N, disp, lab, cfg, rng, nominal, nominal_nrm)
        X, y, dev_mm, _aligned = extract_pointnet_features(scen, nominal,
                                                           nominal_nrm, tree)
        Xs.append(X); ys.append(y); devs.append(dev_mm)
        depths.append(scen["gt_depth_mm"].astype(np.float32))
        types.append(meta["type"])
        n = len(X)
        kk = max(1, min(knn_k, n - 1))
        _, ii = cKDTree(X[:, :3]).query(X[:, :3], k=kk + 1)
        ii = ii[:, 1:]                                   # drop self
        if kk < knn_k:                                    # tiny scene: pad
            ii = np.concatenate([ii] + [ii[:, -1:]
                                        for _ in range(knn_k - kk)], axis=1)
        knns.append(ii.astype(np.int32))
        if verbose and (i + 1) % 100 == 0:
            print("  gen %d/%d scenes, %.1fs" % (i + 1, num, time.time() - t0))
    lens = np.array([len(y) for y in ys])
    offsets = np.concatenate([[0], np.cumsum(lens)])[:-1]
    return dict(X=np.concatenate(Xs), y=np.concatenate(ys),
                dev_mm=np.concatenate(devs),
                gt_depth_mm=np.concatenate(depths),
                knn=np.concatenate(knns),
                offsets=offsets.astype(np.int64), lens=lens,
                types=np.asarray(types))


class SceneDataset(Dataset):
    """One sample = one scene (variable-length point set)."""

    def __init__(self, arrays, scene_idx):
        import torch
        self.arr = arrays
        self.idx = list(scene_idx)
        self.torch = torch

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        a = int(self.arr["offsets"][i]); b = a + int(self.arr["lens"][i])
        X = self.torch.from_numpy(self.arr["X"][a:b])
        y = self.torch.from_numpy(self.arr["y"][a:b].astype(np.float32))
        dev = self.torch.from_numpy(self.arr["dev_mm"][a:b])
        knn = self.torch.from_numpy(self.arr["knn"][a:b])
        return X, y, dev, knn


def pad_collate(batch):
    """Pad variable-length scenes to the batch max (masked in training).

    Returns (X (B,D,Nmax), Y (B,Nmax), DV (B,Nmax), mask (B,Nmax) bool,
             KN (B,Nmax,K) long). Padded knn entries stay in-range.
    """
    import torch
    Xs, ys, devs, knns = zip(*batch)
    nmax = max(x.shape[0] for x in Xs)
    B, D = len(Xs), Xs[0].shape[1]
    K = knns[0].shape[1]
    X = torch.zeros(B, D, nmax)
    Y = torch.zeros(B, nmax)
    DV = torch.zeros(B, nmax)
    mask = torch.zeros(B, nmax, dtype=torch.bool)
    KN = torch.zeros(B, nmax, K, dtype=torch.long)
    for b in range(B):
        n = Xs[b].shape[0]
        X[b, :, :n] = Xs[b].T
        Y[b, :n] = ys[b]
        DV[b, :n] = devs[b]
        mask[b, :n] = True
        KN[b, :n, :] = knns[b]
    return X, Y, DV, mask, KN