"""Feature-subset ablation of the context-aware learned detector (CPU).

Question the paper needs answered: which parts of the 12-d feature vector
actually carry the damage signal? Progressive subsets (all other things
equal: same scenes, same split, same RF, threshold tuned on TRAIN only):

  A  raw deviation        dev_mm, abs_dev_mm
  B  + local statistics   dev_lmean, dev_lstd, dev_lmax
  C  + coherence          coh_frac
  D  + location           x, z
  E  + normal context     nrm_x, nrm_z, nrm_disp
  F  + edge distance      edge_dist  (= full 12-d)

Usage:
  python run_ablation.py --num 120 --points 20000
Outputs:
  results/ablation_<num>.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import csv, json, time, argparse
import numpy as np
from scipy.spatial import cKDTree

_DEFAULT_MPL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "work", "mpl"))
os.environ.setdefault("MPLCONFIGDIR", _DEFAULT_MPL)

from data_gen.config import GenConfig
from data_gen.geometry import loft_blade
from data_gen import damage as dm
from data_gen.sensor_sim import to_scan, build_reference
from learned.features import extract_scene_features, FEAT_NAMES
from learned.models import make_model

TYPE_ORDER = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM, dm.HEALTHY]
SUBSETS = [
    ("A_raw_dev", [0, 1]),
    ("B_local_stats", [0, 1, 2, 3, 4]),
    ("C_coherence", [0, 1, 2, 3, 4, 5]),
    ("D_location", [0, 1, 2, 3, 4, 5, 6, 7]),
    ("E_normals", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("F_full12d", list(range(12))),
]


def _micro(mask, gt):
    tp = int((mask & gt).sum()); fp = int((mask & ~gt).sum()); fn = int((~mask & gt).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=120)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--split-seed", type=int, default=2024)
    ap.add_argument("--n-est", type=int, default=200)
    ap.add_argument("--train-pts-cap", type=int, default=10000)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = GenConfig(num_scenes=args.num, points=args.points, n_ref=args.n_ref,
                    seed=args.seed, out_dir="data")
    rng = np.random.default_rng(cfg.seed)
    rng_ref = np.random.default_rng(cfg.seed + 1000)
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    nominal, nominal_nrm = build_reference(V, F, N, cfg.n_ref, rng_ref)
    tree = cKDTree(nominal)

    t0 = time.time()
    scenes = []
    for i in range(args.num):
        disp, lab, meta = dm.generate_damage(V, N, cfg, rng)
        scen = to_scan(V, F, N, disp, lab, cfg, rng, nominal, nominal_nrm)
        X, y, dev_mm, aligned = extract_scene_features(scen, nominal, nominal_nrm,
                                                       tree)
        scenes.append(dict(X=X, y=y, dev_mm=dev_mm, type=meta["type"]))
    print("scenes ready: %.1fs" % (time.time() - t0))

    types = np.array([s["type"] for s in scenes])
    rsplit = np.random.default_rng(args.split_seed)
    is_train = np.zeros(args.num, dtype=bool)
    for t in TYPE_ORDER:
        idx = np.where(types == t)[0]
        rsplit.shuffle(idx)
        k = max(2, int(round(len(idx) * args.train_frac)))
        is_train[idx[:k]] = True
    tr, te = np.where(is_train)[0], np.where(~is_train)[0]

    rsub = np.random.default_rng(args.split_seed + 1)
    rows = []
    for name, cols in SUBSETS:
        parts, yparts = [], []
        for i in tr:
            s = scenes[i]; n = len(s["y"])
            sel = (rsub.choice(n, size=min(n, args.train_pts_cap), replace=False)
                   if n > args.train_pts_cap else np.arange(n))
            parts.append(s["X"][sel][:, cols]); yparts.append(s["y"][sel])
        Xtr = np.concatenate(parts); ytr = np.concatenate(yparts)
        model = make_model("rf", args.n_est)
        t1 = time.time()
        model.fit(Xtr, ytr)
        ncap = min(len(ytr), 200000)
        sel = rsub.choice(len(ytr), ncap, replace=False)
        p_tr = model.predict_proba(Xtr[sel])[:, 1]
        thr = max(np.linspace(0.02, 0.98, 97),
                  key=lambda t: _micro(p_tr > t, ytr[sel])[2])
        for t in TYPE_ORDER + [-1]:
            if t == -1:
                idxs = te; tname = "MICRO"
            else:
                idxs = [i for i in te if scenes[i]["type"] == t]
                tname = dm.TYPE_NAMES[t]
            Xte = np.concatenate([scenes[i]["X"][:, cols] for i in idxs])
            yte = np.concatenate([scenes[i]["y"] for i in idxs])
            p, r, f1 = _micro(model.predict_proba(Xte)[:, 1] > thr, yte)
            rows.append(dict(subset=name, n_feats=len(cols), type=t,
                             type_name=tname, n_scenes=len(idxs),
                             precision=p, recall=r, f1=f1, thr=thr,
                             fit_s=float(time.time() - t1)))
        micro = [r for r in rows if r["type_name"] == "MICRO"][-1]
        print("  %-14s (k=%2d, thr=%.3f)  MICRO F1=%.3f  P=%.3f R=%.3f" % (
            name, len(cols), thr, micro["f1"], micro["precision"], micro["recall"]))

    csv_path = os.path.join(args.out, "ablation_%d.csv" % args.num)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved: %s" % csv_path)


if __name__ == "__main__":
    main()