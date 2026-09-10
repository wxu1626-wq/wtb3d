"""Cross-sensor-setting robustness of the learned detector (CPU).

Honest protocol:
  * ONE Random Forest is trained on NOMINAL-sensor scenes (train split only)
  * the same damage realizations (same seed -> same rng draws for damage)
    are re-scanned under 8 sensor settings, one perturbation at a time
  * baseline tau and the RF decision threshold are both tuned on the
    nominal TRAIN split and frozen
  * metrics (micro + per-type F1) are reported on the TEST split of the
    perturbed scans -> measures generalization to unseen sensor conditions

Settings (nominal: noise 0.2 mm, dropout 0.15, angle 0.3 deg, trans 5 mm):
  nominal / noise0.5 / noise1.0 / dropout0.3 / dropout0.5 /
  angle1.0 / angle2.0 / trans15

Usage:
  python run_robustness.py --num 200 --points 20000
Outputs:
  results/robustness_<num>.csv
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
from learned.features import extract_scene_features
from learned.models import make_model

TYPE_ORDER = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM, dm.HEALTHY]

SETTINGS = [
    ("nominal",  {}),
    ("noise0.5", dict(noise_mm=0.5)),
    ("noise1.0", dict(noise_mm=1.0)),
    ("dropout0.3", dict(dropout=0.3)),
    ("dropout0.5", dict(dropout=0.5)),
    ("angle1.0", dict(max_angle_deg=1.0)),
    ("angle2.0", dict(max_angle_deg=2.0)),
    ("trans15",  dict(max_trans_mm=15.0)),
]


def _micro(mask, gt):
    tp = int((mask & gt).sum()); fp = int((mask & ~gt).sum()); fn = int((~mask & gt).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--split-seed", type=int, default=2024)
    ap.add_argument("--n-est", type=int, default=300)
    ap.add_argument("--train-pts-cap", type=int, default=10000)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse saved RF model from a previous run (same seed)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from dataclasses import replace

    cfg = GenConfig(num_scenes=args.num, points=args.points, n_ref=args.n_ref,
                    seed=args.seed, out_dir="data")
    rng_ref = np.random.default_rng(cfg.seed + 1000)
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    nominal, nominal_nrm = build_reference(V, F, N, cfg.n_ref, rng_ref)
    tree = cKDTree(nominal)

    def gen(setting):
        """Same damage realizations under one sensor setting."""
        cfg2 = replace(cfg, **setting)
        rng = np.random.default_rng(cfg.seed)
        out = []
        for i in range(args.num):
            disp, lab, meta = dm.generate_damage(V, N, cfg2, rng)
            scen = to_scan(V, F, N, disp, lab, cfg2, rng, nominal, nominal_nrm)
            X, y, dev_mm, aligned = extract_scene_features(
                scen, nominal, nominal_nrm, tree)
            out.append(dict(X=X, y=y, dev_mm=dev_mm, type=meta["type"]))
        return out

    # ---- nominal scenes: split, tune tau + train RF ----
    t0 = time.time()
    nom = gen({})
    print("nominal scenes ready: %.1fs" % (time.time() - t0))
    types = np.array([s["type"] for s in nom])
    rsplit = np.random.default_rng(args.split_seed)
    is_train = np.zeros(args.num, dtype=bool)
    for t in TYPE_ORDER:
        idx = np.where(types == t)[0]
        rsplit.shuffle(idx)
        k = max(2, int(round(len(idx) * args.train_frac)))
        is_train[idx[:k]] = True
    tr, te = np.where(is_train)[0], np.where(~is_train)[0]

    rsub = np.random.default_rng(args.split_seed + 1)
    parts, yparts = [], []
    for i in tr:
        s = nom[i]; n = len(s["y"])
        sel = (rsub.choice(n, size=min(n, args.train_pts_cap), replace=False)
               if n > args.train_pts_cap else np.arange(n))
        parts.append(s["X"][sel]); yparts.append(s["y"][sel])
    Xtr = np.concatenate(parts); ytr = np.concatenate(yparts)

    dev_tr = np.concatenate([nom[i]["dev_mm"] for i in tr])
    y_tr_full = np.concatenate([nom[i]["y"] for i in tr])
    best_tau = args.tau
    for tau in [0.5, 1, 2, 3, 5, 8, 12, 20]:
        p, r, f1 = _micro(np.abs(dev_tr) > tau, y_tr_full)
        if f1 > _micro(np.abs(dev_tr) > best_tau, y_tr_full)[2]:
            best_tau = tau
    print("tau (train-tuned on nominal): %.1f mm" % best_tau)

    import joblib
    ckpt = os.path.join(args.out, "robustness_rf_%d.joblib" % args.num)
    if args.reuse and os.path.exists(ckpt):
        obj = joblib.load(ckpt)
        model, best_thr = obj["model"], float(obj["thr"])
        print("reused model %s (thr=%.3f)" % (ckpt, best_thr))
    else:
        model = make_model("rf", args.n_est)
        t1 = time.time()
        model.fit(Xtr, ytr)
        print("RF fit: %.1fs" % (time.time() - t1))
        ncap = min(len(ytr), 200000)
        sel = rsub.choice(len(ytr), ncap, replace=False)
        p_tr = model.predict_proba(Xtr[sel])[:, 1]
        best_thr = max(np.linspace(0.02, 0.98, 97),
                       key=lambda t: _micro(p_tr > t, ytr[sel])[2])
        print("RF thr (train-tuned on nominal): %.3f" % best_thr)
        joblib.dump({"model": model, "thr": best_thr}, ckpt)
        print("model saved: %s" % ckpt)

    # ---- evaluate each setting on the TEST split ----
    rows = []
    for name, setting in SETTINGS:
        scenes = gen(setting)
        METHODS = ("baseline_tau%.1f" % best_tau, "rf_thr%.3f" % best_thr)
        agg = {t: {m: dict(tp=0, fp=0, fn=0) for m in METHODS} for t in TYPE_ORDER}
        tot = {m: dict(tp=0, fp=0, fn=0) for m in METHODS}
        for i in te:
            s = scenes[i]
            for method, mask in [(m,
                     (np.abs(s["dev_mm"]) > best_tau)
                     if m.startswith("baseline") else
                     (model.predict_proba(s["X"])[:, 1] > best_thr))
                    for m in METHODS]:
                m = mask; y = s["y"]
                tp = int((m & y).sum()); fp = int((m & ~y).sum())
                fn = int((~m & y).sum())
                agg[s["type"]][method]["tp"] += tp
                agg[s["type"]][method]["fp"] += fp
                agg[s["type"]][method]["fn"] += fn
                for k in ("tp", "fp", "fn"):
                    tot[method][k] += dict(tp=tp, fp=fp, fn=fn)[k]
        for t in TYPE_ORDER:
            for method in ("baseline_tau%.1f" % best_tau,
                           "rf_thr%.3f" % best_thr):
                a = agg[t][method]
                p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 0.0
                r = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) else 0.0
                rows.append(dict(setting=name, method=method, type=t,
                                 type_name=dm.TYPE_NAMES[t],
                                 n_scenes=int((types[te] == t).sum()),
                                 precision=p, recall=r, f1=f1))
        for method in ("baseline_tau%.1f" % best_tau,
                       "rf_thr%.3f" % best_thr):
            a = tot[method]
            p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 0.0
            r = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            rows.append(dict(setting=name, method=method, type=-1,
                             type_name="MICRO", n_scenes=len(te),
                             precision=p, recall=r, f1=f1))
        bm = [r for r in rows if r["setting"] == name and
              r["type_name"] == "MICRO" and r["method"].startswith("baseline")][0]
        rf = [r for r in rows if r["setting"] == name and
              r["type_name"] == "MICRO" and r["method"].startswith("rf")][0]
        print("  %-11s baseline microF1=%.3f | RF microF1=%.3f" % (
            name, bm["f1"], rf["f1"]))

    csv_path = os.path.join(args.out, "robustness_%d.csv" % args.num)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved: %s" % csv_path)


if __name__ == "__main__":
    main()