"""Learned (data-driven) point-level damage detection, head-to-head with the
classical ICP -> signed-deviation -> global-threshold baseline.

Fairness protocol:
  * identical scan generation, identical deviation field (ICP + noise floor)
  * scene-level stratified train/test split (by damage type)
  * baseline tau AND learned decision threshold tuned on TRAIN only
  * both frozen, evaluated on the held-out TEST set (all points)

Usage:
  python run_learned.py --num 1000 --points 20000 --model rf --n-est 300
Outputs:
  results/learned_<num>_comparison.csv   per-type P/R/F1/IoU, baseline vs learned
  results/learned_<num>_config.json
  paper/figs/learned_vs_baseline.png
"""
import os, csv, json, argparse, time
import numpy as np
from scipy.spatial import cKDTree

_DEFAULT_MPL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "work", "mpl"))
os.environ.setdefault("MPLCONFIGDIR", _DEFAULT_MPL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_gen.config import GenConfig
from data_gen.geometry import loft_blade
from data_gen import damage as dm
from data_gen.sensor_sim import to_scan, build_reference
from learned.features import extract_scene_features, FEAT_NAMES
from learned.models import make_model
from baselines.evaluate import prf, geo_err

TYPE_ORDER = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM, dm.HEALTHY]


def _micro(mask, gt):
    tp = int((mask & gt).sum()); fp = int((mask & ~gt).sum()); fn = int((~mask & gt).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return p, r, f1, iou, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--split-seed", type=int, default=2024)
    ap.add_argument("--model", type=str, default="rf", choices=["rf", "gbm", "mlp"])
    ap.add_argument("--n-est", type=int, default=300)
    ap.add_argument("--train-pts-cap", type=int, default=10000)
    ap.add_argument("--k-local", type=int, default=12)
    ap.add_argument("--taus", type=str, default="0.5,1,2,3,5,8,12,20")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--figs", type=str, default="paper/figs")
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",")]

    cfg = GenConfig(num_scenes=args.num, points=args.points, n_ref=args.n_ref,
                    seed=args.seed, out_dir="data")
    rng = np.random.default_rng(cfg.seed)
    rng_ref = np.random.default_rng(cfg.seed + 1000)
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    nominal, nominal_nrm = build_reference(V, F, N, cfg.n_ref, rng_ref)
    tree = cKDTree(nominal)

    os.makedirs(args.out, exist_ok=True); os.makedirs(args.figs, exist_ok=True)
    t0 = time.time()
    scenes = []
    for i in range(args.num):
        disp, lab, meta = dm.generate_damage(V, N, cfg, rng)
        scen = to_scan(V, F, N, disp, lab, cfg, rng, nominal, nominal_nrm)
        X, y, dev_mm = extract_scene_features(scen, nominal, nominal_nrm, tree,
                                              k_local=args.k_local)
        scenes.append(dict(X=X, y=y, dev_mm=dev_mm,
                           gt_depth_mm=scen["gt_depth_mm"].astype(np.float32),
                           type=meta["type"]))
        if (i + 1) % 100 == 0:
            print("  %d/%d scenes, %.1fs" % (i + 1, args.num, time.time() - t0))

    # ---- stratified scene-level split ----
    rsplit = np.random.default_rng(args.split_seed)
    types = np.array([s["type"] for s in scenes])
    is_train = np.zeros(args.num, dtype=bool)
    for t in TYPE_ORDER:
        idx = np.where(types == t)[0]
        rsplit.shuffle(idx)
        k = max(2, int(round(len(idx) * args.train_frac)))
        is_train[idx[:k]] = True
    tr = [i for i in range(args.num) if is_train[i]]
    te = [i for i in range(args.num) if not is_train[i]]
    print("split: %d train / %d test scenes (per-type: %s)" % (
        len(tr), len(te),
        {dm.TYPE_NAMES[t]: int((types == t).sum()) for t in TYPE_ORDER}))

    # ---- training matrix (capped per scene for RAM/time) ----
    rsub = np.random.default_rng(args.split_seed + 1)
    Xtr_parts, ytr_parts, devtr_parts = [], [], []
    for i in tr:
        s = scenes[i]
        n = len(s["y"])
        sel = (rsub.choice(n, size=min(n, args.train_pts_cap), replace=False)
               if n > args.train_pts_cap else np.arange(n))
        Xtr_parts.append(s["X"][sel]); ytr_parts.append(s["y"][sel])
        devtr_parts.append(s["dev_mm"][sel])
    Xtr = np.concatenate(Xtr_parts); ytr = np.concatenate(ytr_parts)
    devtr = np.concatenate(devtr_parts)
    print("train points: %d (pos rate %.4f)" % (len(ytr), ytr.mean()))

    # ---- tune baseline tau on TRAIN only ----
    best_tau, best_f1_tr = taus[0], -1.0
    for tau in taus:
        p, r, f1, iou, tp, fp, fn = _micro(np.abs(devtr) > tau, ytr)
        if f1 > best_f1_tr:
            best_f1_tr, best_tau = f1, tau
    print("baseline best tau (train): %.1f mm  (train micro-F1 %.3f)" % (best_tau, best_f1_tr))

    # ---- fit learned model + tune decision threshold on TRAIN ----
    model = make_model(args.model, args.n_est)
    t1 = time.time()
    model.fit(Xtr, ytr)
    print("fit %s: %.1fs" % (args.model, time.time() - t1))
    ncap = min(len(ytr), 200000)
    sel = rsub.choice(len(ytr), ncap, replace=False)
    p_tr = model.predict_proba(Xtr[sel])[:, 1]
    best_thr, best_thr_f1 = 0.5, -1.0
    for thr in np.linspace(0.02, 0.98, 97):
        p, r, f1, iou, tp, fp, fn = _micro(p_tr > thr, ytr[sel])
        if f1 > best_thr_f1:
            best_thr_f1, best_thr = f1, float(thr)
    print("learned decision threshold (train): %.3f  (train micro-F1 %.3f)" % (best_thr, best_thr_f1))

    # ---- evaluate both on TEST (all points) ----
    rows = []
    for method, pred_fn in [
        ("baseline_tau%.1f" % best_tau, lambda s: np.abs(s["dev_mm"]) > best_tau),
        ("learned_%s" % args.model, None),
    ]:
        agg = {t: dict(tp=0, fp=0, fn=0) for t in TYPE_ORDER}
        geo = {t: [] for t in TYPE_ORDER}
        tot = dict(tp=0, fp=0, fn=0)
        for i in te:
            s = scenes[i]
            mask = pred_fn(s) if pred_fn is not None else model.predict_proba(s["X"])[:, 1] > best_thr
            res = prf(mask, s["y"])
            t = s["type"]
            agg[t]["tp"] += res["tp"]; agg[t]["fp"] += res["fp"]; agg[t]["fn"] += res["fn"]
            tot["tp"] += res["tp"]; tot["fp"] += res["fp"]; tot["fn"] += res["fn"]
            g = geo_err(s["dev_mm"] * 1e-3, s["gt_depth_mm"], s["y"])
            if g["n"]:
                geo[t].append(g)
        for t in TYPE_ORDER:
            a = agg[t]
            p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 0.0
            r = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            iou = a["tp"] / (a["tp"] + a["fp"] + a["fn"]) if (a["tp"] + a["fp"] + a["fn"]) else 0.0
            maes = [x["mae"] for x in geo[t]]; rmses = [x["rmse"] for x in geo[t]]
            rows.append(dict(method=method, type=t, type_name=dm.TYPE_NAMES[t],
                             n_scenes=len([j for j in te if scenes[j]["type"] == t]),
                             precision=p, recall=r, f1=f1, iou=iou,
                             tp=a["tp"], fp=a["fp"], fn=a["fn"],
                             geo_mae_mm=float(np.mean(maes)) if maes else np.nan,
                             geo_rmse_mm=float(np.mean(rmses)) if rmses else np.nan))
        p, r, f1, iou, tp, fp, fn = _micro(
            np.concatenate([model.predict_proba(scenes[i]["X"])[:, 1] > best_thr
                            if method.startswith("learned") else
                            np.abs(scenes[i]["dev_mm"]) > best_tau
                            for i in te]),
            np.concatenate([scenes[i]["y"] for i in te]))
        rows.append(dict(method=method, type=-1, type_name="MICRO",
                         n_scenes=len(te), precision=p, recall=r, f1=f1, iou=iou,
                         tp=tp, fp=fp, fn=fn,
                         geo_mae_mm=np.nan, geo_rmse_mm=np.nan))

    csv_path = os.path.join(args.out, "learned_%d_comparison.csv" % args.num)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.out, "learned_%d_config.json" % args.num), "w") as f:
        json.dump(dict(num=args.num, points=args.points, seed=args.seed,
                       model=args.model, n_est=args.n_est, k_local=args.k_local,
                       train_frac=args.train_frac, split_seed=args.split_seed,
                       best_tau_mm=best_tau, best_thr=best_thr,
                       n_train_pts=int(len(ytr)), feats=FEAT_NAMES), f, indent=2)
    _plot(rows, os.path.join(args.figs, "learned_vs_baseline.png"))

    print("")
    print("=== LEARNED vs BASELINE  (%d test scenes, tau=%.1fmm, thr=%.3f) ===" % (
        len(te), best_tau, best_thr))
    for r in rows:
        if r["type_name"] == "MICRO":
            print("  %-18s MICRO  n=%3d  P=%.3f R=%.3f F1=%.3f IoU=%.3f" % (
                r["method"], r["n_scenes"], r["precision"], r["recall"], r["f1"], r["iou"]))
    for t in TYPE_ORDER:
        b = [r for r in rows if r["type"] == t and r["method"].startswith("baseline")][0]
        l = [r for r in rows if r["type"] == t and r["method"].startswith("learned")][0]
        print("  %-13s baseline F1=%.3f P=%.3f R=%.3f | learned F1=%.3f P=%.3f R=%.3f" % (
            b["type_name"], b["f1"], b["precision"], b["recall"],
            l["f1"], l["precision"], l["recall"]))
    print("saved: %s" % csv_path)


def _plot(rows, path):
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    names = [dm.TYPE_NAMES[t] for t in TYPE_ORDER]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for j, m in enumerate(methods):
        f1 = [dict((r["type"], r) for r in rows if r["method"] == m)[t]["f1"] for t in TYPE_ORDER]
        ax.bar(x + (j - 0.5) * w, f1, w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("point-level F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Learned vs classical threshold (test set)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()