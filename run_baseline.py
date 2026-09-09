"""Run the classical ICP -> signed-distance -> threshold baseline over many scenes.

Model: one dense undamaged CAD reference (built once, KD-tree built once) vs many
sparse registered scans of damaged blades. ICP recovers each scan's residual pose,
then point-to-plane signed deviation from the CAD localizes + quantifies damage.

Usage:
  python run_baseline.py --num 1000 --points 5000
Outputs:
  results/baseline_<num>.csv          one row per (scene, tau)
  results/baseline_<num>_summary.csv  per-type aggregation at the best tau
  results/tau_sweep.csv               micro-F1 vs tau
  results/config.json                 run settings
  paper/figs/*.png                    deviation map + per-type F1 bars
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
from baselines.classical_pipeline import (HAVE_O3D, icp_pp,
                                          point_to_plane_distance, smooth_dev)
from baselines.evaluate import prf, geo_err

TYPE_ORDER = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM, dm.HEALTHY]


def _micro_f1(records, tau):
    tp = sum(r["tp"] for r in records if r["tau_mm"] == tau)
    fp = sum(r["fp"] for r in records if r["tau_mm"] == tau)
    fn = sum(r["fn"] for r in records if r["tau_mm"] == tau)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=str, default="0.5,1,2,3,5,8,12,20")
    ap.add_argument("--viz-n", type=int, default=4)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--figs", type=str, default="paper/figs")
    args = ap.parse_args()
    taus = [float(x) for x in args.tau.split(",")]

    cfg = GenConfig(num_scenes=args.num, points=args.points, n_ref=args.n_ref,
                    seed=args.seed, out_dir="data")
    rng = np.random.default_rng(cfg.seed)
    rng_ref = np.random.default_rng(cfg.seed + 1000)
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    nominal, nominal_nrm = build_reference(V, F, N, cfg.n_ref, rng_ref)
    tree = cKDTree(nominal)

    os.makedirs(args.out, exist_ok=True); os.makedirs(args.figs, exist_ok=True)
    records = []
    viz = []
    hp_p95_list = []
    t0 = time.time()
    for i in range(args.num):
        disp, lab, meta = dm.generate_damage(V, N, cfg, rng)
        scen = to_scan(V, F, N, disp, lab, cfg, rng, nominal, nominal_nrm)
        G, tG, cost = icp_pp(scen["scanned"], nominal, tree=tree)
        aligned = scen["scanned"] @ G + tG
        signed = point_to_plane_distance(aligned, nominal, nominal_nrm, tree=tree)
        dev_s = smooth_dev(aligned, signed, k=3)
        hpm = ~scen["gt_mask"]
        hp_p95_list.append(float(np.percentile(np.abs(dev_s[hpm]) * 1000, 95)) if hpm.any() else 0.0)
        if i < args.viz_n:
            viz.append((i, meta, scen, dev_s))
        for tau in taus:
            mask = np.abs(dev_s) > (tau * 1e-3)
            row = prf(mask, scen["gt_mask"])
            row.update(geo_err(dev_s, scen["gt_depth_mm"], scen["gt_mask"]))
            row.update(scene=i, type=meta["type"], type_name=meta["type_name"],
                       icp_cost_mm=cost * 1000.0, n=int(len(scen["scanned"])),
                       tau_mm=tau)
            records.append(row)
        if (i + 1) % 100 == 0:
            print("  %d/%d scenes, %.1fs" % (i + 1, args.num, time.time() - t0))

    sweep = []
    best_tau, best_f1 = taus[0], -1.0
    for tau in taus:
        p, r, f1, tp, fp, fn = _micro_f1(records, tau)
        sweep.append(dict(tau_mm=tau, precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn))
        if f1 > best_f1:
            best_f1, best_tau = f1, tau

    summary = []
    for t in TYPE_ORDER:
        sub = [r for r in records if r["tau_mm"] == best_tau and r["type"] == t]
        if not sub:
            continue
        tp = sum(x["tp"] for x in sub); fp = sum(x["fp"] for x in sub); fn = sum(x["fn"] for x in sub)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        maes = [x["mae"] for x in sub if x["mae"] == x["mae"]]
        rmses = [x["rmse"] for x in sub if x["rmse"] == x["rmse"]]
        summary.append(dict(type=t, type_name=dm.TYPE_NAMES[t], n_scenes=len(sub),
                            precision=prec, recall=rec, f1=f1, iou=iou,
                            geo_mae_mm=(float(np.mean(maes)) if maes else np.nan),
                            geo_rmse_mm=(float(np.mean(rmses)) if rmses else np.nan)))

    csv_path = os.path.join(args.out, "baseline_%d.csv" % args.num)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys())); w.writeheader()
        for r in records:
            w.writerow(r)
    sum_path = os.path.join(args.out, "baseline_%d_summary.csv" % args.num)
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader()
        for r in summary:
            w.writerow(r)
    with open(os.path.join(args.out, "tau_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys())); w.writeheader()
        for r in sweep:
            w.writerow(r)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"n_scenes": args.num, "points": args.points, "n_ref": args.n_ref,
                   "seed": args.seed, "taus": taus, "best_tau_mm": best_tau,
                   "best_micro_f1": best_f1, "have_open3d": HAVE_O3D}, f, indent=2)

    _plot_viz(viz, os.path.join(args.figs, "deviation_map.png"))
    _plot_summary(summary, best_tau, os.path.join(args.figs, "f1_by_type.png"))

    print("")
    print("=== CLASSICAL BASELINE (%d scenes, %d ref pts) ===" % (args.num, args.n_ref))
    print("HAVE_OPEN3D=%s  best_tau=%.1fmm  micro_F1=%.3f" % (HAVE_O3D, best_tau, best_f1))
    for s in summary:
        gmae = s["geo_mae_mm"]; grmse = s["geo_rmse_mm"]
        gmae = gmae if gmae == gmae else float("nan")
        grmse = grmse if grmse == grmse else float("nan")
        print("  %-13s n=%3d  P=%.3f R=%.3f F1=%.3f IoU=%.3f  MAE=%5.2f RMSE=%5.2f mm" % (
            s["type_name"], s["n_scenes"], s["precision"], s["recall"], s["f1"],
            s["iou"], gmae, grmse))
    hp = np.array(hp_p95_list)
    print("Healthy-region |dev| (alignment+noise floor), per-scene p95:  mean=%.2fmm  p95=%.2fmm  max=%.2fmm" % (
        float(np.mean(hp)), float(np.percentile(hp, 95)), float(np.max(hp))))
    icp_costs = [r["icp_cost_mm"] for r in records if r["tau_mm"] == taus[0]]
    print("ICP point-to-point NN objective (sampling-limited):  mean=%.2fmm" % float(np.mean(icp_costs)))
    print("saved: %s ; %s ; %s" % (csv_path, sum_path, os.path.join(args.out, "tau_sweep.csv")))


def _plot_viz(viz, path):
    if not viz:
        return
    idx, meta, scen, dev_s = viz[0]
    pts = scen["scanned"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 2], c="#cccccc", s=2, label="scan")
    dmm = np.abs(dev_s) * 1000.0
    sc = ax.scatter(pts[:, 0][dmm > 1.0], pts[:, 2][dmm > 1.0],
                    c=dmm[dmm > 1.0], cmap="magma", s=4)
    ax.set_title("Scan + deviation > 1mm  (%s)" % meta["type_name"])
    ax.set_xlabel("chord (m)"); ax.set_ylabel("span (m)"); ax.legend(loc="upper right")
    fig.colorbar(sc, ax=ax, label="|deviation| (mm)")
    ax = axes[1]
    sc2 = ax.scatter(pts[:, 0], pts[:, 2], c=dev_s * 1000.0, cmap="coolwarm", s=3, vmin=-12, vmax=12)
    ax.set_title("Signed deviation map (mm)")
    ax.set_xlabel("chord (m)"); ax.set_ylabel("span (m)")
    fig.colorbar(sc2, ax=ax, label="signed dev (mm)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_summary(summary, best_tau, path):
    names = [s["type_name"] for s in summary]
    f1 = [s["f1"] for s in summary]
    prec = [s["precision"] for s in summary]
    rec = [s["recall"] for s in summary]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w, prec, w, label="precision")
    ax.bar(x, rec, w, label="recall")
    ax.bar(x + w, f1, w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.05)
    ax.set_title("Point-level detection by damage type (tau=%.1fmm)" % best_tau)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()