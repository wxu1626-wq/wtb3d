"""Paper figures for the 3D WTB damage-detection benchmark.

Model-free figures (run without a trained model):
  fig1_damage_gallery.png       4 damage types, scan + GT side views (full + zoom)
  fig2_deviation_field.png      dent scene: ICP signed-deviation field in mm
  fig4_scene_quantification.png scene-level area/volume/localization errors
  fig7_tau_sweep.png            baseline tau sweep P/R/F1

Model-dependent figures (needs results/learned_1000_rf_model.joblib):
  fig3_detection_comparison.png per type: GT mask vs baseline mask vs RF mask
  fig5_fp_suppression.png       healthy scenes: FP spatial maps baseline vs RF
  fig6_feature_importance.png   RF feature importances

Usage:
  python paper/make_figs.py
  python paper/make_figs.py --model results/learned_1000_rf_model.joblib

Scene regeneration is deterministic (seed 0 protocol identical to run_learned.py).
"""
import os, sys, csv, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
from learned.features import (deviation_field, extract_scene_features,
                              FEAT_NAMES)

TYPES = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM]
GRAY = "#c9c9c9"; DAM = "#d62728"; BASE = "#1f77b4"; RF = "#2ca02c"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 150})


def gen_scenes(num, points, n_ref, seed):
    """Regenerate scenes with the exact run_learned.py protocol."""
    cfg = GenConfig(num_scenes=num, points=points, n_ref=n_ref, seed=seed,
                    out_dir="data")
    rng = np.random.default_rng(cfg.seed)
    rng_ref = np.random.default_rng(cfg.seed + 1000)
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    nominal, nominal_nrm = build_reference(V, F, N, cfg.n_ref, rng_ref)
    tree = cKDTree(nominal)
    t0 = time.time()
    scenes = []
    for i in range(num):
        disp, lab, meta = dm.generate_damage(V, N, cfg, rng)
        scen = to_scan(V, F, N, disp, lab, cfg, rng, nominal, nominal_nrm)
        scenes.append(dict(scen=scen, meta=meta, type=meta["type"], idx=i))
        if (i + 1) % 50 == 0:
            print("  %d/%d scenes, %.1fs" % (i + 1, num, time.time() - t0))
    ctx = dict(V=V, F=F, N=N, nominal=nominal, nominal_nrm=nominal_nrm, tree=tree)
    return ctx, scenes


def _side(ax, pts, mask=None, col=GRAY, s=1.0):
    m = np.ones(len(pts), dtype=bool) if mask is None else mask
    ax.scatter(pts[m, 0], pts[m, 2], s=s, c=col, linewidths=0)


def fig1_gallery(ctx, scenes, outdir):
    first = {}
    for s in scenes:
        first.setdefault(s["type"], s)
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0))
    for j, t in enumerate(TYPES):
        s = first[t]; pts = s["scen"]["scanned"]; gt = s["scen"]["gt_mask"]
        ax = axes[0, j]
        _side(ax, pts, ~gt); _side(ax, pts, gt, DAM, 1.8)
        ax.set_title(dm.TYPE_NAMES[t])
        if gt.any():
            x0, x1 = pts[gt, 0].min() - 0.05, pts[gt, 0].max() + 0.05
            z0, z1 = pts[gt, 2].min() - 0.15, pts[gt, 2].max() + 0.15
            axz = axes[1, j]
            axz.set_xlim(x0, x1); axz.set_ylim(z0, z1)
            _side(axz, pts, ~gt); _side(axz, pts, gt, DAM, 1.8)
            axz.set_title(dm.TYPE_NAMES[t] + " (zoom)")
        else:
            axes[1, j].axis("off")
    for row in axes:
        for ax in row:
            ax.set_xlabel("chord (m)"); ax.set_ylabel("span (m)")
    fig.suptitle("Synthetic 3D damage gallery (scan gray, ground-truth damage red)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "fig1_damage_gallery.png")); plt.close(fig)
    print("saved fig1_damage_gallery.png")


def fig2_deviation(ctx, scenes, outdir):
    nominal, nominal_nrm, tree = ctx["nominal"], ctx["nominal_nrm"], ctx["tree"]
    # pick the deepest dent among the first 12 candidates (visibility)
    cands = [x for x in scenes if x["type"] == dm.DENT][:12]
    best, bestscore = cands[0], -1.0
    for x in cands:
        _a, _d = deviation_field(x["scen"], nominal, nominal_nrm, tree)
        _g = x["scen"]["gt_mask"]
        score = float(np.abs(_d[_g]).mean()) if _g.any() else 0.0
        if score > bestscore:
            best, bestscore = x, score
    s = best
    aligned, dev = deviation_field(s["scen"], nominal, nominal_nrm, tree)
    gt = s["scen"]["gt_mask"]
    print("  fig2 dent scene %d: mean |dev| on GT = %.2f mm" % (
        s["idx"], float(np.abs(dev[gt]).mean())))
    x0, x1 = aligned[:, 0].min(), aligned[:, 0].max()
    z0, z1 = aligned[:, 2].min(), aligned[:, 2].max()
    gx0, gx1 = aligned[gt, 0].min() - 0.06, aligned[gt, 0].max() + 0.06
    gz0, gz1 = aligned[gt, 2].min() - 0.18, aligned[gt, 2].max() + 0.18
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for k, (ax, (lx0, lx1, ly0, ly1)) in enumerate(zip(axes, [
            (x0, x1, z0, z1), (gx0, gx1, gz0, gz1)])):
        sc = ax.scatter(aligned[:, 0], aligned[:, 2],
                        s=(1.0 if k == 0 else 2.5), c=dev,
                        cmap="RdBu_r", vmin=-5, vmax=5, linewidths=0)
        ax.axvspan(x0, x0 + 0.03, color=BASE, alpha=0.10)
        ax.axvspan(x1 - 0.03, x1, color=BASE, alpha=0.10)
        ax.set_xlim(lx0, lx1); ax.set_ylim(ly0, ly1)
        ax.set_xlabel("chord (m)"); ax.set_ylabel("span (m)")
    axes[0].set_title("ICP signed deviation (mm), dent scene %d (clipped +/-5 mm)" % s["idx"])
    axes[0].annotate("LE/TE", xy=(x0 + 0.015, z1 * 0.92),
                     fontsize=8, ha="center", va="top", color=BASE)
    axes[0].annotate("artifacts", xy=(x0 + 0.015, z1 * 0.84),
                     fontsize=8, ha="center", va="top", color=BASE)
    axes[1].set_title("zoom on dent")
    axes[1].annotate("dent", xy=(0.5 * (gx0 + gx1), gz0 + 0.02),
                     fontsize=9, ha="center", va="bottom", color="white")
    fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02,
                 label="deviation (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_deviation_field.png")); plt.close(fig)
    print("saved fig2_deviation_field.png")


def _read_scene_csv(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if "method" in r and r["method"]:
                key = "baseline" if r["method"].startswith("baseline") else "learned"
            else:
                key = "baseline"
            out.setdefault(key, {})[r["type"]] = r
    return out


def fig4_scene_metrics(results, outdir):
    rows = _read_scene_csv(os.path.join(results, "learned_1000_rf_scene.csv"))
    base, rf = rows.get("baseline"), rows.get("learned")
    if base is None or rf is None:
        rows2 = _read_scene_csv(os.path.join(results, "scene_metrics_baseline_1000.csv"))
        base = base if base is not None else rows2.get("baseline")
    if base is None or rf is None:
        print("SKIPPED fig4 (scene-metric CSVs not found)")
        return
    names = [dm.TYPE_NAMES[t] for t in TYPES]
    x = np.arange(4); w = 0.38
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.1))
    panels = [
        ("area_abs_err_m2", "damage area MAE (m$^2$)", False),
        ("vol_mae_cm3", "damage volume MAE (cm$^3$)", True),
        ("centroid_err_m", "damage centroid error (m)", False),
        ("healthy", "healthy-scene false alarm", None),
    ]
    for ax, (key, title, logy) in zip(axes, panels):
        if key == "healthy":
            b = float(base[str(dm.HEALTHY)]["area_pred_m2"])
            r = float(rf[str(dm.HEALTHY)]["area_pred_m2"])
            bv = float(base[str(dm.HEALTHY)]["vol_pred_cm3"])
            rv = float(rf[str(dm.HEALTHY)]["vol_pred_cm3"])
            ax.bar(x[:1] - w / 2, [b], w, label="ICP + tau=2.0mm", color=BASE)
            ax.bar(x[:1] + w / 2, [r], w, label="RF (context features)", color=RF)
            ax.set_yscale("log"); ax.set_ylim(0.001, 10.0)
            ax.text(0 - w / 2, b * 1.4, "%.3f m2\n(%.0f cm3)" % (b, bv),
                    ha="center", fontsize=8)
            ax.text(0 + w / 2, max(r * 1.4, 0.002), "%.3f m2\n(%.0f cm3)" % (r, rv),
                    ha="center", fontsize=8)
        else:
            b = [float(base[str(t)][key]) for t in TYPES]
            r = [float(rf[str(t)][key]) for t in TYPES]
            ax.bar(x - w / 2, b, w, label="ICP + tau=2.0mm", color=BASE)
            ax.bar(x + w / 2, r, w, label="RF (context features)", color=RF)
            if logy:
                ax.set_yscale("log")
        ax.set_title(title)
        if key == "healthy":
            ax.set_xticks([-w / 2, w / 2]); ax.set_xticklabels(["ICP+tau", "RF"])
        else:
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
    axes[0].legend(fontsize=8)
    fig.suptitle("Scene-level quantification error (199 test scenes)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "fig4_scene_quantification.png")); plt.close(fig)
    print("saved fig4_scene_quantification.png")


def fig7_tau(results, outdir):
    path = os.path.join(results, "tau_sweep.csv")
    if not os.path.exists(path):
        print("SKIPPED fig7 (tau_sweep.csv not found)")
        return
    taus, ps, rs, fs = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            taus.append(float(r["tau_mm"])); ps.append(float(r["precision"]))
            rs.append(float(r["recall"])); fs.append(float(r["f1"]))
    taus = np.array(taus)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.semilogx(taus, ps, "o-", label="precision")
    ax.semilogx(taus, rs, "s-", label="recall")
    ax.semilogx(taus, fs, "^-", label="F1")
    ax.axvline(2.0, color="k", ls=":", lw=1)
    ax.annotate("tau = 2.0 mm (train-tuned)", xy=(2.0, 0.12), xytext=(2.8, 0.32),
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("tau (mm)"); ax.set_ylabel("score")
    ax.set_title("Baseline global-threshold sweep (test set)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig7_tau_sweep.png")); plt.close(fig)
    print("saved fig7_tau_sweep.png")


def _load_model(path):
    import joblib
    obj = joblib.load(path)
    return obj["model"], float(obj["thr"])


def _first_by_type(scenes):
    first = {}
    for s in scenes:
        first.setdefault(s["type"], s)
    return first


def fig3_detection(ctx, scenes, model, thr, tau, outdir):
    nominal, nominal_nrm, tree = ctx["nominal"], ctx["nominal_nrm"], ctx["tree"]
    first = _first_by_type(scenes)
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 12.0))
    for i, t in enumerate(TYPES):
        s = first[t]
        X, y, dev, aligned = extract_scene_features(s["scen"], nominal, nominal_nrm, tree)
        preds = [("ground truth", y, DAM),
                 ("ICP + tau=%.0f mm" % tau, np.abs(dev) > tau, BASE),
                 ("RF mask", model.predict_proba(X)[:, 1] > thr, RF)]
        for j, (name, pred, col) in enumerate(preds):
            ax = axes[i, j]
            _side(ax, aligned, ~y, GRAY, 0.8)
            _side(ax, aligned, pred, col, 1.3)
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel("%s\nscene %d" % (dm.TYPE_NAMES[t], s["idx"]),
                              fontsize=9)
            if name != "ground truth":
                ax.text(0.02, 0.97, "FP=%d" % int((pred & ~y).sum()),
                        transform=ax.transAxes, va="top", fontsize=8)
    fig.suptitle("Detection masks per damage type (one representative scene each)")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(outdir, "fig3_detection_comparison.png")); plt.close(fig)
    print("saved fig3_detection_comparison.png")


def fig5_fp(ctx, scenes, model, thr, tau, outdir, n_health=4):
    nominal, nominal_nrm, tree = ctx["nominal"], ctx["nominal_nrm"], ctx["tree"]
    health = [s for s in scenes if s["type"] == dm.HEALTHY][:n_health]
    fig, axes = plt.subplots(2, n_health, figsize=(3.1 * n_health, 5.6))
    for i, s in enumerate(health):
        X, y, dev, aligned = extract_scene_features(s["scen"], nominal, nominal_nrm, tree)
        preds = [("ICP + tau=%.0f mm" % tau, np.abs(dev) > tau, BASE, 0),
                 ("RF mask", model.predict_proba(X)[:, 1] > thr, RF, 1)]
        for name, pred, col, row in preds:
            ax = axes[row, i]
            _side(ax, aligned, None, GRAY, 0.7)
            _side(ax, aligned, pred, col, 1.4)
            if i == 0:
                ax.set_ylabel(name, fontsize=9)
            ax.set_title("scene %d: FP=%d" % (s["idx"], int(pred.sum())),
                         fontsize=9)
    fig.suptitle("Healthy scenes: false-positive spatial maps")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "fig5_fp_suppression.png")); plt.close(fig)
    print("saved fig5_fp_suppression.png")


def fig6_importance(model, outdir):
    imp = model.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.barh([FEAT_NAMES[i] for i in order], imp[order], color="#4c72b0")
    ax.set_xlabel("Gini importance")
    ax.set_title("RF feature importances (12-d context features)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_feature_importance.png")); plt.close(fig)
    print("saved fig6_feature_importance.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=240)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str,
                    default="results/learned_1000_rf_model.joblib")
    ap.add_argument("--config", type=str,
                    default="results/learned_1000_rf_config.json")
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--out", type=str, default="paper/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("generating %d scenes (seed=%d) ..." % (args.num, args.seed))
    t0 = time.time()
    ctx, scenes = gen_scenes(args.num, args.points, args.n_ref, args.seed)
    print("scenes ready in %.1fs" % (time.time() - t0))

    fig1_gallery(ctx, scenes, args.out)
    fig2_deviation(ctx, scenes, args.out)
    fig4_scene_metrics(args.results, args.out)
    fig7_tau(args.results, args.out)

    if os.path.exists(args.model):
        model, thr = _load_model(args.model)
        tau = 2.0
        if os.path.exists(args.config):
            with open(args.config) as f:
                tau = float(json.load(f).get("best_tau_mm", 2.0))
        print("model loaded: thr=%.3f tau=%.1f" % (thr, tau))
        fig3_detection(ctx, scenes, model, thr, tau, args.out)
        fig5_fp(ctx, scenes, model, thr, tau, args.out)
        fig6_importance(model, args.out)
    else:
        print("model not found at %s; skipping fig3/fig5/fig6 (re-run after RF job)"
              % args.model)


if __name__ == "__main__":
    main()