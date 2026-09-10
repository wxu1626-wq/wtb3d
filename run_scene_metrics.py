"""Scene-level engineering metrics for the classical ICP baseline.

Point-level P/R/F1 answers "did the points look right". An inspector also
needs: WHERE is the damage (localization), HOW BIG is it (area, m2), HOW
MUCH material is affected (volume, cm3, shallow-defect approximation).

Usage:
  python run_scene_metrics.py --num 1000 --points 20000 --tau 2.0
Outputs:
  results/scene_metrics_baseline_<num>.csv  per-type area/vol/localization
"""
import os, csv, argparse, time
import numpy as np
from scipy.spatial import cKDTree

_DEFAULT_MPL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "work", "mpl"))
os.environ.setdefault("MPLCONFIGDIR", _DEFAULT_MPL)

from data_gen.config import GenConfig
from data_gen.geometry import loft_blade
from data_gen import damage as dm
from data_gen.sensor_sim import to_scan, build_reference
from learned.features import deviation_field
from baselines.scene_metrics import (mesh_area, scene_damage_metrics,
                                     aggregate_scene_metrics)

TYPE_ORDER = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM, dm.HEALTHY]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    cfg = GenConfig(num_scenes=args.num, points=args.points, n_ref=args.n_ref,
                    seed=args.seed, out_dir="data")
    rng = np.random.default_rng(cfg.seed)
    rng_ref = np.random.default_rng(cfg.seed + 1000)
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    S_mesh = mesh_area(V, F)
    nominal, nominal_nrm = build_reference(V, F, N, cfg.n_ref, rng_ref)
    tree = cKDTree(nominal)
    print("blade mesh area: %.2f m2" % S_mesh)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = []
    types = []
    for i in range(args.num):
        disp, lab, meta = dm.generate_damage(V, N, cfg, rng)
        scen = to_scan(V, F, N, disp, lab, cfg, rng, nominal, nominal_nrm)
        aligned, dev_mm = deviation_field(scen, nominal, nominal_nrm, tree)
        n = len(aligned)
        a_w = S_mesh / n
        mask = np.abs(dev_mm) > args.tau
        rows.append(scene_damage_metrics(mask, scen["gt_mask"].astype(bool),
                                         aligned, dev_mm, scen["gt_depth_mm"], a_w))
        types.append(meta["type"])
        if (i + 1) % 100 == 0:
            print("  %d/%d scenes, %.1fs" % (i + 1, args.num, time.time() - t0))

    agg = aggregate_scene_metrics(rows, types, dm.TYPE_NAMES, TYPE_ORDER)
    path = os.path.join(args.out, "scene_metrics_baseline_%d.csv" % args.num)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader()
        for r in agg:
            w.writerow(r)

    print("")
    print("=== SCENE-LEVEL ENGINEERING METRICS, ICP baseline tau=%.1f mm ===" % args.tau)
    for r in agg:
        print("  %-8s n=%3d | area gt=%.4f pred=%.4f m2 MAE=%.4f | "
              "vol gt=%.2f pred=%.2f cm3 MAE=%.2f | locErr=%.3f m ok=%.0f%%" % (
                  r["type_name"], r["n_scenes"], r["area_gt_m2"], r["area_pred_m2"],
                  r["area_abs_err_m2"], r["vol_gt_cm3"], r["vol_pred_cm3"],
                  r["vol_mae_cm3"], r["centroid_err_m"], 100 * r["loc_ok_rate"]))
    print("saved: %s" % path)


if __name__ == "__main__":
    main()