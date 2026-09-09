"""Generate the persistent open-source dataset (dense reference + scene scans + manifest)."""
import os, csv, argparse
import numpy as np
from data_gen.config import GenConfig
from data_gen.geometry import loft_blade
from data_gen import damage as dm
from data_gen.sensor_sim import to_scan, build_reference
from data_gen.utils import save_scene, save_reference, write_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--ply-every", type=int, default=10)
    args = ap.parse_args()

    cfg = GenConfig(num_scenes=args.num, points=args.points, n_ref=args.n_ref,
                    seed=args.seed, out_dir=args.out)
    rng = np.random.default_rng(cfg.seed)
    rng_ref = np.random.default_rng(cfg.seed + 1000)
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    nominal, nominal_nrm = build_reference(V, F, N, cfg.n_ref, rng_ref)
    os.makedirs(args.out, exist_ok=True)
    save_reference(os.path.join(args.out, "reference.npz"), nominal, nominal_nrm)

    manifest = []
    for i in range(cfg.num_scenes):
        disp, lab, meta = dm.generate_damage(V, N, cfg, rng)
        scen = to_scan(V, F, N, disp, lab, cfg, rng, nominal, nominal_nrm)
        meta["pose_R"] = scen["pose"][0].tolist()
        meta["pose_t"] = scen["pose"][1].tolist()
        path = os.path.join(args.out, "scene_%04d.npz" % i)
        save_scene(path, scen, meta)
        if i % max(1, args.ply_every) == 0:
            Vd = V + N * disp[:, None]
            write_ply(os.path.join(args.out, "scene_%04d_damaged.ply" % i), Vd, N)
        manifest.append({"id": i, "type": meta["type"], "type_name": meta["type_name"],
                         "n_damaged_vertices": meta.get("n_damaged_vertices", 0),
                         "n_gt_points": int(scen["gt_mask"].sum()),
                         "file": os.path.basename(path)})
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        for r in manifest:
            w.writerow(r)
    print("wrote %d scenes + reference(%d pts) to %s" % (cfg.num_scenes, cfg.n_ref, args.out))


if __name__ == "__main__":
    main()