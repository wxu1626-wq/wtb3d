"""Train + evaluate swappable point-cloud backbones (requires torch, GPU box).

Backbones (same benchmark, same protocol):
  pointnet           classic shared-MLP PointNet (deep/pointnet.py)
  pointnetpp         fixed-resolution 2-stage PointNet++ w/ knn context
  dgcnn              2-layer dynamic-graph CNN (EdgeConv)
  pointtransformer   2-layer kNN relative-position self-attention

Protocol (matches run_learned.py exactly):
  * same seeds -> same scenes, same stratified scene-level split
  * decision threshold tuned on a VAL split (10% of train scenes) only
  * baseline tau tuned on TRAIN only; both frozen and evaluated on TEST
  * scene-level engineering metrics (area / volume / localization) are
    reported for BOTH the learned mask and the classical mask
  * depth head (optional) regresses |deviation| (mm) on GT-damaged points

Usage:
  pip install torch            # CPU or CUDA build
  python deep/run_pointnet.py --num 1000 --epochs 40 --backbone dgcnn --batch 8 --cool 5
Outputs (per backbone):
  results/<backbone>_<num>_comparison.csv
  results/<backbone>_<num>_scene.csv
  results/<backbone>_<num>_config.json
  results/<backbone>_<num>_ckpt.pt  (best-val checkpoint)
"""
import os
# ---- load guardrails: cap native thread pools BEFORE numpy/scipy/torch init ----
# Oversubscribed OMP/BLAS thread pools were the top suspect for the 0x28
# MEMORY_MANAGEMENT BSODs on the laptop. Override from the shell if needed.
for _v, _d in (("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
               ("OPENBLAS_NUM_THREADS", "4")):
    os.environ.setdefault(_v, _d)
import sys, csv, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
try:  # cap torch's own pools (no-op if parallelism already started)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from torch.utils.data import DataLoader

from data_gen.config import GenConfig
from data_gen.geometry import loft_blade
from data_gen import damage as dm
from deep.dataset import (build_arrays, SceneDataset, pad_collate,
                          stratified_scene_split)
from deep.backbones import build_backbone
from baselines.evaluate import prf, geo_err
from baselines.scene_metrics import (mesh_area, scene_damage_metrics,
                                     aggregate_scene_metrics)

TYPE_ORDER = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM, dm.HEALTHY]
BACKBONES = ["pointnet", "pointnetpp", "dgcnn", "pointtransformer"]
TAUS = [0.5, 1, 2, 3, 5, 8, 12, 20]

THR = 0.5  # set by main() before evaluate() is used


def blade_mesh_area():
    """Nominal blade surface area (m2) for the default blade geometry."""
    cfg = GenConfig(num_scenes=1, points=1, n_ref=1, seed=0, out_dir="data")
    V, F, N = loft_blade(cfg.n_sections, cfg.n_airfoil, cfg.span,
                         cfg.c_root, cfg.c_tip, cfg.twist_deg)
    return mesh_area(V, F)


def scene_slice(arrays, i):
    a = int(arrays["offsets"][i]); b = a + int(arrays["lens"][i])
    return a, b


def micro_f1(mask, y):
    tp = int((mask & y).sum()); fp = int((mask & ~y).sum()); fn = int((~mask & y).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


@torch.no_grad()
def predict_probs(model, loader, device):
    probs, devs, yss = [], [], []
    model.eval()
    for X, y, dv, mask, knn in loader:
        Xg, maskg, knng = X.to(device), mask.to(device), knn.to(device)
        out = model(Xg, maskg, knng)
        logits = out[0] if isinstance(out, tuple) else out
        probs.append(torch.sigmoid(logits)[maskg].cpu())
        devs.append(dv[mask])
        yss.append(y[mask])
    return torch.cat(probs), torch.cat(devs), torch.cat(yss)


def evaluate(arrays, scene_idx, pred, baseline_tau, S_mesh, method_name):
    """Per-type + micro point-level metrics AND scene-level engineering
    metrics, for both the learned mask and the classical mask.

    Returns (rows, scene_rows_learned, scene_rows_baseline).
    """
    acc = {t: dict(tp=0, fp=0, fn=0) for t in TYPE_ORDER}
    accb = {t: dict(acc[t]) for t in TYPE_ORDER}  # deep copy: shallow dict(acc) shares inner dicts
    geo = {t: [] for t in TYPE_ORDER}
    sl, sb = [], []
    for i in scene_idx:
        a, b = scene_slice(arrays, i)
        y = arrays["y"][a:b]
        prob = pred[i][0]; dev = pred[i][1]
        mask_l = prob > THR
        mask_b = np.abs(dev) > baseline_tau
        res_l = prf(mask_l, y); res_b = prf(mask_b, y)
        t = int(arrays["types"][i])
        for k in ("tp", "fp", "fn"):
            acc[t][k] += res_l[k]; accb[t][k] += res_b[k]
        g = geo_err(dev * 1e-3, arrays["gt_depth_mm"][a:b], y)
        if g["n"]:
            geo[t].append(g)
        pts = arrays["X"][a:b, :3]
        a_w = S_mesh / (b - a)
        sl.append(scene_damage_metrics(mask_l, y, pts, dev,
                                       arrays["gt_depth_mm"][a:b], a_w))
        sb.append(scene_damage_metrics(mask_b, y, pts, dev,
                                       arrays["gt_depth_mm"][a:b], a_w))
    rows = []
    for m, A in (("baseline_tau%.1f" % baseline_tau, accb),
                 (method_name, acc)):
        for t in TYPE_ORDER:
            a = A[t]
            p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 0.0
            r = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            iou = a["tp"] / (a["tp"] + a["fp"] + a["fn"]) if (a["tp"] + a["fp"] + a["fn"]) else 0.0
            maes = [x["mae"] for x in geo[t]]; rmses = [x["rmse"] for x in geo[t]]
            rows.append(dict(method=m, type=t, type_name=dm.TYPE_NAMES[t],
                             n_scenes=int((arrays["types"] == t)[np.array(scene_idx)].sum()),
                             precision=p, recall=r, f1=f1, iou=iou,
                             tp=a["tp"], fp=a["fp"], fn=a["fn"],
                             geo_mae_mm=float(np.mean(maes)) if maes else np.nan,
                             geo_rmse_mm=float(np.mean(rmses)) if rmses else np.nan))
    return rows, sl, sb


def main():
    global THR
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--split-seed", type=int, default=2024)
    ap.add_argument("--val-frac-of-train", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cool", type=float, default=0.0,
                    help="sleep seconds between epochs (thermal/RAM relief, laptop)")
    ap.add_argument("--backbone", type=str, default="pointnet",
                    choices=BACKBONES)
    ap.add_argument("--depth-head", action="store_true")
    ap.add_argument("--depth-w", type=float, default=0.5)
    ap.add_argument("--init-from", type=str, default="",
                    help="optional .pt state_dict (e.g. pretrain_<b>_<n>.pt)")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(args.out, exist_ok=True)

    cache = os.path.join("work", "arrays_%d_%d_%d_%d.npz" % (
        args.num, args.points, args.n_ref, args.seed))
    if os.path.exists(cache):
        z = np.load(cache)
        arrays = {k: z[k] for k in z.files}
        print("arrays loaded from cache: %s" % cache)
    else:
        arrays = build_arrays(args.num, args.points, args.n_ref, args.seed)
        os.makedirs("work", exist_ok=True)
        np.savez(cache, **arrays)
        print("arrays saved to cache: %s" % cache)
    S_mesh = blade_mesh_area()
    train_idx, test_idx = stratified_scene_split(arrays["types"], args.train_frac,
                                                 args.split_seed)
    rsplit = np.random.default_rng(args.split_seed + 1)
    rsplit.shuffle(train_idx)
    nval = max(1, int(round(len(train_idx) * args.val_frac_of_train)))
    val_idx, fit_idx = train_idx[:nval], train_idx[nval:]
    print("split: fit=%d val=%d test=%d scenes" % (len(fit_idx), len(val_idx), len(test_idx)))

    # baseline tau on TRAIN (fit+val) deviation field
    dev_tr = np.concatenate([arrays["dev_mm"][scene_slice(arrays, i)[0]:scene_slice(arrays, i)[1]] for i in train_idx])
    y_tr = np.concatenate([arrays["y"][scene_slice(arrays, i)[0]:scene_slice(arrays, i)[1]] for i in train_idx])
    best_tau = max(TAUS, key=lambda t: micro_f1(np.abs(dev_tr) > t, y_tr))
    print("baseline best tau (train): %.1f mm" % best_tau)

    fit_ds = SceneDataset(arrays, fit_idx)
    val_ds = SceneDataset(arrays, val_idx)
    test_ds = SceneDataset(arrays, test_idx)
    fit_ld = DataLoader(fit_ds, batch_size=args.batch, shuffle=True, collate_fn=pad_collate)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=pad_collate)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=pad_collate)

    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    pos_w = torch.tensor(n_neg / max(n_pos, 1), device=device)
    model = build_backbone(args.backbone, in_dim=7,
                           depth_head=args.depth_head).to(device)
    if args.init_from:
        st = torch.load(args.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(st, strict=False)
        print("init-from %s: ckpt %d params, %d unexpected, %d missing"
              % (args.init_from, len(st), len(unexpected), len(missing)))
    hb = os.path.join("work", "heartbeat_%s.txt" % args.backbone)
    os.makedirs("work", exist_ok=True)
    with open(hb, "a") as fh:
        fh.write("t=%s START backbone=%s batch=%d cool=%.1f device=%s\n" % (
            time.strftime("%H:%M:%S"), args.backbone, args.batch, args.cool, device))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); tot, npts = 0.0, 0
        for X, y, dv, mask, knn in fit_ld:
            X, y, dv, mask, knn = (z.to(device)
                                   for z in (X, y, dv, mask, knn))
            opt.zero_grad()
            out = model(X, mask, knn)
            if isinstance(out, tuple):
                logits, depth = out
                loss = bce(logits[mask], y[mask])
                dmg = mask & (y > 0.5)
                if int(dmg.sum()) > 0:
                    loss = loss + args.depth_w * nn.functional.l1_loss(
                        depth[dmg], dv.abs()[dmg])
            else:
                loss = bce(out[mask], y[mask])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * int(mask.sum()); npts += int(mask.sum())
        sched.step()
        vp, vd, vy = predict_probs(model, val_ld, device)
        best_thr = max(np.linspace(0.05, 0.95, 91),
                       key=lambda t: micro_f1(vp.numpy() > t, vy.numpy() > 0.5))
        v_f1 = micro_f1(vp.numpy() > best_thr, vy.numpy() > 0.5)
        print("epoch %02d  loss=%.4f  valF1=%.4f (thr %.2f)  %.1fs" % (
            ep, tot / max(npts, 1), v_f1, best_thr, time.time() - t0))
        with open(hb, "a") as fh:
            fh.write("t=%s ep=%d valF1=%.4f thr=%.2f gpuMB=%.0f\n" % (
                time.strftime("%H:%M:%S"), ep, v_f1, best_thr,
                torch.cuda.max_memory_allocated(device) / 2**20
                if device == "cuda" else 0))
        if args.cool > 0:
            time.sleep(args.cool)
        if v_f1 > best_f1:
            best_f1, best_thr = v_f1, float(best_thr)
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print("early stop at epoch %d" % ep)
                break

    if device == "cuda":
        print("peak GPU mem: %.0f MB allocated / %.0f MB reserved" % (
            torch.cuda.max_memory_allocated(device) / 2**20,
            torch.cuda.max_memory_reserved(device) / 2**20))
    tag = "%s_%d" % (args.backbone, args.num)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(args.out, "%s_ckpt.pt" % tag))
    THR = best_thr

    # ---- final evaluation on TEST ----
    tp_, td, ty = predict_probs(model, test_ld, device)
    per_scene = {}
    flat = 0
    for i in test_idx:
        a, b = scene_slice(arrays, i)
        n = b - a
        per_scene[i] = (tp_[flat:flat + n].numpy(), td[flat:flat + n].numpy())
        flat += n
    rows, sl, sb = evaluate(arrays, test_idx, per_scene, best_tau, S_mesh,
                            args.backbone)

    csv_path = os.path.join(args.out, "%s_comparison.csv" % tag)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- scene-level engineering metrics CSV ----
    types_te = [int(arrays["types"][i]) for i in test_idx]
    sm_l = aggregate_scene_metrics(sl, types_te, dm.TYPE_NAMES, TYPE_ORDER)
    sm_b = aggregate_scene_metrics(sb, types_te, dm.TYPE_NAMES, TYPE_ORDER)
    scene_csv = os.path.join(args.out, "%s_scene.csv" % tag)
    with open(scene_csv, "w", newline="") as f:
        fieldnames = ["method"] + list(sm_l[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for m, sm in (("baseline_tau%.1f" % best_tau, sm_b),
                      (args.backbone, sm_l)):
            for r in sm:
                w.writerow(dict(method=m, **r))

    with open(os.path.join(args.out, "%s_config.json" % tag), "w") as f:
        json.dump(dict(num=args.num, points=args.points, seed=args.seed,
                       device=device, epochs=args.epochs, batch=args.batch,
                       lr=args.lr, backbone=args.backbone,
                       depth_head=args.depth_head,
                       init_from=(args.init_from or None),
                       train_frac=args.train_frac, split_seed=args.split_seed,
                       best_tau_mm=best_tau, best_thr=THR, best_val_f1=best_f1),
                  f, indent=2)

    print("")
    print("=== %s vs BASELINE  (test, tau=%.1fmm, thr=%.2f) ===" % (
        args.backbone.upper(), best_tau, THR))
    for r in rows:
        print("  %-18s %-13s n=%3d P=%.3f R=%.3f F1=%.3f IoU=%.3f" % (
            r["method"], r["type_name"], r["n_scenes"], r["precision"],
            r["recall"], r["f1"], r["iou"]))
    print("")
    print("=== SCENE-LEVEL ENGINEERING METRICS (test) ===")
    for m, sm in (("baseline_tau%.1f" % best_tau, sm_b),
                  (args.backbone, sm_l)):
        for r in sm:
            print("  %-18s %-8s area %.4f->%.4f m2 (MAE %.4f) | vol %.2f->%.2f cm3 "
                  "(MAE %.2f) | loc %.3f m ok %.0f%%" % (
                      m, r["type_name"], r["area_gt_m2"], r["area_pred_m2"],
                      r["area_abs_err_m2"], r["vol_gt_cm3"], r["vol_pred_cm3"],
                      r["vol_mae_cm3"], r["centroid_err_m"],
                      100 * r["loc_ok_rate"]))
    print("saved: %s" % csv_path)
    print("saved: %s" % scene_csv)


if __name__ == "__main__":
    main()

