"""Train + evaluate the PointNet-style model (requires torch, run on GPU box).

Protocol (matches run_learned.py exactly):
  * same seeds -> same scenes, same stratified scene-level split
  * decision threshold tuned on a VAL split (10% of train scenes) only
  * baseline tau tuned on TRAIN only; both frozen and evaluated on TEST
  * depth head (optional) regresses |deviation| (mm) on GT-damaged points

Usage:
  pip install torch            # CPU or CUDA build
  python deep/run_pointnet.py --num 1000 --epochs 40
Outputs:
  results/pointnet_<num>_comparison.csv
  results/pointnet_<num>_config.json
  results/pointnet_<num>_ckpt.pt  (best-val checkpoint)
"""
import os, sys, csv, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_gen import damage as dm
from deep.dataset import (build_arrays, SceneDataset, pad_collate,
                          stratified_scene_split)
from deep.pointnet import PointNetSeg
from baselines.evaluate import prf, geo_err

TYPE_ORDER = [dm.LEE, dm.CRACK, dm.DENT, dm.DEFORM, dm.HEALTHY]
TAUS = [0.5, 1, 2, 3, 5, 8, 12, 20]


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
    for X, y, dv, mask in loader:
        X = X.to(device); mask = mask.to(device)
        out = model(X)
        logits = out[0] if isinstance(out, tuple) else out
        probs.append(torch.sigmoid(logits)[mask].cpu())
        devs.append(dv[mask])
        yss.append(y[mask])
    return torch.cat(probs), torch.cat(devs), torch.cat(yss)


def evaluate(arrays, scene_idx, pred, baseline_tau, device):
    """Per-type + micro metrics on a set of scenes.

    pred: dict scene_idx -> (prob_array, dev_array) or callable.
    """
    acc = {t: dict(tp=0, fp=0, fn=0) for t in TYPE_ORDER}
    accb = dict(acc)
    geo = {t: [] for t in TYPE_ORDER}
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
    rows = []
    for m, A in (("baseline_tau%.1f" % baseline_tau, accb),
                 ("pointnet", acc)):
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
    return rows


THR = 0.5  # set by main() before evaluate() is used


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
    ap.add_argument("--depth-head", action="store_true")
    ap.add_argument("--depth-w", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(args.out, exist_ok=True)

    arrays = build_arrays(args.num, args.points, args.n_ref, args.seed)
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
    model = PointNetSeg(in_dim=7, depth_head=args.depth_head).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); tot, npts = 0.0, 0
        for X, y, dv, mask in fit_ld:
            X, y, dv, mask = (z.to(device) for z in (X, y, dv, mask))
            opt.zero_grad()
            out = model(X)
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
            tot += float(loss) * int(mask.sum()); npts += int(mask.sum())
        sched.step()
        vp, vd, vy = predict_probs(model, val_ld, device)
        best_thr = max(np.linspace(0.05, 0.95, 91),
                       key=lambda t: micro_f1(vp.numpy() > t, vy.numpy() > 0.5))
        v_f1 = micro_f1(vp.numpy() > best_thr, vy.numpy() > 0.5)
        print("epoch %02d  loss=%.4f  valF1=%.4f (thr %.2f)  %.1fs" % (
            ep, tot / max(npts, 1), v_f1, best_thr, time.time() - t0))
        if v_f1 > best_f1:
            best_f1, best_thr = v_f1, float(best_thr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print("early stop at epoch %d" % ep)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(args.out, "pointnet_%d_ckpt.pt" % args.num))
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
    rows = evaluate(arrays, test_idx, per_scene, best_tau, device)

    csv_path = os.path.join(args.out, "pointnet_%d_comparison.csv" % args.num)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.out, "pointnet_%d_config.json" % args.num), "w") as f:
        json.dump(dict(num=args.num, points=args.points, seed=args.seed,
                       device=device, epochs=args.epochs, batch=args.batch,
                       lr=args.lr, depth_head=args.depth_head,
                       train_frac=args.train_frac, split_seed=args.split_seed,
                       best_tau_mm=best_tau, best_thr=THR, best_val_f1=best_f1),
                  f, indent=2)

    print("")
    print("=== POINTNET vs BASELINE  (test, tau=%.1fmm, thr=%.2f) ===" % (best_tau, THR))
    for r in rows:
        print("  %-18s %-13s n=%3d P=%.3f R=%.3f F1=%.3f IoU=%.3f" % (
            r["method"], r["type_name"], r["n_scenes"], r["precision"],
            r["recall"], r["f1"], r["iou"]))
    print("saved: %s" % csv_path)


if __name__ == "__main__":
    main()