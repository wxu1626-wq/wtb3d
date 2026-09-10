"""Self-supervised deviation-field pretraining (GPU box, requires torch).

Task: regress |deviation| (mm) at EVERY point of EVERY scene using only the
measurement (xyz + scanned normal + ICP signed deviation). No GT damage
label is consumed, so the backbone can be pretrained on arbitrarily many
scans -- including healthy / unknown-condition ones. The pretrained weights
are then loaded as the starting point for supervised fine-tuning:

  python deep/pretrain.py --num 1000 --epochs 30 --backbone dgcnn
  python deep/run_pointnet.py --num 1000 --backbone dgcnn \
      --init-from results/pretrain_dgcnn_1000.pt

Contribution I2: the network learns a dense, geometry-aware representation
of "what looks like a local surface anomaly" before ever seeing a damage
label, and the label is only needed for the cheap final fine-tuning stage.

Outputs:
  results/pretrain_<backbone>_<num>.pt
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from deep.dataset import build_arrays, SceneDataset, pad_collate
from deep.backbones import build_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", type=str, default="dgcnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(args.out, exist_ok=True)

    arrays = build_arrays(args.num, args.points, args.n_ref, args.seed)
    scene_idx = list(range(args.num))               # ALL scenes; no GT used
    ds = SceneDataset(arrays, scene_idx)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=pad_collate)

    model = build_backbone(args.backbone, in_dim=7, depth_head=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    l1 = torch.nn.L1Loss()

    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); tot, npts = 0.0, 0
        for X, y, dv, mask, knn in ld:
            X, dv, mask, knn = (X.to(device), dv.to(device),
                                mask.to(device), knn.to(device))
            opt.zero_grad()
            out = model(X, mask, knn)
            _, depth = out
            loss = l1(depth[mask], dv[mask].abs())
            loss.backward()
            opt.step()
            tot += float(loss) * int(mask.sum()); npts += int(mask.sum())
        sched.step()
        print("pretrain epoch %02d  |dev| L1 = %.3f mm  %.1fs" % (
            ep + 1, tot / max(npts, 1), time.time() - t0))

    ckpt = os.path.join(args.out, "pretrain_%s_%d.pt" % (args.backbone, args.num))
    torch.save(model.state_dict(), ckpt)
    print("saved: %s" % ckpt)
    print("fine-tune with:  python deep/run_pointnet.py --backbone %s "
          "--init-from %s" % (args.backbone, ckpt))


if __name__ == "__main__":
    main()