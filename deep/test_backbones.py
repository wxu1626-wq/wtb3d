"""Smoke test: forward + backward pass for every backbone on a random batch.

Usage (GPU box, after pip install torch):
  python deep/test_backbones.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deep.backbones import build_backbone

NAMES = ["pointnet", "pointnetpp", "dgcnn", "pointtransformer"]


def main():
    torch.manual_seed(0)
    B, D, N, K = 4, 7, 512, 16
    X = torch.randn(B, D, N)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 100:] = False                          # variable-length scene
    knn = torch.randint(0, 512, (B, N, K))

    for name in NAMES:
        for depth_head in (False, True):
            model = build_backbone(name, in_dim=D, depth_head=depth_head)
            out = model(X, mask, knn)
            if depth_head:
                logits, depth = out
                assert logits.shape == (B, N) and depth.shape == (B, N)
                assert (depth >= 0).all()
                loss = logits[mask].float().mean() + depth[mask].float().mean()
            else:
                logits = out
                assert logits.shape == (B, N)
                loss = logits[mask].float().mean()
            loss.backward()
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print("%-17s depth_head=%-5s forward+backward ok  (%.2f M params)" % (
                name, depth_head, n_params))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("all backbones ok on", dev)


if __name__ == "__main__":
    main()