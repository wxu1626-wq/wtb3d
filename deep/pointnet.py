"""PointNet-style per-point damage segmentation on scanned blade point clouds.

Input per point: aligned xyz (m), scanned normal, ICP signed deviation (mm).
The network learns a location/neighbourhood-aware damage decision that
subsumes the classical global-threshold rule. An optional regression head
predicts |deviation| (mm) for quantitative depth estimation.
"""
import torch
import torch.nn as nn


class PointNetSeg(nn.Module):
    """Shared-MLP PointNet with global context, per-point damage head.

    forward: (B, D, N) -> (B, N) logits  [and (B, N) depth if depth_head]
    """

    def __init__(self, in_dim=7, hidden=(64, 128), head_dim=(256, 128),
                 depth_head=False, dropout=0.1):
        super().__init__()
        feats = []
        d = in_dim
        for h in hidden:
            feats += [nn.Conv1d(d, h, 1), nn.BatchNorm1d(h), nn.ReLU(),
                      nn.Dropout(dropout)]
            d = h
        self.feat = nn.Sequential(*feats)
        gdim = d

        ctx = []
        d = gdim * 2
        for h in head_dim:
            ctx += [nn.Conv1d(d, h, 1), nn.BatchNorm1d(h), nn.ReLU(),
                    nn.Dropout(dropout)]
            d = h
        self.ctx = nn.Sequential(*ctx)
        self.head = nn.Conv1d(d, 1, 1)
        self.depth_head = nn.Conv1d(d, 1, 1) if depth_head else None

    def forward(self, x):
        f = self.feat(x)                                   # (B, g, N)
        g = f.max(dim=2, keepdim=True).values              # (B, g, 1)
        g = g.expand(-1, -1, f.size(2))                    # (B, g, N)
        c = self.ctx(torch.cat([f, g], dim=1))             # (B, H, N)
        logits = self.head(c).squeeze(1)                   # (B, N)
        if self.depth_head is not None:
            depth = self.depth_head(c).squeeze(1).abs()
            return logits, depth
        return logits