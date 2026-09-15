"""Swappable point-cloud backbones for per-point damage segmentation.

All backbones share one interface:

    forward(X (B,D,N), mask (B,N) bool, knn (B,N,K) long)
        -> logits (B,N)                       (classification)
        -> (logits (B,N), depth (B,N) mm)     if depth_head=True

X: per-point features [aligned xyz (m), scanned normal, dev_mm].
knn: scene-local kNN indices (padded; always in-range for the batch).

Backbones:
  pointnet           classic shared-MLP PointNet (ignores knn; adapter)
  pointnetpp         fixed-resolution 2-stage PointNet++ w/ knn max-pool ctx
  dgcnn              2-layer dynamic-graph CNN (EdgeConv)
  pointtransformer   2-layer kNN relative-position self-attention

Memory note: neighbour gathering materialises (B,D,N,K) tensors; if a GPU
OOMs, lower --batch in run_pointnet.py.
"""
import torch
import torch.nn as nn

from deep.pointnet import PointNetSeg


def _gather_neighbors(X, knn):
    """X (B,D,N), knn (B,N,K) -> neighbour features (B,D,N,K).

    out[b,d,i,k] = X[b,d,knn[b,i,k]]
    """
    Xp = X.permute(0, 2, 1)                                # (B, N, D)
    D = X.shape[1]; K = knn.size(2)
    idx = knn.unsqueeze(-1).expand(-1, -1, -1, D)          # (B, N, K, D)
    Xrep = Xp.unsqueeze(2).expand(-1, -1, K, D)            # zero-copy view
    return Xrep.gather(1, idx).permute(0, 3, 1, 2)         # (B, D, N, K)


def _gather_mask(mask, knn):
    """mask (B,N) bool, knn (B,N,K) -> neighbour validity (B,1,N,K)."""
    B, N = mask.shape
    K = knn.size(2)
    M = mask.unsqueeze(1).unsqueeze(3).expand(B, 1, N, K)  # mask[b,1,n,k] = mask[b,n]
    return M.gather(2, knn.unsqueeze(1).expand(B, 1, N, K))


class _EdgeConv(nn.Module):
    """EdgeConv: MLP on [x_i, x_j - x_i] over the knn set, max-pool to N."""

    def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Dropout(dropout))

    def forward(self, X, knn, mask):
        B, D, N = X.shape
        K = knn.size(2)
        Xn = _gather_neighbors(X, knn)                      # (B, D, N, K)
        nm = _gather_mask(mask, knn)                        # (B, 1, N, K)
        xi = X.unsqueeze(3).expand(B, D, N, K)
        edge = torch.cat([xi, Xn - xi], dim=1)              # (B, 2D, N, K)
        edge = edge.masked_fill(~nm, 0.0)  # nm (B,1,N,K) broadcasts over 2D
        h = self.net(edge.reshape(B, -1, N * K))            # (B, out, N*K)
        h = h.reshape(B, -1, N, K)
        h = h.masked_fill(~nm, -1e9)
        return h.amax(dim=3)                                # (B, out, N)


class PointNetPPSeg(nn.Module):
    """Fixed-resolution PointNet++-style segmentation.

    Two stacked conv stages; each stage's input is augmented with the
    max-pooled kNN neighbourhood (light-weight stand-in for SA modules --
    random sampling is unnecessary for a single rigid scan per scene).
    """

    def __init__(self, in_dim=7, hidden=64, hidden2=128, head_dim=(256, 128),
                 depth_head=False, dropout=0.1):
        super().__init__()
        self.sa1 = nn.Sequential(
            nn.Conv1d(2 * in_dim, hidden, 1), nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden, hidden, 1), nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2))
        self.sa2 = nn.Sequential(
            nn.Conv1d(2 * hidden, hidden2, 1), nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden2, hidden2, 1), nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.2))
        d = hidden2 + hidden + in_dim  # h2 + h1 + X
        ctx = []
        din = 2 * d                    # f + global ctx g
        for h in head_dim:
            ctx += [nn.Conv1d(din, h, 1), nn.BatchNorm1d(h), nn.ReLU(),
                    nn.Dropout(dropout)]
            din = h
            d = h
        self.ctx = nn.Sequential(*ctx)
        self.head = nn.Conv1d(d, 1, 1)
        self.depth_head = nn.Conv1d(d, 1, 1) if depth_head else None

    def forward(self, X, mask, knn):
        X = X.masked_fill(~mask.unsqueeze(1), 0.0)
        nm = _gather_mask(mask, knn)
        Xnb = _gather_neighbors(X, knn).masked_fill(~nm, -1e9).amax(dim=3)
        Xnb = Xnb.masked_fill(~mask.unsqueeze(1), 0.0)
        h1 = self.sa1(torch.cat([X, Xnb], dim=1))           # (B, 64, N)
        h1 = h1.masked_fill(~mask.unsqueeze(1), 0.0)
        h1n = _gather_neighbors(h1, knn).masked_fill(~nm, -1e9).amax(dim=3)
        h1n = h1n.masked_fill(~mask.unsqueeze(1), 0.0)
        h2 = self.sa2(torch.cat([h1, h1n], dim=1))          # (B, 128, N)
        h2 = h2.masked_fill(~mask.unsqueeze(1), 0.0)
        f = torch.cat([h2, h1, X], dim=1)
        g = f.max(dim=2, keepdim=True).values.expand(-1, -1, X.size(2))
        c = self.ctx(torch.cat([f, g], dim=1))
        logits = self.head(c).squeeze(1)
        if self.depth_head is not None:
            return logits, self.depth_head(c).squeeze(1).abs()
        return logits


class DGCNNSeg(nn.Module):
    """Two-layer dynamic-graph CNN (EdgeConv, Wang et al. 2019)."""

    def __init__(self, in_dim=7, hidden=64, hidden2=128, depth_head=False,
                 dropout=0.1):
        super().__init__()
        self.ec1 = _EdgeConv(2 * in_dim, hidden, hidden, dropout)
        self.ec2 = _EdgeConv(2 * (in_dim + hidden), hidden2, hidden2, dropout)
        self.ctx = nn.Sequential(
            nn.Conv1d(hidden2 * 2, hidden2, 1), nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Conv1d(hidden2, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout))
        self.head = nn.Conv1d(64, 1, 1)
        self.depth_head = nn.Conv1d(64, 1, 1) if depth_head else None

    def forward(self, X, mask, knn):
        X = X.masked_fill(~mask.unsqueeze(1), 0.0)
        h1 = self.ec1(X, knn, mask)
        h1 = h1.masked_fill(~mask.unsqueeze(1), 0.0)
        h2 = self.ec2(torch.cat([h1, X], dim=1), knn, mask)
        h2 = h2.masked_fill(~mask.unsqueeze(1), 0.0)
        g = h2.max(dim=2, keepdim=True).values.expand(-1, -1, X.size(2))
        c = self.ctx(torch.cat([h2, g], dim=1))
        logits = self.head(c).squeeze(1)
        if self.depth_head is not None:
            return logits, self.depth_head(c).squeeze(1).abs()
        return logits


class _RelPosBlock(nn.Module):
    """kNN relative-position self-attention (one point attends to its K
    neighbours: query = own feature, key = neighbour feature + relative
    position, value = neighbour feature).
    """

    def __init__(self, d=64, heads=4, dropout=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.norm_q = nn.LayerNorm(d)
        self.norm_k = nn.LayerNorm(d)
        self.norm_v = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.pos = nn.Linear(d, d)  # relative-feature embedding (hidden has no raw xyz)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, X, knn, mask):
        B, D, N = X.shape
        K = knn.size(2)
        valid = _gather_mask(mask, knn)                     # (B, 1, N, K)
        xn = _gather_neighbors(X, knn).masked_fill(~valid, 0.0)
        rel = _gather_neighbors(X, knn) - X.unsqueeze(3)    # (B, 3, N, K)
        q = self.q(self.norm_q(X.permute(0, 2, 1))).permute(0, 2, 1)                          # (B, D, N)
        k = self.k(self.norm_k(xn.permute(0, 3, 2, 1))).permute(0, 3, 2, 1) + self.pos(rel.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)         # (B, D, N, K)
        v = self.v(self.norm_v(xn.permute(0, 3, 2, 1))).permute(0, 3, 2, 1)                         # (B, D, N, K)
        qh = q.reshape(B, N, self.h, self.dh).permute(0, 2, 1, 3)      # (B,h,N,dh)
        kh = k.reshape(B, N, K, self.h, self.dh).permute(0, 3, 1, 2, 4)
        vh = v.reshape(B, N, K, self.h, self.dh).permute(0, 3, 1, 2, 4)
        scores = torch.einsum("bhnd,bhnkd->bhnk", qh, kh) / (self.dh ** 0.5)
        scores = scores.masked_fill(~valid, -1e9)  # valid (B,1,N,K) broadcasts over heads
        attn = self.drop(scores.softmax(dim=-1))
        out = torch.einsum("bhnk,bhnkd->bhnd", attn, vh)    # (B, h, N, dh)
        out = out.permute(0, 2, 3, 1).reshape(B, N, D)
        return self.proj(out).permute(0, 2, 1)              # (B, D, N)


class PointTransformerSeg(nn.Module):
    """kNN relative-position transformer, 2 layers + global max-pool ctx."""

    def __init__(self, in_dim=7, d=64, heads=4, layers=2, depth_head=False,
                 dropout=0.1):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Conv1d(in_dim, d, 1),
                                     nn.BatchNorm1d(d), nn.ReLU())
        self.blocks = nn.ModuleList([_RelPosBlock(d, heads, dropout)
                                     for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ctx = nn.Sequential(nn.Conv1d(d * 2, 128, 1), nn.BatchNorm1d(128),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Conv1d(128, 1, 1)
        self.depth_head = nn.Conv1d(128, 1, 1) if depth_head else None

    def forward(self, X, mask, knn):
        X = X.masked_fill(~mask.unsqueeze(1), 0.0)
        h = self.in_proj(X)
        for blk, ln in zip(self.blocks, self.norms):
            h = h + blk(h, knn, mask)
            h = ln(h.permute(0, 2, 1)).permute(0, 2, 1)  # back to (B, D, N)
            h = h.masked_fill(~mask.unsqueeze(1), 0.0)
        g = h.max(dim=2, keepdim=True).values.expand(-1, -1, X.size(2))
        c = self.ctx(torch.cat([h, g], dim=1))
        logits = self.head(c).squeeze(1)
        if self.depth_head is not None:
            return logits, self.depth_head(c).squeeze(1).abs()
        return logits


class _PointNetAdapter(nn.Module):
    """Wrap the legacy PointNetSeg (forward(x) only) in the shared interface."""

    def __init__(self, in_dim=7, depth_head=False, **_kw):
        super().__init__()
        self.inner = PointNetSeg(in_dim=in_dim, depth_head=depth_head)

    def forward(self, X, mask, knn):
        return self.inner(X.masked_fill(~mask.unsqueeze(1), 0.0))


def build_backbone(name, in_dim=7, depth_head=False):
    """Factory: name -> backbone module with the shared forward interface."""
    if name == "pointnet":
        return _PointNetAdapter(in_dim=in_dim, depth_head=depth_head)
    if name == "pointnetpp":
        return PointNetPPSeg(in_dim=in_dim, depth_head=depth_head)
    if name == "dgcnn":
        return DGCNNSeg(in_dim=in_dim, depth_head=depth_head)
    if name == "pointtransformer":
        return PointTransformerSeg(in_dim=in_dim, depth_head=depth_head)
    raise ValueError("unknown backbone: %s" % name)
















