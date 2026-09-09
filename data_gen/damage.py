"""Damage generators on the blade vertex grid.

Each generator returns (disp, lab, meta):
  disp : (M,) per-vertex displacement along the outward normal (meters).
          Negative = material removed (dent / erosion / crack); positive = bulge.
  lab  : (M,) per-vertex damage label (int). 0 = healthy.
  meta : dict describing the synthetic damage (for the dataset card / reproducibility).

All damage is LOCAL and physically plausible, and distinguishable by shape:
  DENT      spherical inward dimple (fod / hail)
  LEE       leading-edge erosion band over a span interval
  CRACK     spanwise groove band at a fixed chordwise line
  DEFORM    localized outward bulge (delamination / deformation)
Spans are chosen by ring index so every region is guaranteed non-empty.
"""
import numpy as np

HEALTHY = 0
LEE = 1
CRACK = 2
DENT = 3
DEFORM = 4

TYPE_NAMES = {0: "healthy", 1: "lee_erosion", 2: "crack", 3: "dent", 4: "deformation"}


def _blade_ctx(V, N, cfg):
    M = V.shape[0]
    naf = M // cfg.n_sections
    ring_of = np.arange(M) // naf
    X = V[:, 0].copy()
    le_idx = np.empty(cfg.n_sections, dtype=np.int64)
    for r in range(cfg.n_sections):
        lo, hi = r * naf, (r + 1) * naf
        le_idx[r] = lo + int(np.argmin(X[lo:hi]))
    le_off = X - X[le_idx][ring_of]              # each vertex distance from its ring LE
    ringZ = V[np.arange(cfg.n_sections) * naf, 2].copy()
    return dict(naf=naf, ring_of=ring_of, le_off=le_off, X=X, ringZ=ringZ,
                Z=V[:, 2].copy(), vmin=float(V[:, 2].min()), vmax=float(V[:, 2].max()))


def apply_dent(V, N, cfg, rng):
    M = V.shape[0]
    cv = int(rng.integers(0, M))
    r = float(rng.uniform(0.15, 0.25))
    depth = float(rng.uniform(0.003, 0.008))
    d = np.linalg.norm(V - V[cv], axis=1)
    m = d < r
    disp = np.zeros(M)
    disp[m] = -depth * np.cos(0.5 * np.pi * np.clip(d[m] / r, 0.0, 1.0))
    lab = np.zeros(M, dtype=np.int64); lab[m] = DENT
    meta = {"type": DENT, "type_name": TYPE_NAMES[DENT], "center_vertex": int(cv),
            "radius_m": r, "depth_mm": depth * 1000.0, "n_damaged_vertices": int(m.sum())}
    return disp, lab, meta


def apply_lee(V, N, cfg, rng):
    ctx = _blade_ctx(V, N, cfg)
    M = V.shape[0]
    ns = cfg.n_sections
    r0 = int(rng.integers(2, ns - 4))
    maxw = max(3, min(12, ns - r0 - 1))
    r1 = r0 + int(rng.integers(3, maxw + 1))
    band = float(rng.uniform(0.05, 0.08))
    depth = float(rng.uniform(0.002, 0.005))
    Z = ctx["Z"]; ringZ = ctx["ringZ"]
    lo = np.where((Z >= ringZ[r0]) & (Z <= ringZ[r1]) & (ctx["le_off"] < band))[0]
    rough = rng.uniform(0.0, 1.0, len(lo)) * 0.30
    disp = np.zeros(M); lab = np.zeros(M, dtype=np.int64)
    disp[lo] = -depth * (1.0 + rough); lab[lo] = LEE
    meta = {"type": LEE, "type_name": TYPE_NAMES[LEE], "span_start_m": float(ringZ[r0]),
            "span_end_m": float(ringZ[r1]), "le_band_m": band, "depth_mm": depth * 1000.0,
            "n_damaged_vertices": int(lo.size)}
    return disp, lab, meta


def apply_crack(V, N, cfg, rng):
    ctx = _blade_ctx(V, N, cfg)
    M = V.shape[0]
    xc = float(rng.uniform(-0.1, 0.1))
    hw = float(rng.uniform(0.008, 0.015))
    depth = float(rng.uniform(0.002, 0.006))
    X = ctx["X"]
    m = np.abs(X - xc) < hw
    widen = 1.0
    while m.sum() < 8 and widen < 20.0:      # keep crack representable at mesh resolution
        widen *= 4.0
        m = np.abs(X - xc) < hw * widen
    disp = np.zeros(M); lab = np.zeros(M, dtype=np.int64)
    disp[m] = -depth; lab[m] = CRACK
    meta = {"type": CRACK, "type_name": TYPE_NAMES[CRACK], "x_line_m": xc,
            "half_width_m": hw * widen, "depth_mm": depth * 1000.0,
            "n_damaged_vertices": int(m.sum())}
    return disp, lab, meta


def apply_deform(V, N, cfg, rng):
    ctx = _blade_ctx(V, N, cfg)
    M = V.shape[0]
    ns = cfg.n_sections
    r0 = int(rng.integers(2, ns - 3))
    maxw = max(3, min(6, ns - r0 - 1))
    r1 = r0 + int(rng.integers(3, maxw + 1))
    ringZ = ctx["ringZ"]; a, b = float(ringZ[r0]), float(ringZ[r1])
    amp = float(rng.uniform(0.004, 0.010))
    Z = ctx["Z"]
    m = (Z >= a) & (Z <= b)
    s = (Z[m] - a) / max(b - a, 1e-9)
    disp = np.zeros(M); lab = np.zeros(M, dtype=np.int64)
    disp[m] = amp * np.sin(np.pi * s); lab[m] = DEFORM
    meta = {"type": DEFORM, "type_name": TYPE_NAMES[DEFORM], "span_start_m": a,
            "span_end_m": b, "amp_mm": amp * 1000.0, "n_damaged_vertices": int(m.sum())}
    return disp, lab, meta


def apply_healthy(V, N, cfg, rng):
    M = V.shape[0]
    return (np.zeros(M), np.zeros(M, dtype=np.int64),
            {"type": HEALTHY, "type_name": TYPE_NAMES[HEALTHY], "n_damaged_vertices": 0})


def generate_damage(V, N, cfg, rng, healthy_prob=0.15):
    if rng.random() < healthy_prob:
        return apply_healthy(V, N, cfg, rng)
    t = int(rng.choice([LEE, CRACK, DENT, DEFORM]))
    if t == LEE:
        return apply_lee(V, N, cfg, rng)
    if t == CRACK:
        return apply_crack(V, N, cfg, rng)
    if t == DENT:
        return apply_dent(V, N, cfg, rng)
    return apply_deform(V, N, cfg, rng)