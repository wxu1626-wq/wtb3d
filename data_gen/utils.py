"""Lightweight I/O: binary PLY writer + npz reference/scene save/load (no external deps).

Dataset layout:
  <out>/reference.npz          dense undamaged CAD (nominal, nominal_nrm)
  <out>/scene_XXXX.npz         one scan (scanned, scanned_nrm, gt_mask, gt_depth_mm, type)
  <out>/scene_XXXX.meta.json   per-scene damage metadata + pose
  <out>/manifest.csv           index
"""
import os, json
import numpy as np


def write_ply(path, pts, nrm=None, colors=None):
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    cols = [pts[:, 0], pts[:, 1], pts[:, 2]]
    if nrm is not None:
        nrm = np.asarray(nrm, dtype=np.float32)
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        cols += [nrm[:, 0], nrm[:, 1], nrm[:, 2]]
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        cols += [colors[:, 0], colors[:, 1], colors[:, 2]]
    rec = np.empty(n, dtype=fields)
    for (nm, _), c in zip(fields, cols):
        rec[nm] = c
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
        for nm, dt in fields:
            hdr += "property %s %s\n" % (nm, "float" if dt == "<f4" else "uchar")
        hdr += "end_header\n"
        f.write(hdr.encode("ascii")); f.write(rec.tobytes())


def save_reference(path, nominal, nominal_nrm):
    np.savez_compressed(path, nominal=nominal, nominal_nrm=nominal_nrm)


def load_reference(path):
    d = np.load(path)
    return d["nominal"], d["nominal_nrm"]


def save_scene(path, scen, meta):
    np.savez_compressed(
        path,
        scanned=scen["scanned"], scanned_nrm=scen["scanned_nrm"],
        gt_mask=scen["gt_mask"], gt_depth_mm=scen["gt_depth_mm"],
        type=scen["type"])
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_scene(path, reference=None):
    d = np.load(path)
    scen = {k: d[k] for k in d.files}
    if reference is not None:
        scen["nominal"], scen["nominal_nrm"] = reference
    return scen