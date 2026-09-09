# wtb3d - 3D Point-Cloud Dataset, Classical Baseline & Learned Methods for Wind-Turbine-Blade Damage

A small, self-contained, reproducible 3D point-cloud benchmark for wind-turbine-blade
(WTB) surface-damage detection, with:

- a **classical baseline**: ICP registration -> point-to-plane signed distance -> threshold
- a **learned (CPU) method**: scikit-learn per-point classifier on context features
- a **deep (GPU) scaffold**: PyTorch PointNet-style per-point segmentation (`deep/`)

The setting is a realistic *CAD-assisted inspection* task: one dense, undamaged CAD
surface of the blade is available, and the goal is to detect and quantify local surface
damage (erosion / crack / dent / deformation) from a **sparse, noisy, slightly
mis-registered** scan of the same blade. The classical pipeline is the baseline that
any learned / deep 3D / multimodal method should outperform.

> Research context: supporting material for a study on 3D point-cloud / multimodal WTB
> damage detection (follow-up to 2D object-detection work). All data here is synthetic
> but physically plausible; real scan + CAD data can be dropped in through the same I/O
> format (see `DATASETCARD.md`).

## Highlights
- Deterministic blade generator (lofted NACA-4 sections) + 4 physically plausible damage models.
- Sensor simulation: Gaussian noise, point dropout, small rigid mis-registration.
- Classical baseline that runs on **scipy alone**; Open3D is optional (an equivalent
  Kabsch point-to-point ICP runs automatically when Open3D is absent).
- Per-scene point-level ground truth (mask + true depth) enabling detection P/R/F1 and
  geometric MAE/RMSE, plus a threshold (tau) sweep and per-damage-type breakdown.
- Learned method (RF/GBM/MLP) with a fair head-to-head protocol: tau / decision
  threshold tuned on train scenes only, frozen evaluation on a held-out test set.
- A committed 12-scene sample in `data/sample/` so the whole pipeline is testable in seconds.

## Repository layout
```
wtb3d/
  data_gen/                  blade + damage + sensor-simulation data generator
    config.py                GenConfig (geometry, damage, sensor, GT thresholds)
    geometry.py              lofted NACA-4 blade mesh (vertices, faces, normals)
    damage.py                dent / LEE / crack / deformation generators
    sensor_sim.py            build_reference + to_scan (noise, dropout, pose)
    utils.py                 NPZ / PLY I/O, manifest helpers
    generate.py              CLI: build the persistent dataset + manifest
  baselines/
    classical_pipeline.py    ICP -> point-to-plane signed distance -> smoothing
    evaluate.py              point-level P/R/F1/IoU + geometric MAE/RMSE
  learned/
    features.py              per-point feature extraction (12-d) + PointNet input (7-d)
    models.py                RF / GBM / MLP wrappers
  deep/                      PyTorch (run on a GPU box; see deep/README.md)
    dataset.py               in-memory dataset over generated scenes
    pointnet.py              PointNet-style per-point segmentation (+ depth head)
    run_pointnet.py          train/eval driver, same protocol as run_learned.py
  run_baseline.py            driver: classical baseline over N scenes
  run_learned.py             driver: learned classifier vs baseline (head-to-head)
  data/sample/               committed 12-scene sample + reference.npz + manifest.csv
  results/                   (generated) baseline / learned CSVs + config.json
  paper/figs/                (generated) figures
  DATASETCARD.md             full dataset card
  requirements.txt
```

## Installation
Requires Python >= 3.10.
```
pip install -r requirements.txt        # numpy, scipy, matplotlib, scikit-learn
pip install open3d                     # optional; enables the native ICP path
pip install torch                      # optional; only for deep/ (GPU box)
```
Without Open3D the code is fully functional (it falls back to an equivalent scipy ICP).
Without torch everything outside `deep/` runs.

## Quick start
Run the classical baseline on the committed 12-scene sample (a few seconds):
```
python run_baseline.py --num 12
```
Full-scale runs (1000 scenes, ~15 min each on a 24-core CPU box):
```
python run_baseline.py --num 1000 --points 20000
python run_learned.py  --num 1000 --points 20000 --model rf --n-est 300
python run_learned.py  --num 1000 --points 20000 --model gbm --n-est 600   # ablation
```

## The data model
- **Reference (shared, built once):** a dense undamaged CAD surface of the blade
  (`n_ref` points + per-point normals). One `scipy.cKDTree` over it is reused across all
  scenes for nearest-neighbour / point-to-plane queries.
- **Scene (per blade state):** `to_scan` displaces the blade vertices along their outward
  normals by the per-vertex damage displacement, samples `points` points on the damaged
  surface, applies the sensor model (Gaussian noise, dropout, small random rigid pose),
  and returns scan points + normals, the point-level GT mask (|deviation| > gt_eps_mm),
  and the true per-point depth (mm).

## Damage types
| Type | Model | Typical parameters |
|---|---|---|
| `healthy` | none | - |
| `lee_erosion` | leading-edge material loss over a span interval | band 5-8 cm off LE, depth 2-5 mm |
| `crack` | spanwise groove at a fixed chordwise line | half-width 8-15 mm, depth 2-6 mm |
| `dent` | spherical inward dimple (FOD / hail) | radius 15-25 cm, depth 3-8 mm |
| `deformation` | localised outward bulge (delamination) | amplitude 4-10 mm |

Spans are chosen by ring index so every region is guaranteed non-empty. About 15% of
scenes are healthy. See `DATASETCARD.md` for exact ranges and the full sensor model.

## The classical baseline
Per scene, `run_baseline.py`:
1. **ICP** registers the sparse scan to the dense reference (point-to-point Kabsch;
   Open3D or the scipy fallback).
2. **Point-to-plane signed distance** measures each scan point's deviation from the CAD
   surface along the local reference normal.
3. **Nearest-neighbour smoothing** (k=3) denoises the deviation field.
4. **Threshold** `|deviation| > tau` yields the damage mask; performance is reported
   against the point-level GT, and the geometric error (recovered vs true depth) on
   detected damage.

A tau sweep reports the threshold that maximises micro-F1; the per-type breakdown
explains *where* the classical method struggles (thin / shallow damage: high recall,
low precision; gross deformation: strong). The persistent alignment + noise floor
(~2 mm, worst at LE/TE edges) creates many false positives that a fixed threshold
cannot remove.

## The learned method (CPU)
`run_learned.py` replaces the global threshold with a per-point classifier trained on
**12 context features** built from the *same* deviation field (fair upgrade, no new
sensors): signed/abs deviation, kNN(12) local mean/std/max, neighbour coherence
(fraction of neighbours above 0.5 mm - real damage is a connected patch while boundary
artefacts are isolated), chordwise/spanwise location, surface normal, local normal
dispersion, and distance to LE/TE edges.

**Protocol (fair head-to-head):** identical scenes; scene-level stratified 80/20 split
by damage type; baseline tau AND learned decision threshold tuned on TRAIN scenes only;
both frozen on the held-out TEST set (all points).

**Results (1000 scenes, 20k pts, seed 0; 199 test scenes):**

| method | type | P | R | F1 | IoU |
|---|---|--:|--:|--:|--:|
| baseline tau=2.0 | lee_erosion | 0.256 | 0.754 | 0.383 | 0.237 |
| learned RF | lee_erosion | 0.878 | 0.679 | 0.766 | 0.620 |
| baseline tau=2.0 | crack | 0.319 | 0.549 | 0.404 | 0.253 |
| learned RF | crack | 0.847 | 0.900 | 0.873 | 0.774 |
| baseline tau=2.0 | dent | 0.197 | 0.511 | 0.284 | 0.166 |
| learned RF | dent | 0.925 | 0.832 | 0.876 | 0.779 |
| baseline tau=2.0 | deformation | 0.698 | 0.827 | 0.757 | 0.609 |
| learned RF | deformation | 0.991 | 0.941 | 0.965 | 0.933 |
| baseline tau=2.0 | **MICRO** | 0.385 | 0.716 | **0.501** | 0.334 |
| learned RF | **MICRO** | **0.935** | 0.888 | **0.911** | 0.836 |

Healthy scenes (false-alarm check): 23,361 FP points with the baseline vs 256 with the
learned RF - the classifier suppresses LE/TE boundary artefacts almost completely.
Geometric MAE/RMSE is identical for both rows of each type by construction (both read
the same deviation field); a genuine depth improvement requires a regression head
(see `deep/`, `--depth-head`).

## The deep method (GPU)
`deep/` contains a PointNet-style per-point segmentation network (shared Conv1d MLP +
global max-pool context + per-point sigmoid head, optional depth-regression head)
trained on the 7-d per-point input (aligned xyz, normal, deviation). It uses the exact
same scenes and split as `run_learned.py`, so its test numbers drop into the same table.
See `deep/README.md` (requires `pip install torch`, run on your GPU box).

## Outputs
- `results/baseline_<N>.csv`, `results/baseline_<N>_summary.csv`, `results/tau_sweep.csv`.
- `results/learned_<N>_comparison.csv` - per-type P/R/F1/IoU + micro, baseline vs learned.
- `results/pointnet_<N>_comparison.csv` - same table, PointNet (GPU run).
- `results/config.json`, `results/learned_<N>_config.json`, `results/pointnet_<N>_config.json`.
- `paper/figs/deviation_map.png`, `paper/figs/f1_by_type.png`, `paper/figs/learned_vs_baseline.png`.

## Extending to deep / multimodal methods
The NPZ format (points, normals, GT mask, true depth, per-scene metadata + the shared
`reference.npz`) is a clean input for 3D backbones (PointNet++, DGCNN, PointTransformer)
or a CAD-difference representation. Feed the deviation field, the scan, and/or the
reference difference as features; supervise with the same point-level GT. The data
generator is independent of the baseline, so a learned method can be dropped in without
changing it. A natural next step is fusing the 3D deviation channel with 2D thermal /
visual imagery (multimodal) - the per-scene GT is shared, so fusion can be supervised
with the exact same labels.

## License
MIT. See `LICENSE`.