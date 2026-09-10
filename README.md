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
- Scene-level engineering metrics for inspectors: damage area (m2), volume (cm3),
  centroid localization error (per damage type).
- Swappable deep backbones behind `--backbone`: PointNet / PointNet++-style /
  DGCNN / PointTransformer (same benchmark, same protocol).
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
    scene_metrics.py         scene-level area / volume / localization metrics
  learned/
    features.py              per-point feature extraction (12-d) + PointNet input (7-d)
    models.py                RF / GBM / MLP wrappers
  deep/                      PyTorch (run on a GPU box; see deep/README.md)
    dataset.py               in-memory dataset + per-scene kNN precompute
    pointnet.py              PointNet-style per-point segmentation (+ depth head)
    backbones.py             pointnet / pointnetpp / dgcnn / pointtransformer
    run_pointnet.py          train/eval driver (--backbone), same protocol
  run_baseline.py            driver: classical baseline over N scenes
  run_learned.py             driver: learned classifier vs baseline (head-to-head)
  run_scene_metrics.py       driver: scene-level area/vol/localization of the baseline
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
python run_scene_metrics.py --num 1000 --points 20000 --tau 2.0
python deep/run_pointnet.py --num 1000 --epochs 40 --backbone dgcnn        # GPU box
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
| learned RF | lee_erosion | 0.881 | 0.675 | 0.765 | 0.619 |
| learned GBM | lee_erosion | 0.853 | 0.648 | 0.737 | 0.583 |
| baseline tau=2.0 | crack | 0.319 | 0.549 | 0.404 | 0.253 |
| learned RF | crack | 0.851 | 0.897 | 0.874 | 0.776 |
| learned GBM | crack | 0.836 | 0.901 | 0.868 | 0.766 |
| baseline tau=2.0 | dent | 0.197 | 0.511 | 0.284 | 0.166 |
| learned RF | dent | 0.928 | 0.829 | 0.876 | 0.779 |
| learned GBM | dent | 0.916 | 0.807 | 0.858 | 0.752 |
| baseline tau=2.0 | deformation | 0.698 | 0.827 | 0.757 | 0.609 |
| learned RF | deformation | 0.991 | 0.941 | 0.965 | 0.932 |
| learned GBM | deformation | 0.990 | 0.936 | 0.962 | 0.927 |
| baseline tau=2.0 | **MICRO** | 0.385 | 0.716 | **0.501** | 0.334 |
| baseline tau=2.0 | **MICRO** | 0.385 | 0.716 | **0.501** | 0.334 |
| learned RF | **MICRO** | 0.937 | 0.886 | **0.911** | 0.836 |
| learned GBM | **MICRO** | 0.928 | 0.878 | **0.902** | 0.822 |

RF and GBM agree closely (GBM micro-F1 0.902), so the gain is a property of the
context features, not of one particular tree-based implementation.

Healthy scenes (false-alarm check): 23,361 FP points with the baseline vs 256 with the
learned RF - the classifier suppresses LE/TE boundary artefacts almost completely.
Geometric MAE/RMSE is identical for both rows of each type by construction (both read
the same deviation field); a genuine depth improvement requires a regression head
(see `deep/`, `--depth-head`).

## The deep method (GPU)
`deep/` trains per-point segmentation networks on the exact same scenes and split as
`run_learned.py`, so its test numbers drop into the same table. Four swappable
backbones share one interface (7-d per-point features [aligned xyz, normal, dev_mm]
+ precomputed scene-local kNN + valid-point mask; optional `--depth-head`):

| backbone | description |
|---|---|
| `pointnet` | classic shared-MLP PointNet + global context (ignores kNN) |
| `pointnetpp` | fixed-resolution 2-stage PointNet++ w/ kNN max-pool context |
| `dgcnn` | 2-layer dynamic-graph CNN (EdgeConv) |
| `pointtransformer` | 2-layer kNN relative-position self-attention |

All report point-level P/R/F1/IoU plus the scene-level engineering metrics.
See `deep/README.md` (requires `pip install torch`, run on your GPU box).

## Scene-level engineering metrics (where / how big / how much)
Point-level F1 says whether the points look right; an inspector also needs WHERE the
damage is, HOW BIG it is (m2), and HOW MUCH material is affected (cm3,
shallow-defect approximation V = sum depth_i x a_w, a_w = S_mesh / n).

- `run_scene_metrics.py` reports these for the classical baseline at any tau.
- `run_learned.py` and `deep/run_pointnet.py` report the same quantities per damage
  type for both the learned mask and the baseline mask: mean GT/predicted area (m2)
  + MAE, mean GT/predicted volume (cm3) + MAE, centroid localization error (m) and
  the fraction of scenes localized within 0.5 m.

**Classical baseline, tau = 2.0 mm, 1000 scenes:**

| type | n | area gt (m2) | area pred (m2) | area MAE (m2) | vol gt (cm3) | vol pred (cm3) | vol MAE (cm3) | centroid err (m) | loc OK (<0.5 m) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| lee_erosion | 218 | 0.176 | 0.513 | 0.337 | 601 | 2180 | 1579 | 0.992 | 22% |
| crack | 207 | 0.353 | 0.597 | 0.245 | 916 | 2324 | 1408 | 0.163 | 100% |
| dent | 242 | 0.200 | 0.507 | 0.307 | 473 | 2082 | 1609 | 1.064 | 21% |
| deformation | 192 | 0.866 | 1.072 | 0.206 | 3987 | 5203 | 1215 | 0.470 | 65% |
| healthy | 141 | 0.000 | 0.403 | 0.403 | 0 | 1711 | - | - | - |

**Learned RF (decision threshold tuned on train; 199 test scenes):**

| type | n | area gt (m2) | area pred (m2) | area MAE (m2) | vol gt (cm3) | vol pred (cm3) | vol MAE (cm3) | centroid err (m) | loc OK (<0.5 m) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| lee_erosion | 44 | 0.174 | 0.133 | 0.058 | 610 | 506 | 236 | 0.077 | 95% |
| crack | 41 | 0.355 | 0.374 | 0.020 | 934 | 888 | 61 | 0.046 | 100% |
| dent | 48 | 0.193 | 0.173 | 0.026 | 450 | 430 | 56 | 0.055 | 98% |
| deformation | 38 | 0.951 | 0.902 | 0.048 | 4342 | 4129 | 213 | 0.009 | 100% |
| healthy | 28 | 0.000 | 0.004 | 0.004 | 0 | 14 | - | - | - |

The global threshold over-predicts area/volume by 2-4x (LE/TE artefacts dilute the
mask) and mislocalizes patch-like damage (lee erosion, dent) by ~0.9 m (only 29-34%
of scenes within 0.5 m) because the artefact band, not the damage, dominates the
predicted mask. The context-aware learned decision fixes all three: area MAE drops
~6-12x (e.g. crack 0.255 -> 0.020 m2), volume MAE drops ~6-7x (e.g. dent
1600 -> 56 cm3), localization improves from 0.88/0.88 m to 0.08/0.06 m
(95-100% of scenes within 0.5 m), and the healthy-scene false alarm shrinks from
0.40 m2 / 1704 cm3 to 0.004 m2 / 14 cm3 (~100x).

## Outputs
- `results/baseline_<N>.csv`, `results/baseline_<N>_summary.csv`, `results/tau_sweep.csv`.
- `results/learned_<N>_<model>_comparison.csv` - per-type P/R/F1/IoU + micro, baseline vs learned (one file per model: rf / gbm / mlp).
- `results/scene_metrics_baseline_<N>.csv` - baseline scene-level area/vol/localization.
- `results/learned_<N>_<model>_scene.csv` - learned vs baseline scene metrics (per model).
- `results/<backbone>_<N>_comparison.csv`, `_scene.csv`, `_config.json`, `_ckpt.pt` (deep, per backbone).
- `results/config.json`, `results/learned_<N>_<model>_config.json`, `results/pointnet_<N>_config.json`.
- `paper/figs/deviation_map.png`, `paper/figs/f1_by_type.png`, `paper/figs/learned_vs_baseline_<model>.png`.

## Extending to deep / multimodal methods
The NPZ format (points, normals, GT mask, true depth, per-scene metadata + the shared
`reference.npz`) is a clean input for 3D backbones (PointNet++, DGCNN, PointTransformer)
or a CAD-difference representation. Feed the deviation field, the scan, and/or the
reference difference as features; supervise with the same point-level GT. The data
generator is independent of the baseline, so a learned method can be dropped in without
changing it. A natural next step is fusing the 3D deviation channel with 2D thermal /
visual imagery (multimodal) - the per-scene GT is shared, so fusion can be supervised
with the exact same labels.

## Paper figures
`python paper/make_figs.py` regenerates scenes with the exact run_learned.py seed-0
protocol and writes `paper/figs/`:
- `fig1_damage_gallery.png` (4 damage types, scan + GT overlay, full + zoom)
- `fig2_deviation_field.png` (signed ICP deviation field, LE/TE artifact bands)
- `fig4_scene_quantification.png` (scene-level area / volume / localization errors)
- `fig7_tau_sweep.png` (baseline tau sweep P/R/F1)
- `fig3_detection_comparison.png`, `fig5_fp_suppression.png`,
  `fig6_feature_importance.png` (require `results/learned_<N>_rf_model.joblib`)

## Robustness + ablation (CPU)
- `python run_robustness.py --num 200` -- one RF trained on nominal-sensor train
  split, then the SAME damage realizations re-scanned under 8 sensor settings
  (noise 0.5/1.0 mm, dropout 0.3/0.5, angle 1.0/2.0 deg, trans 15 mm);
  baseline + RF evaluated on the test split. -> `results/robustness_<N>.csv`
- `python run_ablation.py --num 120` -- progressive 12-d feature ablation
  (raw deviation -> local stats -> coherence -> location -> normals -> edge).
  -> `results/ablation_<N>.csv`

## Self-supervised pretraining (GPU box)
```
python deep/pretrain.py --num 1000 --epochs 30 --backbone dgcnn
python deep/run_pointnet.py --num 1000 --backbone dgcnn \
    --init-from results/pretrain_dgcnn_1000.pt
```
Pretraining target: |deviation| (mm) on ALL points of ALL scenes (no GT label
consumed); the resulting weights initialize supervised fine-tuning.

## License
MIT. See `LICENSE`.