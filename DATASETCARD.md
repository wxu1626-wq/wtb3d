# Dataset Card: wtb3d - Synthetic 3D Point-Cloud WTB Damage Dataset

## 1. Overview
| Field | Value |
|---|---|
| Dataset name | wtb3d |
| Version | 0.1.0 |
| License | MIT |
| Task | 3D point-cloud surface-damage detection + quantification on wind-turbine blades |
| Setting | CAD-assisted inspection (dense undamaged CAD reference vs a sparse registered scan) |
| Modality | 3D point cloud (positions + normals); per-point GT (mask + true depth) |
| Damage types | healthy, lee_erosion, crack, dent, deformation |
| Source | Fully synthetic (deterministic procedural generator) |
| Default size | 1,000 scenes x 20,000 points/scan; 20,000-point shared reference |
| Committed sample | 12 scenes in `data/sample/` |

**Motivation.** Wind-turbine-blade damage (leading-edge erosion, cracks, dents,
delamination/deformation) is a safety-critical inspection problem, but most public
datasets are 2D (RGB / thermal) images. 3D inspection with a CAD reference is realistic
for modern blade monitoring (lidar / structured light / photogrammetry), yet there is no
small, reproducible 3D benchmark with *per-point geometric* ground truth. wtb3d fills
that gap: a deterministic blade + damage generator, a physical sensor model, and a
classical baseline (ICP -> signed distance -> threshold) that a deep 3D / multimodal
method must beat.

## 2. What the data represents
- **Blade geometry:** a lofted airfoil surface. 40 spanwise sections x 48 airfoil points,
  NACA-4 profile, span 5.0 m, root chord 1.2 m, tip chord 0.4 m, twist 8 deg. Vertices
  lie on the lofted surface with analytically estimated outward normals.
- **Reference surface (`reference.npz`):** a dense undamaged CAD surface (20,000 points +
  normals). Built once and shared by every scene; the baseline builds one KD-tree over it.
- **Scene (`scene_i.npz`):** a damaged, sensor-simulated scan of the blade plus
  point-level ground truth.

## 3. Damage models
Damage is a per-vertex displacement along the outward surface normal (negative =
material removed, positive = bulge), restricted to a local region. Spans are chosen by
**ring index**, so every region is guaranteed non-empty at the mesh resolution.

| Type | ID | Mechanism | Region | Parameters (random range) |
|---|---|---|---|---|
| healthy | 0 | none | - | - |
| lee_erosion | 1 | LE material loss | span interval, `band` off LE | band 5-8 cm; depth 2-5 mm; +30% roughness |
| crack | 2 | spanwise groove | chord line near mid-chord | line x in [-0.1, 0.1] m; half-width 8-15 mm; depth 2-6 mm |
| dent | 3 | spherical dimple | disc of radius `r` | r 15-25 cm; depth 3-8 mm (cosine falloff) |
| deformation | 4 | outward bulge | span interval | amplitude 4-10 mm (sinusoidal over span) |

About 15% of scenes are healthy; the remainder are drawn uniformly from the four damage
types. In the 1,000-scene reference run the class counts are: dent 242, lee_erosion 218,
crack 207, deformation 192, healthy 141.

## 4. Sensor simulation (`to_scan`)
Applied to the damaged blade to produce a realistic scan:
- **Sampling:** `points` points sampled on the damaged surface (barycentric on faces).
- **Noise:** zero-mean Gaussian, std 0.2 mm, on each coordinate.
- **Dropout:** 15% of points randomly dropped (missing returns).
- **Mis-registration:** a small random rigid pose (rotation up to 0.3 deg, translation up
  to 5 mm), so the scan is *not* perfectly aligned to the reference and ICP must recover it.
- **Occlusion:** optional single-sided (visible-only) view, off by default (full-view scan).

## 5. File format
`reference.npz`
| key | shape | dtype | meaning |
|---|---|---|---|
| `points` | (n_ref, 3) | float32 | dense CAD surface (m) |
| `normals` | (n_ref, 3) | float32 | per-point outward normals |

`scene_<id>.npz`
| key | shape | dtype | meaning |
|---|---|---|---|
| `scanned` | (n, 3) | float32 | scan points (m) |
| `normals` | (n, 3) | float32 | scan normals |
| `gt_mask` | (n,) | bool | True = point on damage (\|dev\| > gt_eps) |
| `gt_depth_mm` | (n,) | float32 | true damage depth at that point (mm) |
| `meta` (json) | - | - | type, type_name, pose R/t, damage params, n_damaged_vertices |

`manifest.csv`: one row per scene with id, type, type_name, n_damaged_vertices,
n_gt_points, file.

`gt_eps_mm = 0.6` mm defines the ground-truth damage threshold: a point whose deviation
from the undamaged CAD exceeds this is labelled damaged.

## 6. Splits
Single dataset, one scene per blade state; there is no shared identity across scenes, so
any scene can be used for train / val / test. Recommended: stratified 80/10/10 by damage
type over the 1,000-scene set. The committed 12-scene sample is for smoke-testing only,
not a training set.

## 7. Classical baseline and reference numbers
Pipeline: ICP (point-to-point Kabsch) -> point-to-plane signed distance -> k=3 nearest-
neighbour smoothing -> threshold \|dev\| > tau.

Reference run: 1,000 scenes, 20,000 reference points, 20,000 points/scan, seed 0, scipy
ICP fallback (Open3D not installed). **Best tau = 2.0 mm, micro-F1 = 0.489.**

Per damage type at tau = 2.0 mm:
| Type | n | Precision | Recall | F1 | IoU | Geom. MAE (mm) | Geom. RMSE (mm) |
|---|---|---|---|---|---|---|---|
| lee_erosion | 218 | 0.266 | 0.773 | 0.395 | 0.246 | 0.99 | 1.36 |
| crack | 207 | 0.313 | 0.531 | 0.394 | 0.245 | 0.66 | 0.85 |
| dent | 242 | 0.209 | 0.529 | 0.300 | 0.176 | 0.41 | 0.78 |
| deformation | 192 | 0.667 | 0.825 | 0.738 | 0.584 | 0.44 | 0.97 |
| healthy | 141 | - | - | - | - | - | - |

Interpretation: gross deformation is detected strongly (F1 0.74). Thin / shallow damage
(LEE, crack, dent) is *found* (recall 0.53-0.77) but with many false positives
(precision 0.21-0.31) - the core weakness a learned method should fix. The healthy-region
|deviation| noise floor is ~1.94 mm mean per-scene p95 (max 3.48 mm), which is why a
fixed threshold over-fires on thin damage.

Micro-F1 vs tau:
| tau (mm) | Precision | Recall | F1 |
|---|---|---|---|
| 0.5 | 0.222 | 0.973 | 0.362 |
| 1.0 | 0.320 | 0.902 | 0.472 |
| 2.0 | 0.373 | 0.709 | 0.489 |
| 3.0 | 0.404 | 0.531 | 0.459 |
| 5.0 | 0.447 | 0.247 | 0.318 |

Caveats:
- The point-to-point ICP NN objective settles at ~10.0 mm, which is *sampling-limited*
  (20k-point spacing), not the true sub-mm mis-registration; point-to-plane ICP
  (Open3D) would register tighter.
- Metrics are point-level (per sampled point), the natural protocol for dense 3D damage
  segmentation.

## 8. Supported tasks
- **Detection / segmentation:** binary (damaged / not) point-level classification.
- **Multi-class:** 5-class point labelling (healthy + 4 damage types).
- **Quantification:** regress per-point depth (mm) on damage.
- **CAD-difference:** any method that consumes scan + reference, or their deviation field.

## 9. Limitations
- Synthetic: damage morphology and sensor statistics are idealized; real scans add
  texture, scale variation, and richer noise.
- Single blade shape / scale (5 m); no cross-blade generalization stress test.
- Default full-view scan; occlusion is available but untested in the reference run.
- Ground-truth assumes an exact undamaged CAD; real CAD may differ from the as-built blade.

## 10. Evaluation protocol for learned methods (run_learned.py / deep/)
- Scene-level stratified 80/20 train/test split by damage type (split seed 2024);
  for 1000 scenes this is 801 train / 199 test.
- Baseline tau and the learned decision threshold are tuned on TRAIN scenes only,
  then frozen for the TEST evaluation (no test-set leakage).
- The deep model additionally uses a 10% validation slice of the train scenes for
  threshold selection and early stopping.
- Scene i of a run is a deterministic function of the global seed: the same
  (num, points, seed) always yields the same scenes for every method.

## 11. Attribution

Open research software (MIT). If used in a paper, cite the associated manuscript and add a
software citation to this repository.