# wtb3d - 3D Point-Cloud Dataset & Classical Baseline for Wind-Turbine-Blade Damage

A small, self-contained, reproducible 3D point-cloud benchmark for wind-turbine-blade
(WTB) surface-damage detection, plus a classical reference baseline
**ICP registration -> point-to-plane signed distance -> threshold**.

The setting is a realistic *CAD-assisted inspection* task: one dense, undamaged CAD
surface of the blade is available, and the goal is to detect and quantify local surface
damage (erosion / crack / dent / deformation) from a **sparse, noisy, slightly
mis-registered** scan of the same blade. This classical pipeline is the baseline that
any deep 3D / multimodal method should outperform.

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
  run_baseline.py            driver: run the baseline over N scenes, write results + figs
  data/sample/               committed 12-scene sample + reference.npz + manifest.csv
  results/                   (generated) baseline CSVs + config.json
  paper/figs/                (generated) deviation map + per-type F1 bars
  DATASETCARD.md             full dataset card
  requirements.txt
```

## Installation
Requires Python >= 3.10.
```
pip install -r requirements.txt        # numpy, scipy, matplotlib
pip install open3d                     # optional; enables the native ICP path
```
Without Open3D the code is fully functional (it falls back to an equivalent scipy ICP).

## Quick start
Run the classical baseline on the committed 12-scene sample (a few seconds):
```
python run_baseline.py --num 12
```
Generate a larger dataset, then run the baseline on it:
```
python -m data_gen.generate --num 1000 --points 20000 --out data/full1k
python run_baseline.py --num 1000 --points 20000
```
Each run builds its own shared reference, prints a per-damage-type summary, and saves
CSVs under `results/` and figures under `paper/figs/`.

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
low precision; gross deformation: strong).

## Outputs
- `results/baseline_<N>.csv` - one row per (scene, tau).
- `results/baseline_<N>_summary.csv` - per-damage-type P/R/F1/IoU + geometric MAE/RMSE at the best tau.
- `results/tau_sweep.csv` - micro-F1 vs tau.
- `results/config.json` - run settings + whether Open3D was used.
- `paper/figs/deviation_map.png`, `paper/figs/f1_by_type.png`.

## Extending to deep / multimodal methods
The NPZ format (points, normals, GT mask, true depth, per-scene metadata + the shared
`reference.npz`) is a clean input for 3D backbones (PointNet++, DGCNN, PointTransformer)
or a CAD-difference representation. Feed the deviation field, the scan, and/or the
reference difference as features; supervise with the same point-level GT. The data
generator is independent of the baseline, so a learned method can be dropped in without
changing it.

## License
MIT. See `LICENSE`.