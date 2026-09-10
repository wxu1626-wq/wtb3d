# W1-W18 Roadmap vs. Repo Status

Tracks the 18-week plan from the research proposal against concrete artifacts.
Legend: [done] / [in progress] / [not started].

| Wk | Milestone | Artifact(s) | Status |
|----|-----------|-------------|--------|
| W1 | Blade CAD + 4 damage generators | `data_gen/geometry.py`, `data_gen/damage.py` | [done] |
| W2 | Sensor sim (noise, dropout, misregistration, occlusion) | `data_gen/sensor_sim.py` | [done] |
| W3 | 1000-scene library + dataset card + sample data | `data_gen/generate.py`, `DATASETCARD.md`, `data/sample/` | [done] |
| W4 | Classical deviation field + ICP baseline | `baselines/`, `run_baseline.py` | [done] |
| W5 | Context features + RF/GBM/MLP | `learned/`, `run_learned.py` | [done] |
| W6 | Deep backbones (pointnet/pp/dgcnn/pt) + deviation-field pretraining | `deep/backbones.py`, `deep/pretrain.py`, `run_pointnet.py --init-from` | [in progress] code ready; GPU training pending |
| W7 | Dual head (detection + depth) | `deep/run_pointnet.py --depth-head --depth-w` | [in progress] code ready; GPU training pending |
| W8 | Scene-level quantification (area / volume / localization) | `baselines/scene_metrics.py`, `run_scene_metrics.py` | [done] |
| W9 | Full training runs on GPU box | user machine (torch) | [not started] |
| W10 | Baseline battery (classical + learned + 4 deep backbones) | `results/*_comparison.csv`, `deep/test_backbones.py` | [in progress] classical/RF/GBM done; deep runs pending GPU |
| W11 | Feature ablation | `run_ablation.py`, `results/ablation_120.csv` | [done] |
| W12 | Sensor-robustness study | `run_robustness.py`, `results/robustness_200.csv` | [done] |
| W13 | Visualization suite | `paper/make_figs.py`, `paper/figs/` | [done] all 7 figs; regenerate via `python paper/make_figs.py` |
| W14 | Real-data validation (30-50 scenes, handheld structured light or public cross-domain set) | TBD | [not started] |
| W15 | Paper draft | `paper/` | [not started] |
| W16 | Multi-seed statistics + significance tests | - | [not started] |
| W17 | Extra experiments (2D-YOLO cross-view control, anomaly-detection baselines) | - | [not started] |
| W18 | Submission prep (cover letter, rebuttal notes) | - | [not started] |

## Headline numbers so far (1000 scenes, 801 train / 199 test)

Point-level (micro-F1): baseline tau=2.0mm 0.482 (tau sweep) / RF 0.911 / GBM 0.902.
Per-type RF F1: lee 0.765, crack 0.874, dent 0.876, deform 0.965.

Scene-level (test): baseline area MAE 0.18-0.34 m2, volume MAE 1123-1600 cm3,
localization 0.16-0.88 m (lee/dent only 29-34% within 0.5 m), healthy false
alarm 0.40 m2 / 1704 cm3. RF: area MAE 0.020-0.058 m2, volume MAE 56-236 cm3,
localization 0.009-0.077 m (95-100% within 0.5 m), healthy false alarm
0.004 m2 / 14 cm3.

## Feature ablation (120 scenes, progressive subsets, RF)

A raw dev [dev_mm, abs] 0.417 (worse than the classical 0.482!);
B +local stats 0.666; C +coherence 0.666; D +location (x, z) 0.868;
E +normals 0.891; F full 12-d 0.889.
Interpretation: the gain is NOT raw deviation magnitude but context -
local statistics + location-awareness kills the LE/TE artifact bands;
normals give the last small step; 12-d does not overfit.

## Sensor robustness (200 scenes; RF trained on nominal sensor only;
tau/thr tuned on nominal train; same damage realizations re-scanned)

| setting   | baseline | RF   |
|-----------|----------|------|
| nominal   | 0.468 | 0.889 |
| noise 0.5 | 0.468 | 0.802 |
| noise 1.0 | 0.458 | 0.301 |
| dropout 0.3 | 0.467 | 0.882 |
| dropout 0.5 | 0.455 | 0.869 |
| angle 1.0 | 0.422 | 0.697 |
| angle 2.0 | 0.365 | 0.407 |
| trans 15  | 0.452 | 0.879 |

Interpretation: RF is robust to moderate scan degradation (dropout,
translation, mild noise/angle) but degrades beyond its training sensor
distribution (noise 1.0 -> 0.301, angle 2.0 -> 0.407, both below the
classical line). The baseline looks "flat" only because its global
2 mm threshold is coarsely over-detecting (it never had correct F1).
Honest paper finding: context-aware detection trades a large nominal
gain for OOD-sensor sensitivity; fix = sensor-noise augmentation in
training (future work / optional experiment).

## Remaining concrete tasks (in suggested order)

1. GPU box: `pip install torch`, then per backbone
   `python deep/run_pointnet.py --num 1000 --backbone <b>` and the pretrain
   + fine-tune pair with `--init-from`. Compare micro-F1 + scene metrics
   against the RF row (is deep > hand-crafted features? by how much?).
2. Optional: RF with sensor-noise augmentation (train on nominal + noise/
   angle mix) to restore OOD robustness; extend `run_robustness.py`.
3. Real-data validation: acquire or curate 30-50 real scan scenes; at minimum
   run the frozen classical + RF pipeline on them (zero training on real data).
4. Multi-seed: repeat RF + one deep backbone with seeds 1/2/3; report mean+std.
5. Optional: I1 weak-registration dual-encoding backbone (cross-attention over
   scan + nominal), anomaly baselines (PatchCore-style), 2D-YOLO cross-view
   control to justify the 3D modality.
