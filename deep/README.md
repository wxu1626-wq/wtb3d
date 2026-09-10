# Deep models (PyTorch) - run on your GPU box

The CPU environment of this repo has no torch/GPU, so this package is the
**deep-learning counterpart** of `run_learned.py` (scikit-learn). Both share
the identical data pipeline, so their numbers are directly comparable.

## Install

```bash
pip install torch          # CUDA build if you have an NVIDIA GPU
pip install -r requirements.txt
```

## Run

```bash
python deep/run_pointnet.py --num 1000 --epochs 40 --backbone pointnet
python deep/run_pointnet.py --num 1000 --epochs 40 --backbone pointnetpp
python deep/run_pointnet.py --num 1000 --epochs 40 --backbone dgcnn
python deep/run_pointnet.py --num 1000 --epochs 40 --backbone pointtransformer
python deep/run_pointnet.py --num 1000 --epochs 40 --backbone dgcnn --depth-head
```

First run regenerates all scenes in-process (~15 min CPU, includes per-scene
kNN precompute, k=16); afterwards training takes ~10-30 min on a modern GPU
(batch=16, ~20k pts/scene). If a GPU OOMs, lower `--batch`: the kNN neighbour
gather materialises (B, D, N, K) tensors.

## Method

All backbones share one interface: per-point input (7-d: ICP-aligned xyz (m),
scanned surface normal, signed point-to-plane deviation (mm) -- the same
measurement the classical baseline thresholds on) + scene-local kNN indices
(precomputed in `dataset.py`) + a valid-point mask. They output per-point
logits (BCE with pos_weight for the ~5% positive rate), optionally plus an L1
depth head regressing |deviation| (mm) on GT-damaged points.

| backbone | description |
|---|---|
| `pointnet` | classic shared-MLP PointNet + global max-pool context (ignores kNN) |
| `pointnetpp` | fixed-resolution 2-stage PointNet++-style; each stage augmented with the kNN max-pool neighbourhood (SA-module stand-in, no random sampling) |
| `dgcnn` | 2-layer dynamic-graph CNN (EdgeConv on [x_i, x_j - x_i], max-pool) |
| `pointtransformer` | 2-layer kNN relative-position self-attention (Q = own feature, K = neighbour + rel-pos, V = neighbour) + global context |

The network replaces the global threshold with a location/neighbourhood-aware
decision.

## Protocol (identical to run_learned.py)

- same seeds -> same 1000 scenes
- scene-level stratified split 80/20 by damage type
- decision threshold tuned on a 10% validation slice of the train scenes;
  baseline tau tuned on train only; both frozen on the test set
- point-level P/R/F1/IoU per damage type + micro-average; geometric MAE/RMSE
  vs true depth (shared deviation field, same for both methods)
- scene-level engineering metrics (area m2 / volume cm3 / localization) for
  both the learned mask and the baseline mask

Outputs (per backbone): `results/<backbone>_<num>_comparison.csv`,
`..._scene.csv`, `..._config.json`, `..._ckpt.pt`.

## Notes for the paper

- The learned (CPU) and deep (GPU) methods are complementary baselines for
  the "learned decision rule" contribution; report both in the comparison
  table with the classical ICP threshold.
- Depth MAE/RMSE is identical for any method that only changes the binary
  decision from the same deviation field; a genuine depth improvement
  requires the `--depth-head` regression (trained on GT depth, mm).