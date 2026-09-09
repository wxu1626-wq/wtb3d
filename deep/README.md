# Deep model (PyTorch) - run on your GPU box

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
python deep/run_pointnet.py --num 1000 --epochs 40          # base
python deep/run_pointnet.py --num 1000 --epochs 40 --depth-head   # + depth regression
```

First run regenerates all scenes in-process (~15 min CPU); afterwards training
takes ~10-30 min on a modern GPU (batch=16, ~20k pts/scene).

## Method

Model: PointNet-style per-point segmentation
`shared Conv1d MLP -> global max-pool context -> per-point sigmoid head`
(+ optional L1 depth head on |deviation|, mm).

Per-point input (7-d): ICP-aligned xyz (m), scanned surface normal, signed
point-to-plane deviation (mm) -- the same measurement the classical baseline
thresholds on. The network replaces the global threshold with a
location/neighbourhood-aware decision; BCE loss with pos_weight for the
~5% positive rate.

## Protocol (identical to run_learned.py)

- same seeds -> same 1000 scenes
- scene-level stratified split 80/20 by damage type
- decision threshold tuned on a 10% validation slice of the train scenes;
  baseline tau tuned on train only; both frozen on the test set
- point-level P/R/F1/IoU per damage type + micro-average; geometric MAE/RMSE
  vs true depth (shared deviation field, same for both methods)

Outputs: `results/pointnet_<num>_comparison.csv`, `..._config.json`,
`..._ckpt.pt`.

## Notes for the paper

- The learned (CPU) and deep (GPU) methods are complementary baselines for
  the "learned decision rule" contribution; report both in the comparison
  table with the classical ICP threshold.
- Depth MAE/RMSE is identical for any method that only changes the binary
  decision from the same deviation field; a genuine depth improvement
  requires the `--depth-head` regression (trained on GT depth, mm).