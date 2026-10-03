# Calibrated safety assessment of offset train collisions

This repository contains cleaned analysis code for the calibrated two-layer assessment framework used in the manuscript. The release focuses on the Deep Ensemble / Joint Conformal Prediction workflow for energy-absorption capacity utilization and the amplitude-normalized conformal calibration used for Layer-1 stability screening.

The FE database is not included in the repository. Data supporting the study can be provided separately according to the manuscript's data-availability statement. See `data/README.md` for the expected file formats.

## Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Train the Layer-2 Deep Ensemble

```bash
python 01_train_deep_ensemble.py \
  --train-x data/layer2_train_X.csv \
  --train-y data/layer2_train_Y.csv \
  --val-x data/layer2_validation_X.csv \
  --val-y data/layer2_validation_Y.csv
```

The default implementation uses five independently initialized members, hidden layers `256-256-128`, a mean head and Softplus variance head, Adam with an initial learning rate of `1e-3`, cosine-annealing scheduling, and member-specific dropout sampled reproducibly from `[0.05, 0.15]`.

### 2. Calibrate Independent CP and Joint CP

Use only the independent Layer-2 calibration subset that remains calibrated stable after Layer-1 screening.

```bash
python 02_calibrate_layer2_joint_cp.py \
  --cal-x data/layer2_cal_stable_X.csv \
  --cal-y data/layer2_cal_stable_Y.csv
```

The finite-sample corrected order statistic is used for all conformal quantiles. Joint CP uses the maximum standardized nonconformity score across `eta_head` and `eta_mid_max`.

### 3. Evaluate the held-out Layer-2 test subset

```bash
python 03_evaluate_layer2.py \
  --test-x data/layer2_test_stable_X.csv \
  --test-y data/layer2_test_stable_Y.csv
```

This script compares Raw DE, Independent CP and Joint CP coverage/MPIW and performs the four-category capacity screening: `Unflagged`, `Head-car flagged`, `Mid-car flagged`, and `Jointly flagged`.

### 4. SHAP analysis of the two capacity-utilization indicators

```bash
python 04_shap_capacity_indicators.py \
  --background-x data/layer2_train_X.csv \
  --explain-x data/layer2_test_stable_X.csv
```

The released feature order is documented in `data/README.md`. In particular, dimensions 15 and 16 are `V_off` and `H_off`, respectively.

### 5. Layer-1 amplitude-normalized conformal calibration

```bash
python 05_layer1_normalized_cp.py \
  --calib data/layer1_calibration.csv \
  --test data/layer1_test.csv
```

The nonconformity score is

`abs(true_max - pred_max) / max(pred_max, delta)`

with `delta = 1e-6 mm` by default. The resulting sample-dependent margin is `q_norm * max(pred_max, delta)`. The script also exports a diagnostic comparison with the original global absolute-error CP rule.

## Notes on data independence

Training, validation, conformal calibration and final test data must be kept independent. The scripts intentionally accept explicit subset files rather than relying on hard-coded row numbers. This avoids accidental data leakage and makes the data-partitioning assumptions visible.

## Repository scope

The released scripts implement the uncertainty-calibration, Layer-2 surrogate, screening and SHAP analyses. High-fidelity FE model files are not distributed here.
