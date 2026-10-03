# Data interface

The FE database used in the study is not bundled with this repository. The scripts expect numeric CSV files prepared from the independent training, validation, calibration and held-out test subsets.

## Layer-2 Deep Ensemble inputs

`X` files contain 16 columns in this order:

1. `F_ac`
2. `F_ea`
3. `F_hc`
4. `S_ac`
5. `S_ea`
6. `S_hc`
7. `F_s1`
8. `F_s2`
9. `F_s3`
10. `F_s4`
11. `S_s1`
12. `S_s2`
13. `S_s3`
14. `S_s4`
15. `V_off`
16. `H_off`

`Y` files contain two columns in this order:

1. `eta_head`
2. `eta_mid_max`

The Layer-2 calibration and test `X/Y` files should already contain only samples retained as calibrated stable by Layer 1. In the current study, the Joint-CP calibration subset contains 113 retained samples and the held-out Layer-2 test subset contains 55 retained samples.

## Layer-1 conformal-calibration inputs

Both the Layer-1 calibration and held-out test CSV files must contain named columns:

- `pred_max`: surrogate-predicted global maximum wheelset lift (mm)
- `true_max`: FE global maximum wheelset lift (mm)

The global maximum is extracted from the 16 monitored wheelset-lift channels associated with the first two cars of the moving consist and the first two cars of the stationary consist (four wheelsets per car).
