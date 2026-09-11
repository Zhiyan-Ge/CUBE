# Figure 4 visualization

This directory contains the scripts used to generate the three Figure 4 panels for CUBE representation complementarity and tissue-content sensitivity. The scripts consume frozen prediction CSVs and do not retrain the models.

## Panel design

- `fig4a_validation_complementarity_v1.py`
  - validation overall concept MSE for UR1-only, UR2-only, and Fusion
  - validation concept-resolved Pearson correlations
- `fig4b_test_pearson_tissue_threshold_v1.py`
  - held-out Test945 Pearson trajectories for DAPI, CD3, and panCK
  - retained-tissue series: Full, ≥1%, ≥5%, ≥10%, ≥25%, ≥30%, ≥50%, ≥75%
- `fig4c_mse_advantage_tissue_threshold_v1.py`
  - overall MSE difference relative to Fusion
  - positive ΔMSE means lower error for Fusion

All panels are calculated from prediction CSVs rather than manually copied headline values.

## Paths

`fig4_paths_v1.py` resolves the repository root from its own location and uses repository-root-relative input paths under `models/`. By default it expects:

```text
models/test_models_3/test15/ur_branch/evaluation/best_val/predictions.csv
models/test_models_3/test15/ur_branch/evaluation/best_test/predictions.csv
models/ur1_ablation/evaluation/best_val/predictions.csv
models/ur1_ablation/evaluation/best_test/predictions.csv
models/ur2_ablation/evaluation/best_val/predictions.csv
models/ur2_ablation/evaluation/best_test/predictions.csv
models/test_models_3/test15/ur_branch/tissue_qc/tissue_fraction_per_patch.csv
```

Outputs are written to:

```text
result/visualization/fig4/result/tissue_stratified/
```

Edit `fig4_paths_v1.py` only if the released checkpoint/result archive is restored to a different repository-relative layout.

## Run

The recommended convention is to run from the repository root:

```bash
python downstream/visualization/fig4/run_fig4_visualization_v1.py
```

The panels can also be run individually:

```bash
python downstream/visualization/fig4/fig4a_validation_complementarity_v1.py
python downstream/visualization/fig4/fig4b_test_pearson_tissue_threshold_v1.py
python downstream/visualization/fig4/fig4c_mse_advantage_tissue_threshold_v1.py
```

`tissue_stratified_evaluation_extended.py` contains the extended retained-tissue sensitivity calculation used by the final Figure 4 series. These thresholds are post hoc sensitivity strata applied consistently to the frozen models; the full series is reported rather than selecting a cutoff according to test performance.
