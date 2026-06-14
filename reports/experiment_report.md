# Experiment report

## Configuration

- Config: `configs/smoke.yaml`
- Config hash: `75e17383bcd6090a`

## Test metrics

| model_path                             |   accuracy |   macro_f1 |   weighted_f1 |   macro_f1_mean |   macro_f1_ci_low |   macro_f1_ci_high |
|:---------------------------------------|-----------:|-----------:|--------------:|----------------:|------------------:|-------------------:|
| models/best_tuned_random_forest.joblib |          1 |          1 |             1 |               1 |                 1 |                  1 |

## Figures

- `reports/figures/confusion_matrix_test.png`
- `reports/figures/spectral_means_by_material.png` (generated in full analysis run)
- `reports/figures/pca_latent_space.png` (generated in full analysis run)
- `reports/figures/wavelength_importance.png` (generated in full analysis run)
- `reports/figures/uncertainty_map_example.png` (generated in full analysis run)

## Interpretation checklist

- Do the top wavelengths align with known reflectance differences between ink/support/material regions?
- Are errors concentrated in mixed-material or degraded/aged regions?
- Are per-document metrics stable, or is the model overfitting acquisition-specific signatures?
- Does group-aware performance remain credible compared with pixel-random splits?
