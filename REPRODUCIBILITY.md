# Reproducibility protocol

This project treats reproducibility as a scientific constraint, not an aesthetic garnish.

## Determinism

- Python, NumPy, scikit-learn, and PyTorch seeds are set from the config.
- PyTorch deterministic algorithms are enabled where supported.
- MPS acceleration is used on Apple Silicon when available, with CPU fallback.

## Data lineage

Every run stores:

- Config file path and config hash
- Git commit if available
- Dataset manifest checksum
- Python version
- Platform and device
- Package versions
- Random seed
- Split seed
- Number of objects/cubes/pixels/classes

## Leakage control

The default split is group-aware:

1. Hold out whole documents/objects/cubes before pixel sampling.
2. Fit preprocessing only on training data.
3. Apply learned transforms to validation/test data.
4. Report per-group and per-class metrics.

Pixel-level hyperspectral datasets can look deceptively strong if neighbouring pixels from the same cube leak across splits. The pipeline therefore makes leakage auditing a first-class output.

## Experiment tiers

### Smoke test

Runs on synthetic spectra and verifies that the CLI, preprocessing, model training, and reporting stack work.

### M2 quick run

Uses bounded sampling and smaller models for local development on my MacBook M2.

### Full run

Uses the full HYPERDOC minicube set, larger Optuna search, repeated seeds, bootstrapped confidence intervals, and full report generation.

## Result reporting rule

I do not write claimed metrics into the README by hand. Metrics, figures, and model cards should be generated from the pipeline and then committed as artifacts with their config and manifest checksum.
