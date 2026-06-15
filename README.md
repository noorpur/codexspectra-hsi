# CodexSpectra-HSI

**Multiscale hyperspectral learning for historical document materials, ageing proxies, and conservation-state mapping**

I built this repository as a research-grade bridge between my earlier work in biomedical imaging, biological signal modelling, structural/protein validation, interpretable AI, and reproducible ML systems. My previous projects already sit at the intersection of noisy biological data, multimodal image/signal pipelines, generative modelling, model auditing, and careful validation. This project transfers that same computational style into non-invasive spectral imaging for materials.

The core idea is simple: historical documents are material-biological-chemical systems, not just images. A manuscript surface contains spectral signatures from support material, inks, pigments, degradation products, conservation treatments, acquisition artefacts, and spatial heterogeneity. I therefore treat hyperspectral cubes as structured scientific measurements rather than ordinary photographs.

## Research questions

1. Can pixel-level and region-level spectral models separate historical-document materials while avoiding leakage between visually similar regions from the same object?
2. Can self-supervised or generative spectral representations learn material-state structure before labels are used?
3. Can model explanations highlight wavelength regions and spatial zones that are physically interpretable rather than only statistically predictive?
4. Can a reproducible pipeline produce conservation-style maps, uncertainty maps, and audit reports from raw hyperspectral cubes?

## Main dataset

This project is designed around **HYPERDOC**, a public hyperspectral imaging dataset of historical documents and mock-ups from 400 to 1700 nm. HYPERDOC contains mock-ups of historical inks on multiple supports, including artificially aged samples, plus real historical documents from the 15th to 17th centuries. It includes VNIR and SWIR reflectance imaging, spatial registration, minicubes, pixel-level ground-truth material annotations, RGB renderings, and metadata.

Dataset/paper sources:

- López-Baldomero et al., *Hyperspectral dataset of historical documents and mock-ups from 400 to 1700 nm (HYPERDOC)*, Scientific Data, 2025.
- HYPERDOC processing code: https://github.com/anabelenlb/HYPERDOC_Database_code
- HYPERDOC resource page: https://hsi.yale.edu/resource/1258

Optional expansion datasets supported by the configuration layer:

- **HySpecNet-11k** for large-scale unsupervised hyperspectral pretraining.
- **HeiPorSPECTRAL** for biological tissue hyperspectral domain-transfer experiments.

I do not commit raw datasets into the repository. The pipeline expects externally downloaded data and creates reproducible manifests, feature tables, model artifacts, figures, and reports.

## Why this project fits my existing research arc

My earlier work includes MRI-to-PET-like cross-modality synthesis, EEG scalogram classification with Grad-CAM/attention visualisation, AlphaFold/NMR validation under controlled noise regimes, protein-state representation learning, VAE/counterfactual biomedical modelling, cryo-EM quality triage, medical segmentation, and reproducible CLI/reporting pipelines. CodexSpectra-HSI extends the same pattern: non-invasive measurement, biological/material structure, spectral-spatial AI, uncertainty-aware validation, and interpretable outputs.

The technical bridge is deliberate:

| Previous strand | This repository extends it into |
|---|---|
| MRI/PET and pelvic MRI synthesis | spectral-spatial material mapping and modality-aware image pipelines |
| EEG scalograms and Grad-CAM | wavelength attribution, saliency, and spatial uncertainty maps |
| AlphaFold/NMR validation | scientific measurement fidelity, noise regimes, and reference-aware validation |
| Protein-state / VAE projects | latent material-state representations and ageing/degradation proxies |
| CryoEM and binder benchmarking | conservative group-level validation and leakage-sensitive evaluation |
| Healthcare-access motivation | non-invasive, lower-risk scientific imaging for fragile heritage objects |

## What the pipeline does

### Data layer

- Scans HYPERDOC-style directories for VNIR/SWIR minicubes, metadata, masks, and RGB previews.
- Builds a deterministic manifest with object IDs, cube IDs, region IDs, wavelength ranges, labels, paths, checksums, and split groups.
- Supports `.mat`, `.npy`, `.npz`, and ENVI-style `.bil`/`.hdr` files.
- Keeps raw data immutable and writes derived data to `data/processed/`.

### Preprocessing

- Bad-band removal and wavelength filtering.
- Reflectance clipping and robust normalization.
- Savitzky-Golay smoothing.
- Standard normal variate correction.
- First/second spectral derivatives.
- PCA/UMAP-ready embeddings.
- Patch extraction for spectral-spatial models.
- Group-aware train/validation/test splits by document/object/cube.

### Models

Classical baselines:

- Logistic regression
- Random forest
- RBF SVM
- HistGradientBoosting
- Optional XGBoost/LightGBM hooks

Deep models:

- Spectral MLP
- 1D spectral CNN
- Spectral autoencoder
- Spectral-spatial patch CNN
- Lightweight transformer-style spectral encoder

Validation and tuning:

- Grouped train/validation/test split
- Stratified fallback splitting when group metadata is absent
- Optuna hyperparameter search
- Calibration metrics
- Bootstrap confidence intervals
- Confusion matrices
- Per-class precision/recall/F1
- Wavelength attribution summaries
- Leakage report
- Reproducibility report

### Outputs

After a full run, the repository writes:

```text
reports/
  figures/
    spectral_means_by_material.png
    pca_latent_space.png
    confusion_matrix_test.png
    calibration_curve.png
    wavelength_importance.png
    uncertainty_map_example.png
  tables/
    manifest_summary.csv
    metrics_test.csv
    metrics_bootstrap_ci.csv
    ablation_summary.csv
    leakage_audit.csv
  model_cards/
    best_model.md
  experiment_report.md
models/
  best_model.joblib
  spectral_autoencoder.pt
  preprocessing_pipeline.joblib
```

The figures and final result tables are intentionally versioned as generated artifacts, not handwritten claims.

## Repository structure

```text
codexspectra-hsi/
  README.md
  DATASET.md
  MODEL_CARD_TEMPLATE.md
  REPRODUCIBILITY.md
  CITATION.cff
  pyproject.toml
  Makefile
  configs/
    default.yaml
    m2_quick.yaml
    full_hyperdoc.yaml
  src/codexspectra/
    data/
    features/
    models/
    training/
    evaluation/
    utils/
  scripts/
    00_make_manifest.py
    01_preprocess.py
    02_train_baselines.py
    03_tune_models.py
    04_train_deep.py
    05_evaluate.py
    06_make_report.py
    run_smoke_test.py
  tests/
  reports/
    figures/.gitkeep
    tables/.gitkeep
```

## Quick start on Apple Silicon MacBook M2

```bash
# 1. Create environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 2. Run the synthetic smoke test first
python scripts/run_smoke_test.py

# 3. Place HYPERDOC data outside git, for example:
# data/raw/hyperdoc/

# 4. Build a manifest
python scripts/00_make_manifest.py --config configs/m2_quick.yaml

# 5. Preprocess spectra
python scripts/01_preprocess.py --config configs/m2_quick.yaml

# 6. Train baseline models
python scripts/02_train_baselines.py --config configs/m2_quick.yaml

# 7. Tune selected models
python scripts/03_tune_models.py --config configs/m2_quick.yaml

# 8. Train deep spectral models with MPS when available
python scripts/04_train_deep.py --config configs/m2_quick.yaml

# 9. Evaluate and generate reports
python scripts/05_evaluate.py --config configs/m2_quick.yaml
python scripts/06_make_report.py --config configs/m2_quick.yaml
```

The code automatically uses `mps` when PyTorch detects Apple Silicon acceleration. The quick config keeps batch sizes and patch counts modest so I can iterate locally before launching full runs.

## Reproducibility note

This repository is built so that results are reproducible without pretending that large scientific data pipelines are magic boxes. I pin random seeds, save dataset manifests with checksums, separate raw and processed data, use group-aware validation to avoid object-level leakage, persist configs with every run, and write machine-readable metrics/tables alongside plots. Large raw datasets are not committed to git; instead, `DATASET.md` records the exact dataset sources and expected directory layout. Every generated report should include the git commit, config hash, Python version, platform, device, and data manifest checksum.

## Current status

- [x] Research-grade repository scaffold
- [x] HYPERDOC-compatible data model
- [x] M2-friendly configuration
- [x] Classical baselines
- [x] Deep spectral models
- [x] Group-aware validation
- [x] Optuna tuning hooks
- [x] Report templates
- [x] Run HYPERDOC minicube document-type benchmark
- [x] Run GT-mask material classification experiment
- [x] Add generated analysis figures and overlay visualizations
- [x] Add final analysis report after evaluation
### Next improvements

- Dense full-document prediction mode
- Calibrated uncertainty maps
- Release-ready model card

## License

Code is released under the MIT License. Dataset usage follows the licenses and terms of the original data providers.
