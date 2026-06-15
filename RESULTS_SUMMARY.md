# HYPERDOC Minicube Results Summary

## Dataset scope

This repository uses the extracted HYPERDOC minicube dataset, not the full ParentCubes archive.

## Experiment 1: document-type benchmark

This earlier benchmark classified document-family labels from filenames:

- `mock-up`
- `genealogies`
- `provincial`
- `royal`

It was useful for validating the pipeline, but it is not the main scientific result.

### Tuned Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.7687 |
| Macro-F1 | 0.5037 |
| Weighted-F1 | 0.7565 |

### Balanced Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.7553 |
| Macro-F1 | 0.4943 |
| Weighted-F1 | 0.7508 |

## Experiment 2: GT-mask material classification

This is the main result. It uses HYPERDOC ground-truth masks to classify sampled spectra into material/region categories.

- `Ink`
- `Pencil`
- `Pigment`
- `Substrate`

Feature table:

- 128,000 GT-mask-labeled spectra
- 360 document groups
- Grouped train/validation/test split
- No document-group leakage

### Tuned Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.9165 |
| Macro-F1 | 0.8882 |
| Weighted-F1 | 0.9156 |

### Per-class F1

| Class | F1 |
|---|---:|
| Ink | 0.9421 |
| Pencil | 0.8513 |
| Pigment | 0.8798 |
| Substrate | 0.8797 |

## Interpretation

The GT-mask material classifier is substantially stronger and more useful than the document-type benchmark. The model separates ink, pencil, pigment, and substrate spectra with high macro-F1 under a document-grouped split.

The remaining limitations are:

1. The model is trained on sampled pixels, not full-scene dense prediction.
2. The current figures do not yet include RGB/GT/prediction overlay maps.
3. The class labels are broad material categories; finer-grained material classes should be evaluated separately.
4. Generalization should be tested on additional held-out manuscripts or external cultural-heritage HSI datasets.

## Recommended next steps

1. Add dense prediction maps for selected documents.
2. Generate RGB + GT + prediction overlays.
3. Add uncertainty maps.
4. Evaluate finer-grained material classes.
5. Add a completed model card.
