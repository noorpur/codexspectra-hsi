# HYPERDOC Minicube Benchmark Results

## Dataset scope

This benchmark uses the extracted HYPERDOC minicube dataset, not the full ParentCubes archive.

- Input: HYPERDOC `.h5` minicubes
- Feature table: 180,000 sampled spectra
- Split strategy: grouped by document ID to reduce VNIR/SWIR sibling leakage
- Task: document-type classification
- Classes: `mock-up`, `genealogies`, `provincial`, `royal`

## Baseline tuned Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.7687 |
| Macro-F1 | 0.5037 |
| Weighted-F1 | 0.7565 |

## Balanced Random Forest

| Metric | Value |
|---|---:|
| Accuracy | 0.7553 |
| Macro-F1 | 0.4943 |
| Weighted-F1 | 0.7508 |

## Honest train-only diagnostic for balanced model

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Train | 0.9616 | 0.9381 | 0.9626 |
| Validation | 0.7501 | 0.4971 | 0.7456 |
| Test | 0.7617 | 0.5111 | 0.7539 |

## Interpretation

The balanced model slightly improves minority-class behavior in the honest diagnostic, but it does not fully solve generalization. The main failure mode is poor transfer to held-out document groups, especially for `royal` and `provincial`.

The current task is a document-type benchmark. The next major usability upgrade is to use HYPERDOC ground-truth masks for material or region classification rather than predicting document-family labels from filenames.

## Recommended next steps

1. Implement GT-mask material/region label extraction.
2. Train material classifiers with document-grouped validation.
3. Add RGB/GT/prediction overlay figures.
4. Add per-document failure analysis.
5. Add uncertainty maps for low-confidence regions.
