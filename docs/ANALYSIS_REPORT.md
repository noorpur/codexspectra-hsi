# Analysis Report: HYPERDOC GT-Mask Material Classification

## 1. Overview

This project builds a reproducible hyperspectral imaging workflow for historical document analysis using the HYPERDOC minicube dataset. The final workflow moves beyond document-type classification and uses HYPERDOC ground-truth masks to classify material/region categories from sampled hyperspectral pixels.

The main material classification task predicts four broad categories:

- Ink
- Pencil
- Pigment
- Substrate

This is the main research-facing result of the repository because it uses GT-mask-derived labels rather than filename-derived document-family labels.

## 2. Dataset and feature construction

The GT-material feature table was generated from HYPERDOC `.h5` minicubes and corresponding GT mask PNG files.

Feature table summary:

| Item | Value |
|---|---:|
| GT-mask-labeled spectra | 128,000 |
| Material categories | 4 |
| Document groups | 360 |
| Train groups | 230 |
| Validation groups | 58 |
| Test groups | 72 |

The grouped split is important because VNIR/SWIR siblings and repeated pixels from the same document can otherwise leak across train/test boundaries. The leakage audit found:

| Split overlap | Count |
|---|---:|
| Train ∩ Validation | 0 |
| Train ∩ Test | 0 |
| Validation ∩ Test | 0 |

This means the reported test performance is evaluated on held-out document groups.

## 3. Material label distribution

The GT-material dataset is imbalanced:

| Class | Sample count |
|---|---:|
| Ink | 82,880 |
| Substrate | 26,720 |
| Pencil | 10,400 |
| Pigment | 8,000 |

Ink dominates the dataset, but the final model still performs well across the minority categories. Macro-F1 is therefore the most important metric because it gives each class equal weight.

## 4. Model performance

The best tuned Random Forest achieved strong test-set performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.9165 |
| Macro-F1 | 0.8882 |
| Weighted-F1 | 0.9156 |
| Macro-F1 bootstrap mean | 0.8884 |
| Macro-F1 95% CI low | 0.8832 |
| Macro-F1 95% CI high | 0.8933 |

These results show that the GT-mask material task is substantially more reliable than the earlier document-type benchmark.

## 5. Per-class performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Ink | 0.9230 | 0.9620 | 0.9421 | 15,360 |
| Pencil | 0.8761 | 0.8279 | 0.8513 | 2,400 |
| Pigment | 0.8963 | 0.8639 | 0.8798 | 1,440 |
| Substrate | 0.9194 | 0.8432 | 0.8797 | 5,600 |

Ink is the strongest class, with very high recall. Pencil, Pigment, and Substrate also perform well, with F1-scores above 0.85. The model is therefore not merely learning the dominant Ink class.

## 6. Confusion matrix interpretation

The normalized confusion matrix shows a strong diagonal:

| True class | Main correct recall |
|---|---:|
| Ink | 0.96 |
| Pencil | 0.83 |
| Pigment | 0.86 |
| Substrate | 0.84 |

The largest remaining confusions are:

- Pencil misclassified as Ink.
- Pigment misclassified as Ink.
- Substrate misclassified as Ink.

This pattern is plausible because dark writing regions, mixed material boundaries, and textured substrate areas may share spectral characteristics with ink-like regions, especially at edges or degraded regions.

## 7. PCA interpretation

The PCA projection shows that the material classes are not perfectly separable in the first two principal components. PC1 explains most of the variance, but the classes overlap substantially in 2D.

This is useful because it explains why a nonlinear classifier such as Random Forest is appropriate. The model likely uses nonlinear thresholds across multiple resampled spectral features rather than relying on a simple 2D separation.

## 8. Overfitting and generalization

The honest split diagnostic shows that model performance remains relatively stable across validation and test splits. Unlike the earlier document-type benchmark, the GT-material task does not show a severe train/test collapse.

This suggests that GT-mask material classification is a more meaningful and generalizable target than broad document-family classification.

## 9. Overlay panel interpretation

The overlay figures compare:

1. RGB preview
2. Ground-truth material mask
3. Predicted material map
4. Prediction confidence

The mock-up examples show strong visual agreement between GT masks and predictions. The model captures ink strokes and substrate regions with high confidence.

The provincial example also shows good alignment between predicted ink and substrate regions.

The royal example is more complex. The model captures much of the text structure but introduces more noisy regions and lower-confidence areas. This suggests that historical manuscript pages with denser writing, blur, aging, or texture variation remain more challenging than controlled mock-ups.

## 10. Key findings

1. GT-mask material classification is the strongest and most meaningful workflow in the repository.
2. The tuned Random Forest achieves high test performance under a document-grouped split.
3. The model performs well across all four material categories, not only the majority Ink class.
4. Remaining errors are concentrated around visually and spectrally ambiguous regions.
5. Overlay panels provide qualitative evidence that the model is learning spatially meaningful material structure.
6. The next major improvement should be dense full-document prediction with uncertainty maps and finer-grained material classes.

## 11. Limitations

The current workflow has several limitations:

- It uses sampled pixels rather than full dense prediction during training.
- It evaluates broad material categories rather than all fine-grained material classes.
- The model has only been tested within HYPERDOC, not on an external cultural-heritage HSI dataset.
- Confidence maps are derived from Random Forest probabilities and should be treated as heuristic uncertainty, not calibrated uncertainty.
- Some overlay examples show noisy predictions around complex handwriting and degraded regions.

## 12. Recommended next steps

1. Add dense prediction over full HYPERDOC minicubes.
2. Generate RGB + GT + prediction + uncertainty panels for multiple held-out documents.
3. Evaluate fine-grained classes such as specific ink and substrate types.
4. Add class-balanced and group-balanced training options for material classification.
5. Add calibrated uncertainty estimation.
6. Add a completed model card describing intended use, limitations, and non-clinical/non-conservation decision boundaries.
7. Add a final README section highlighting the GT-material benchmark as the main result.

## 13. Bottom line

The repository now contains a credible, reproducible hyperspectral material-classification workflow for historical document analysis. The GT-mask material classifier is the main result and should be emphasized over the earlier document-type benchmark.
