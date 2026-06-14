# Model card: <model name>

## Intended use

Material-state classification and spectral-spatial mapping for historical-document hyperspectral images.

## Not intended for

- Authentication of cultural objects without expert review
- Conservation decisions without domain specialist validation
- Species/origin claims without independent reference analysis

## Training data

- Dataset:
- Manifest checksum:
- Number of cubes:
- Number of sampled spectra:
- Classes:
- Split strategy:

## Metrics

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | TBD | TBD |
| Macro F1 | TBD | TBD |
| Weighted F1 | TBD | TBD |
| ECE | TBD | TBD |

## Validation notes

- Group-aware split:
- Leakage audit:
- Calibration:
- Bootstrap confidence intervals:

## Interpretability

- Wavelength attribution method:
- Important wavelength regions:
- Spatial uncertainty patterns:

## Limitations

- Dataset-specific acquisition conditions may affect transfer.
- Pixel labels may not capture every mixed-material region.
- Reflectance spectra are not a direct substitute for molecular reference techniques.
