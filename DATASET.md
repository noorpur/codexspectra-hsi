# Dataset documentation

## Primary dataset: HYPERDOC

**Name:** Hyperspectral dataset of historical documents and mock-ups from 400 to 1700 nm (HYPERDOC)  
**Domain:** cultural heritage, historical document analysis, hyperspectral reflectance imaging  
**Modalities:** VNIR hyperspectral cubes, SWIR hyperspectral cubes, RGB previews, metadata, pixel-level material annotations  
**Typical tasks:** material classification, ink/support separation, spectral unmixing, ageing analysis, binarization, spectral-spatial representation learning

HYPERDOC is used because it is large, public, complete, and scientifically close to non-invasive analysis of historical writing materials. It contains 1681 hyperspectral datacubes and millions of reflectance spectra across 400 to 1700 nm, with historical-document mock-ups and real documents.

### Expected local layout

Raw data are intentionally not committed.

```text
data/raw/hyperdoc/
  full_documents/
  minicubes/
  metadata/
  annotations/
```

The manifest builder recursively searches for `.mat`, `.npy`, `.npz`, `.bil`, and `.hdr` files. If metadata are present, it extracts object/cube/region labels. If labels are missing, it creates an unlabeled manifest that can still be used for self-supervised pretraining and exploratory analysis.

### Citation

López-Baldomero, A. B. et al. Hyperspectral dataset of historical documents and mock-ups from 400 to 1700 nm (HYPERDOC). *Scientific Data* (2025).

HYPERDOC code repository: https://github.com/anabelenlb/HYPERDOC_Database_code

## Optional expansion datasets

### HySpecNet-11k

Used for optional self-supervised hyperspectral pretraining. It contains 11,483 non-overlapping hyperspectral image patches with 224 spectral bands.

### HeiPorSPECTRAL

Used for optional biological-tissue spectral transfer experiments. It contains 5756 HSI cubes of 20 pig organ classes across 11 pigs. This is useful only as a domain-transfer stress test, not as a historical-document dataset.

## Data governance

- Raw data stay in `data/raw/` and are ignored by git.
- Processed feature tables are written to `data/processed/`.
- Each manifest row includes a checksum where possible.
- Splits are group-aware by document/object/cube to prevent inflated pixel-level performance.
- Any manually edited labels must be documented in `reports/tables/label_audit.csv`.
