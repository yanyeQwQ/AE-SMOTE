# AE-SMOTE

Autoencoder-DBSCAN-SMOTE (AE-DBSCAN-SMOTE) for high-dimensional imbalanced data.

## Overview

This repository contains the supplementary code for the manuscript on an improved synthetic minority over-sampling technique that integrates:

- F-statistic based feature redundancy removal
- Autoencoder-driven dimensionality reduction
- DBSCAN-based noise filtering in the latent space
- Standard SMOTE interpolation

The method is designed for high-dimensional imbalanced omics data (e.g. metabolomics, transcriptomics).

## Files

| File | Description |
|------|-------------|
| `AE-SMOTE.py` | Implementation of the AE-DBSCAN-SMOTE over-sampling method |
| `CreateDataset.py` | Synthetic metabolomics imbalanced dataset generator for benchmarking |

## Dependencies

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- tensorflow >= 2.8.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- openpyxl >= 3.0.0

## Usage

Run AE-DBSCAN-SMOTE on your own dataset (the last column is assumed to be the class label):

```bash
python AE-SMOTE.py
```

Generate the benchmark synthetic datasets:

```bash
python CreateDataset.py
```

## Reproducibility

All random seeds are fixed (42) to ensure full reproducibility of results.
