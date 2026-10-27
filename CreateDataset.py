"""
Synthetic Metabolomics Dataset Generator for Imbalanced Classification Studies
===========================================================================
Supplementary code for the manuscript:
[Insert Manuscript Title Here]

Purpose
-------
Generates synthetic imbalanced binary classification datasets that mimic
real-world metabolomics data characteristics, including log-normal abundance
distribution, pathway-based correlation structure, detection-limit sparsity,
measurement noise, outlier samples, and predefined true differential features.
Supports systematic parameter sweeps for benchmarking imbalanced learning
and feature selection methods.

Reproducibility
---------------
All random seeds are fixed to ensure full reproducibility of results.

Dependencies
------------
- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- openpyxl >= 3.0.0 (for Excel output)
"""

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Global reproducibility configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for all involved libraries to ensure reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Global random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize seed on module import
set_global_seed()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def generate_positive_definite_corr(
        size: int,
        min_corr: float = 0.7,
        max_corr: float = 0.9
) -> np.ndarray:
    """Generate a symmetric positive-definite correlation matrix.

    Ensures the matrix is valid for Cholesky decomposition by correcting
    non-positive eigenvalues and re-normalizing the diagonal to 1.0.

    Parameters
    ----------
    size : int
        Dimension of the correlation matrix (number of features in a pathway).
    min_corr : float, default=0.7
        Minimum pairwise correlation coefficient.
    max_corr : float, default=0.9
        Maximum pairwise correlation coefficient.

    Returns
    -------
    corr : ndarray of shape (size, size)
        Symmetric positive-definite correlation matrix with unit diagonal.
    """
    # Generate symmetric random correlation matrix
    corr = np.random.uniform(min_corr, max_corr, size=(size, size))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    # Correct non-positive eigenvalues to ensure positive definiteness
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals[eigvals < 1e-6] = 1e-6
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Re-normalize to unit diagonal
    d = np.diag(corr)
    corr = corr / np.sqrt(d[:, None] * d[None, :])
    return corr


# ---------------------------------------------------------------------------
# Core dataset generation function
# ---------------------------------------------------------------------------
def generate_metabolomics_dataset(
        n_samples: int = 100,
        n_features: int = 1000,
        imbalance_ratio: float = 3.0,
        noise_level: float = 0.1,
        n_pathways: int = 5,
        pathway_size: int = 10,
        outlier_ratio: float = 0.05,
        sparsity_range: Tuple[float, float] = (0.2, 0.5),
        differential_ratio: float = 0.15,
        random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic imbalanced dataset with realistic metabolomics properties.

    Automatically adapts to small sample settings to avoid clustering failures.
    The generated data preserves key characteristics of real metabolomics data:
    log-normal distribution, pathway correlation structure, sparsity, noise,
    outliers, and predefined differential features.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of samples (control + case).
    n_features : int, default=1000
        Total number of metabolic features.
    imbalance_ratio : float, default=3.0
        Imbalance ratio = majority class size / minority class size.
    noise_level : float, default=0.1
        Gaussian noise intensity relative to feature standard deviation.
    n_pathways : int, default=5
        Number of simulated metabolic pathways.
    pathway_size : int, default=10
        Number of metabolites per pathway.
    outlier_ratio : float, default=0.05
        Proportion of outlier samples in the total dataset.
    sparsity_range : tuple of (float, float), default=(0.2, 0.5)
        Range of zero-value proportions per feature (detection limit effect).
    differential_ratio : float, default=0.15
        Proportion of informative features that are truly differential.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Z-score normalized feature matrix.
    y : ndarray of shape (n_samples,)
        Binary class labels (0 = control/majority, 1 = case/minority).
    true_diff_idx : ndarray
        Indices of true differential metabolites for algorithm evaluation.
    """
    np.random.seed(random_state)

    # 1. Calculate class sample sizes
    n_minority = round(n_samples / (1 + imbalance_ratio))
    n_majority = n_samples - n_minority
    class_weights = [n_majority / n_samples, n_minority / n_samples]

    # 2. Adapt pathway/cluster count to sample size (avoid clustering errors)
    n_clusters_per_class = min(n_pathways, n_minority)
    actual_pathway_size = max(10, pathway_size)
    n_informative = n_clusters_per_class * actual_pathway_size
    n_informative = min(n_informative, n_features)

    # 3. Generate base data with cluster structure (simulating pathways)
    X_base, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=n_clusters_per_class,
        weights=class_weights,
        flip_y=0,
        class_sep=1.5,
        random_state=random_state
    )

    # 4. Transform to log-normal distribution (standard for metabolite abundance)
    X_base = np.exp(X_base - X_base.min() + 1)  # Shift to ensure positivity
    X = X_base * np.random.lognormal(mean=0, sigma=0.8, size=X_base.shape)

    # 5. Inject within-pathway strong correlations
    for p_idx in range(n_clusters_per_class):
        start = p_idx * actual_pathway_size
        end = min(start + actual_pathway_size, n_informative)
        size = end - start
        if size < 2:
            continue
        corr_mat = generate_positive_definite_corr(size, 0.7, 0.9)
        L = np.linalg.cholesky(corr_mat)
        X[:, start:end] = X[:, start:end] @ L.T

    # 6. Add Gaussian measurement noise (mass spectrometry error)
    feature_std = np.std(X, axis=0, keepdims=True)
    X += np.random.normal(0, noise_level * feature_std, size=X.shape)

    # 7. Add outlier samples (simulating contamination or technical errors)
    n_outliers = max(1, int(n_samples * outlier_ratio)) if outlier_ratio > 0 else 0
    if 0 < n_outliers < n_samples:
        out_idx = np.random.choice(n_samples, n_outliers, replace=False)
        X[out_idx] *= np.random.uniform(5, 10, size=(n_outliers, n_features))

    # 8. Add metabolomics-specific sparsity (below detection limit)
    zero_probs = np.random.uniform(*sparsity_range, size=n_features)
    zero_mask = np.random.binomial(1, p=zero_probs, size=X.shape)
    X[zero_mask == 1] = 0.0

    # 9. Mark true differential metabolites for evaluation
    n_diff = max(3, int(n_informative * differential_ratio))
    true_diff_idx = np.random.choice(n_informative, size=n_diff, replace=False)

    # 10. Z-score normalization (standard preprocessing step)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, true_diff_idx


# ---------------------------------------------------------------------------
# Dataset validation and visualization
# ---------------------------------------------------------------------------
def validate_single_dataset(
        X: np.ndarray,
        y: np.ndarray,
        true_diff_idx: np.ndarray,
        save_path: str = None
) -> None:
    """Generate a multi-panel validation report for a synthetic dataset.

    Creates a 2x3 figure with: abundance distribution, sparsity distribution,
    pathway correlation heatmap, class distribution bar plot, differential
    feature comparison, and summary statistics.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Class label vector.
    true_diff_idx : ndarray
        Indices of true differential features.
    save_path : str, optional
        File path to save the validation figure. If None, figure is not saved.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Dataset Validation (n={X.shape[0]}, d={X.shape[1]})',
        fontsize=16,
        y=1.02
    )

    # 1. Non-zero abundance distribution
    ax = axes[0, 0]
    non_zero = X[X != 0].flatten()
    sns.histplot(non_zero, kde=True, bins=50, color='#2c7fb8', ax=ax)
    ax.set_title('Non-zero Abundance Distribution')

    # 2. Zero-value ratio per feature
    ax = axes[0, 1]
    zero_ratios = np.mean(X == 0, axis=0)
    sns.histplot(zero_ratios, bins=20, color='#2ca25f', ax=ax)
    ax.axvline(
        np.mean(zero_ratios),
        color='red',
        linestyle='--',
        label=f'Mean: {np.mean(zero_ratios):.2f}'
    )
    ax.set_title('Zero-value Ratio per Feature')
    ax.legend()

    # 3. Pathway correlation heatmap (first 20 features)
    ax = axes[0, 2]
    n_show = min(20, X.shape[1])
    pathway_data = X[:, :n_show]
    corr = np.corrcoef(pathway_data.T)
    sns.heatmap(
        corr,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax
    )
    ax.set_title('Correlation Heatmap (First Pathway)')

    # 4. Class distribution
    ax = axes[1, 0]
    counts = pd.Series(y).value_counts().sort_index()
    counts.plot(kind='bar', color=['#3182bd', '#e6550d'], ax=ax)
    actual_ir = counts[0] / counts[1]
    ax.set_title(f'Class Distribution\nActual IR: {actual_ir:.1f}:1')
    ax.set_xticklabels(['Control', 'Case'], rotation=0)

    # 5. Top true differential metabolites comparison
    ax = axes[1, 1]
    n_top = min(10, len(true_diff_idx))
    top_diff = true_diff_idx[:n_top]
    ctrl_mean = np.mean(X[y == 0][:, top_diff], axis=0)
    case_mean = np.mean(X[y == 1][:, top_diff], axis=0)
    x = np.arange(n_top)
    width = 0.35
    ax.bar(x - width / 2, ctrl_mean, width, label='Control', alpha=0.8, color='#3182bd')
    ax.bar(x + width / 2, case_mean, width, label='Case', alpha=0.8, color='#e6550d')
    ax.set_title('Top True Differential Metabolites')
    ax.legend()

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    avg_pathway_corr = np.mean(corr[np.triu_indices_from(corr, k=1)])
    stats_text = f"""
    Dataset Statistics
    -------------------------
    Total Samples: {X.shape[0]}
    Total Features: {X.shape[1]}
    Control Samples: {np.sum(y == 0)}
    Case Samples: {np.sum(y == 1)}
    Actual Imbalance: {np.sum(y == 0) / np.sum(y == 1):.2f} : 1
    Average Zero Ratio: {np.mean(X == 0):.2%}
    True Diff Features: {len(true_diff_idx)}
    Avg Pathway Corr: {avg_pathway_corr:.2f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, va='center', linespacing=1.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Batch dataset generation
# ---------------------------------------------------------------------------
def generate_combination_datasets(
        output_dir: str = './combination_datasets'
) -> None:
    """Generate all parameter combinations of synthetic datasets.

    Parameter grid:
    - Sample sizes: 100, 200
    - Feature dimensions: 500, 2000
    - Imbalance ratios: 2:1, 5:1

    Output files per dataset:
    - {name}_dataset.xlsx: Feature matrix with label column (last column)
    - {name}_true_diff.npy: Indices of true differential features
    - {name}_validation.png: Validation visualization

    Parameters
    ----------
    output_dir : str, default='./combination_datasets'
        Directory path for output files. Created if not exists.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parameter grid
    sample_sizes = [100, 200]
    feature_nums = [500, 2000]
    imbalance_ratios = [2, 5]

    total = len(sample_sizes) * len(feature_nums) * len(imbalance_ratios)
    print("=" * 60)
    print(f"Start generating {total} combination datasets (single xlsx)...")
    print("=" * 60)

    count = 0
    for n in sample_sizes:
        for d in feature_nums:
            for ir in imbalance_ratios:
                count += 1
                print(f"\n[{count}/{total}] n={n}, d={d}, imbalance={ir}:1")

                # Generate dataset
                X, y, true_diff = generate_metabolomics_dataset(
                    n_samples=n,
                    n_features=d,
                    imbalance_ratio=ir,
                    noise_level=0.1,
                    n_pathways=5,
                    pathway_size=5,
                    random_state=RANDOM_SEED
                )

                # File naming
                base_name = f"n{n}_d{d}_ir{ir}"

                # Combine features and label into single DataFrame
                df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(d)])
                df['label'] = y

                # Save Excel file
                df.to_excel(
                    f"{output_dir}/{base_name}_dataset.xlsx",
                    index=False,
                    engine='openpyxl'
                )

                # Save true differential indices
                np.save(f"{output_dir}/{base_name}_true_diff.npy", true_diff)

                # Generate validation figure
                validate_single_dataset(
                    X, y, true_diff,
                    save_path=f"{output_dir}/{base_name}_validation.png"
                )

                print(f"  ✓ Saved: {base_name}_dataset.xlsx")
                print(f"    Actual samples: control={np.sum(y == 0)}, case={np.sum(y == 1)}")

    print("\n" + "=" * 60)
    print(f"All {total} datasets generated successfully!")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    generate_combination_datasets()
