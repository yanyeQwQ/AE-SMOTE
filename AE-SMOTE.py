"""
Autoencoder-DBSCAN-SMOTE (AE-DBSCAN-SMOTE) for High-Dimensional Imbalanced Data
================================================================================
Supplementary code for the manuscript:
A Stable Feature Screening Framework Integrating Autoencoder and SMOTE for Imbalanced Metabolomics

Purpose
-------
An improved synthetic minority over-sampling technique that integrates
F-statistic based feature redundancy removal, autoencoder-driven dimensionality
reduction, DBSCAN-based noise filtering in latent space, and standard SMOTE
interpolation. The method is specially designed for high-dimensional imbalanced
omics data (e.g. metabolomics, transcriptomics).

Reproducibility
---------------
All random seeds are fixed to ensure full reproducibility of results.

Dependencies
------------
- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- tensorflow >= 2.8.0

Notes
-----
- Input features are expected to be numerical.
- The minority class is automatically identified from label distribution.
- All computations are performed on numpy arrays; DataFrame/Series inputs
  are preserved in output for convenience.
"""

import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# Global reproducibility configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_global_seed(seed=RANDOM_SEED):
    """Set random seeds for all involved libraries to ensure reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Global random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# Initialize seed on module import
set_global_seed()


# ---------------------------------------------------------------------------
# Core utility functions
# ---------------------------------------------------------------------------
def compute_f_statistic(X, y):
    """Compute F-statistic values for feature ranking via ANOVA.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Binary class labels.

    Returns
    -------
    f_values : ndarray of shape (n_features,)
        F-statistic score for each feature.
    """
    f_values, _ = f_classif(X, y)
    return f_values


def remove_redundant_features(X, y, delta=0.0, corr_threshold=0.9):
    """Remove redundant features based on F-statistic ranking and correlation.

    Features are first filtered by F-statistic threshold, then ranked by
    F-statistic in descending order. Highly correlated features with
    lower F-statistic are removed iteratively.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Binary class labels.
    delta : float, default=0.0
        Minimum F-statistic threshold for feature retention.
    corr_threshold : float, default=0.9
        Pearson correlation threshold for redundancy detection.

    Returns
    -------
    mask : ndarray of shape (n_features,)
        Binary mask where 1 indicates retained features, 0 indicates removed.
    n_redundant : int
        Total number of removed redundant features.
    """
    f_values = compute_f_statistic(X, y)
    selected_indices = np.where(f_values > delta)[0]

    if len(selected_indices) == 0:
        return np.ones(X.shape[1]), 0

    # Sort features by F-statistic (descending)
    sorted_indices = sorted(selected_indices, key=lambda i: f_values[i], reverse=True)
    X_selected = X[:, sorted_indices]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_selected.T)

    # Greedy redundancy removal
    non_redundant = []
    for i in range(len(sorted_indices)):
        is_redundant = False
        for j in non_redundant:
            if abs(corr_matrix[i, j]) > corr_threshold:
                is_redundant = True
                break
        if not is_redundant:
            non_redundant.append(i)

    final_indices = [sorted_indices[i] for i in non_redundant]
    n_redundant = len(sorted_indices) - len(final_indices)

    mask = np.zeros(X.shape[1])
    mask[final_indices] = 1
    return mask, n_redundant


def build_autoencoder(input_dim, encoding_dim, l2_reg=0.01, dropout_rate=0.4):
    """Build a single-layer autoencoder with L2 regularization and dropout.

    The architecture consists of one ReLU-activated encoding layer with
    dropout and one sigmoid-activated decoding layer.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    encoding_dim : int
        Dimension of latent embedding space.
    l2_reg : float, default=0.01
        L2 regularization coefficient for kernel weights.
    dropout_rate : float, default=0.4
        Dropout rate applied to the encoding layer.

    Returns
    -------
    autoencoder : Model
        Full autoencoder model (input -> reconstruction).
    encoder : Model
        Encoder sub-model (input -> latent embedding).
    decoder : Model
        Decoder sub-model (latent embedding -> reconstruction).
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoder pathway
    encoded = Dense(
        encoding_dim,
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(input_layer)
    encoded = Dropout(dropout_rate)(encoded)

    # Decoder pathway
    decoded = Dense(
        input_dim,
        activation="sigmoid",
        kernel_regularizer=l2(l2_reg)
    )(encoded)

    # Full autoencoder
    autoencoder = Model(input_layer, decoded)

    # Standalone encoder
    encoder = Model(input_layer, encoded)

    # Standalone decoder
    latent_input = Input(shape=(encoding_dim,))
    decoder = Model(latent_input, autoencoder.layers[-1](latent_input))

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder, decoder


def detect_noise_dbscan(X, eps=0.5, min_samples=5):
    """Detect noise samples using DBSCAN clustering.

    Samples labeled as -1 by DBSCAN are classified as noise.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix (typically latent embeddings).
    eps : float, default=0.5
        Maximum distance between two samples for neighborhood membership.
    min_samples : int, default=5
        Minimum number of samples in a neighborhood to form a core point.

    Returns
    -------
    noise_mask : ndarray of bool, shape (n_samples,)
        Boolean mask where True indicates a noise sample.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    noise_mask = cluster_labels == -1
    return noise_mask


def smote_generate(X, n_generate, k_neighbors=5):
    """Generate synthetic samples using standard SMOTE interpolation.

    For each synthetic sample, a random minority sample and one of its
    k-nearest neighbors are selected, and interpolation is performed
    with a random coefficient in [0, 1].

    Parameters
    ----------
    X : array-like of shape (n_minority, n_features)
        Minority class samples in feature space.
    n_generate : int
        Number of synthetic samples to generate.
    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE interpolation.

    Returns
    -------
    synthetic : ndarray of shape (n_generate, n_features)
        Generated synthetic minority samples.
    """
    synthetic = []
    n_samples = len(X)
    n_neighbors = min(k_neighbors, n_samples)

    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)

    for _ in range(n_generate):
        # Randomly select a base sample
        idx = np.random.randint(0, n_samples)
        point = X[idx]

        # Randomly select one of its neighbors
        neighbors = knn.kneighbors([point], return_distance=False)[0]
        neighbor_idx = random.choice(neighbors)
        neighbor = X[neighbor_idx]

        # Linear interpolation
        alpha = np.random.rand()
        new_sample = point + alpha * (neighbor - point)
        synthetic.append(new_sample)

    return np.array(synthetic)


# ---------------------------------------------------------------------------
# Main over-sampling function
# ---------------------------------------------------------------------------
def ae_dbscan_smote(X, y, eps=20, min_samples=7, encoding_dim=70,
                    k_neighbors=5, delta=0.0, corr_threshold=0.9,
                    l2_reg=0.01, dropout_rate=0.4, epochs=150,
                    batch_size=64, patience=10, verbose=True):
    """Perform AE-DBSCAN-SMOTE over-sampling on imbalanced binary data.

    Workflow:
    1.  Separate minority and majority classes
    2.  Remove redundant features based on F-statistic and correlation
    3.  Normalize minority class features with MinMax scaling
    4.  Train autoencoder on minority class and obtain latent embeddings
    5.  Filter noise samples from latent space via DBSCAN
    6.  Generate synthetic samples via SMOTE in filtered latent space
    7.  Decode synthetic samples back to original feature space
    8.  Apply feature redundancy mask and inverse scaling
    9.  Combine original and synthetic samples into balanced dataset

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : pd.Series or np.ndarray of shape (n_samples,)
        Binary class labels.
    eps : float, default=20
        DBSCAN epsilon parameter (neighborhood radius) in latent space.
    min_samples : int, default=7
        DBSCAN min_samples parameter for core point definition.
    encoding_dim : int, default=70
        Dimension of autoencoder latent space.
    k_neighbors : int, default=5
        Number of neighbors for SMOTE interpolation.
    delta : float, default=0.0
        F-statistic threshold for feature filtering.
    corr_threshold : float, default=0.9
        Pearson correlation threshold for feature redundancy removal.
    l2_reg : float, default=0.01
        L2 regularization strength for autoencoder.
    dropout_rate : float, default=0.4
        Dropout rate for autoencoder encoding layer.
    epochs : int, default=150
        Maximum training epochs for autoencoder.
    batch_size : int, default=64
        Batch size for autoencoder training.
    patience : int, default=10
        Early stopping patience (training loss monitoring).
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    X_resampled : pd.DataFrame or np.ndarray
        Over-sampled feature matrix.
    y_resampled : pd.Series or np.ndarray
        Over-sampled class labels.
    """
    # Preserve input format information
    is_dataframe = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, pd.Series)
    columns = X.columns if is_dataframe else None
    y_name = y.name if is_series and hasattr(y, "name") else None

    # Convert to numpy arrays
    X_np = np.asarray(X)
    y_np = np.asarray(y)

    # Identify minority and majority classes automatically
    classes, counts = np.unique(y_np, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]

    X_minority = X_np[y_np == minority_class]
    X_majority = X_np[y_np == majority_class]
    n_minority = len(X_minority)
    n_majority = len(X_majority)
    n_generate = n_majority - n_minority

    if n_generate <= 0:
        if verbose:
            print("Skipping over-sampling: minority size >= majority size")
        return X, y

    # Step 1: Feature redundancy removal
    feature_mask, n_redundant = remove_redundant_features(
        X_np, y_np, delta=delta, corr_threshold=corr_threshold
    )
    if verbose:
        print(f"Redundant features removed: {n_redundant}")

    # Step 2: Normalize minority class data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_min_scaled = scaler.fit_transform(X_minority)

    # Step 3: Train autoencoder
    autoencoder, encoder, decoder = build_autoencoder(
        input_dim=X_min_scaled.shape[1],
        encoding_dim=encoding_dim,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate
    )

    early_stop = EarlyStopping(monitor="loss", patience=patience)
    autoencoder.fit(
        X_min_scaled, X_min_scaled,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # Obtain latent embeddings
    X_embedded = encoder.predict(X_min_scaled, verbose=0)
    if verbose:
        print(f"Latent embedding dimension: {X_embedded.shape[1]}")

    # Step 4: DBSCAN noise filtering in latent space
    if verbose:
        print("Running DBSCAN noise detection in latent space...")
    noise_mask = detect_noise_dbscan(X_embedded, eps=eps, min_samples=min_samples)
    X_embedded_clean = X_embedded[~noise_mask]

    if verbose:
        print(f"Samples before noise filtering: {len(X_embedded)}")
        print(f"Samples after noise filtering: {len(X_embedded_clean)}")

    if len(X_embedded_clean) == 0:
        if verbose:
            print("Warning: no samples remaining after noise filtering. "
                  "Returning original data.")
        return X, y

    # Step 5: SMOTE generation in latent space
    if verbose:
        print(f"Generating {n_generate} synthetic samples via SMOTE...")
    synthetic_embedded = smote_generate(
        X_embedded_clean, n_generate, k_neighbors=k_neighbors
    )

    # Step 6: Decode and post-process synthetic samples
    synthetic_decoded = decoder.predict(synthetic_embedded, verbose=0)
    synthetic = scaler.inverse_transform(synthetic_decoded)
    # Apply feature mask (zero out redundant features)
    synthetic = synthetic * feature_mask

    # Step 7: Combine original and synthetic data
    X_resampled_np = np.vstack((X_np, synthetic))
    y_resampled_np = np.hstack(
        (y_np, np.full(n_generate, minority_class))
    )

    # Restore original input format
    if is_dataframe:
        X_resampled = pd.DataFrame(
            X_resampled_np,
            columns=columns,
            index=range(len(X_resampled_np))
        )
    else:
        X_resampled = X_resampled_np

    if is_series:
        y_resampled = pd.Series(y_resampled_np, name=y_name)
    else:
        y_resampled = y_resampled_np

    if verbose:
        print("\nOver-sampling completed successfully.")
        print(f"Original dataset size: {len(X)}")
        print(f"Resampled dataset size: {len(X_resampled)}")
        print(f"Original class distribution - "
              f"{minority_class}: {n_minority}, {majority_class}: {n_majority}")
        print(f"Resampled class distribution - "
              f"{minority_class}: {sum(y_resampled == minority_class)}, "
              f"{majority_class}: {sum(y_resampled == majority_class)}")

    return X_resampled, y_resampled


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ================================================================
    # Example: Run AE-DBSCAN-SMOTE on user-provided dataset
    # Replace the file path with your own data path
    # ================================================================
    print("Loading dataset...")
    data = pd.read_excel("EndogenousSubstance_VSN.xlsx")
    data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Separate features and label (assumes last column is class label)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Encode string labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Execute over-sampling
    X_resampled, y_resampled = ae_dbscan_smote(
        X=X,
        y=y_encoded,
        eps=170,
        min_samples=7,
        encoding_dim=900,
        k_neighbors=5,
        verbose=True
    )

    # Save results
    output_data = X_resampled.copy()
    output_data["label"] = y_resampled
    output_data.to_excel("smote_resampled_EndogenousSubstance.xlsx", index=False)
    print("\nResampled dataset saved to 'smote_resampled_EndogenousSubstance.xlsx'")
