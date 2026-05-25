"""
pipeline/quantize.py
K-means codebook quantization for CTLE.

Given a weight tensor W (float32), produces:
  - codebook: np.ndarray [K] float32  — the K centroids (LUT)
  - indices:  np.ndarray [M, N] uint8 — 4-bit indices (values 0..K-1)
"""
import numpy as np


def kmeans_codebook(
    weights: np.ndarray,
    k: int = 16,
    n_iters: int = 300,
    tol: float = 1e-6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1-D K-means on the flattened weight tensor.

    Args:
        weights:  2-D float32 array [M, N]
        k:        codebook size (≤16 for 4-bit CTLE, ≤32 for 5-bit CTLE-5b)
        n_iters:  maximum K-means iterations
        tol:      convergence tolerance on centroid shift
        seed:     random seed for reproducibility

    Returns:
        codebook: float32 array [k]        — centroid values
        indices:  uint8  array [M, N]      — per-weight codebook index (0..k-1)
    """
    assert k <= 32, "5-bit indices support at most 32 centroids (4-bit CTLE uses k≤16)"
    assert weights.ndim == 2, "weights must be 2-D [M, N]"

    rng = np.random.default_rng(seed)
    flat = weights.reshape(-1).astype(np.float64)

    # Initialise centroids at evenly-spaced percentiles (robust to outliers)
    percentiles = np.linspace(0, 100, k)
    centroids = np.percentile(flat, percentiles).astype(np.float64)

    labels = np.zeros(flat.shape, dtype=np.uint8)

    for iteration in range(n_iters):
        # Assignment: each weight → nearest centroid
        distances = np.abs(flat[:, None] - centroids[None, :])   # [N_w, k]
        new_labels = np.argmin(distances, axis=1).astype(np.uint8)

        # Update: centroid = mean of assigned weights
        new_centroids = np.array(
            [
                flat[new_labels == c].mean() if np.any(new_labels == c) else centroids[c]
                for c in range(k)
            ],
            dtype=np.float64,
        )

        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        labels = new_labels

        if shift < tol:
            break

    codebook = centroids.astype(np.float32)
    indices  = labels.reshape(weights.shape)
    return codebook, indices


def pack_nibbles(indices: np.ndarray) -> bytes:
    """
    Pack a 2-D uint8 index array (values 0..15) into 4-bit nibbles.
    Layout: byte[i] = low_nibble | (high_nibble << 4)
    Row-major order; last byte padded with 0 if total count is odd.

    Args:
        indices: uint8 array [M, N] with values in [0, 15]

    Returns:
        Packed bytes of length ceil(M*N / 2)
    """
    flat = indices.reshape(-1).astype(np.uint8)
    if flat.size % 2 == 1:
        flat = np.append(flat, np.uint8(0))   # pad
    low  = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).tobytes()


def reconstruction_error(weights: np.ndarray, codebook: np.ndarray, indices: np.ndarray) -> float:
    """Return relative MSE between original and quantized weights."""
    reconstructed = codebook[indices]
    mse  = float(np.mean((weights - reconstructed) ** 2))
    denom = float(np.mean(weights ** 2)) + 1e-12
    return mse / denom
