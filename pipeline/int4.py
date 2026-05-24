"""
pipeline/int4.py
INT4 quantization baselines for comparison with CTLE.

Two methods — both use 4-bit storage, different granularity of scale:

  INT4 Uniform  (TAG=2): one symmetric scale per entire tensor
    scale = max(|W|) / 7
    q     = clamp(round(W / scale), -7, 7)  →  stored as q+7 ∈ [0,14]
    dequant: W̃ = (nibble - 7) * scale

  INT4 Block-wise (TAG=3): one symmetric scale per row-group of G=32 weights
    For each group of 32 consecutive cols in row r:
      scale_g = max(|W[r, g*G : (g+1)*G]|) / 7
    q and dequant same as above but per-group scale
    Effective cost: 4 + 32/32 = 5 bits/weight  (scale overhead)

Both return the same (RelMSE, size_bytes) interface as CTLE for fair comparison.
"""

import numpy as np

_INT4_LEVELS = 7      # symmetric: range [-7, +7] → stored as [0, 14]
_INT4_OFFSET = 7      # unsigned nibble = signed_q + 7
GROUP_SIZE   = 32     # block-wise group size (cols per scale)


# ── INT4 Uniform ────────────────────────────────────────────────────────────

def int4_uniform(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per-tensor INT4 quantization.

    Args:
        weights: 2-D float32 [M, N]

    Returns:
        scale:    float32 scalar (as 1-element array)
        nibbles:  uint8 [M, N]  values in [0, 14]  (7 = zero)
    """
    w = weights.astype(np.float64)
    scale = float(np.max(np.abs(w))) / _INT4_LEVELS + 1e-12
    q = np.clip(np.round(w / scale), -_INT4_LEVELS, _INT4_LEVELS).astype(np.int8)
    nibbles = (q + _INT4_OFFSET).astype(np.uint8)
    return np.float32(scale), nibbles


def int4_uniform_dequant(scale: float, nibbles: np.ndarray) -> np.ndarray:
    """Reconstruct float32 weights from INT4U representation."""
    return (nibbles.astype(np.float32) - _INT4_OFFSET) * scale


# ── INT4 Block-wise ──────────────────────────────────────────────────────────

def int4_blockwise(
    weights: np.ndarray,
    group_size: int = GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per-group INT4 quantization (block-wise).

    Args:
        weights:    2-D float32 [M, N]
        group_size: number of columns per scale group (default 32)

    Returns:
        scales:  float32 [M, ceil(N/group_size)]  — one scale per row-group
        nibbles: uint8   [M, N]                   — values in [0, 14]
    """
    w    = weights.astype(np.float64)
    rows, cols = w.shape

    # Pad cols to multiple of group_size
    n_groups = (cols + group_size - 1) // group_size
    pad = n_groups * group_size - cols
    w_pad = np.pad(w, ((0, 0), (0, pad))) if pad > 0 else w  # [M, n_groups*G]
    w_g = w_pad.reshape(rows, n_groups, group_size)          # [M, nG, G]

    scales = (np.max(np.abs(w_g), axis=2) / _INT4_LEVELS + 1e-12).astype(np.float32)
    # [M, nG, G] / [M, nG, 1]
    q = np.clip(np.round(w_g / scales[:, :, None]), -_INT4_LEVELS, _INT4_LEVELS).astype(np.int8)
    nibbles = (q + _INT4_OFFSET).astype(np.uint8)             # [M, nG, G]
    nibbles = nibbles.reshape(rows, n_groups * group_size)[:, :cols]  # [M, N]
    return scales, nibbles


def int4_blockwise_dequant(
    scales: np.ndarray,
    nibbles: np.ndarray,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    """Reconstruct float32 weights from INT4BW representation."""
    rows, cols = nibbles.shape
    n_groups   = scales.shape[1]
    pad        = n_groups * group_size - cols
    nib_pad    = np.pad(nibbles, ((0, 0), (0, pad))) if pad > 0 else nibbles
    nib_g      = nib_pad.reshape(rows, n_groups, group_size).astype(np.float32)
    w_g        = (nib_g - _INT4_OFFSET) * scales[:, :, None]
    return w_g.reshape(rows, n_groups * group_size)[:, :cols].astype(np.float32)


# ── Reconstruction error (same interface as quantize.py) ────────────────────

def reconstruction_error_int4u(
    weights: np.ndarray,
    scale: float,
    nibbles: np.ndarray,
) -> float:
    """Relative MSE for INT4 uniform."""
    w_hat = int4_uniform_dequant(scale, nibbles)
    mse   = float(np.mean((weights - w_hat) ** 2))
    norm  = float(np.mean(weights ** 2)) + 1e-12
    return mse / norm


def reconstruction_error_int4bw(
    weights: np.ndarray,
    scales: np.ndarray,
    nibbles: np.ndarray,
    group_size: int = GROUP_SIZE,
) -> float:
    """Relative MSE for INT4 block-wise."""
    w_hat = int4_blockwise_dequant(scales, nibbles, group_size)
    mse   = float(np.mean((weights - w_hat) ** 2))
    norm  = float(np.mean(weights ** 2)) + 1e-12
    return mse / norm


# ── Effective size estimate (for fair comparison) ───────────────────────────

def int4u_size_bytes(rows: int, cols: int) -> int:
    """Storage: 4-byte scale + packed nibbles."""
    return 4 + (rows * cols + 1) // 2


def int4bw_size_bytes(rows: int, cols: int, group_size: int = GROUP_SIZE) -> int:
    """Storage: scales array + packed nibbles."""
    n_groups = (cols + group_size - 1) // group_size
    return rows * n_groups * 4 + (rows * cols + 1) // 2
