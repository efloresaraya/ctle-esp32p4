"""
pipeline/pso.py
Particle Swarm Optimization codebook optimization for CTLE.

Hot path (MSE + assignment) is Numba-JIT compiled — same acceleration
as ga.py, making 60 particles × 200 iterations tractable in seconds.

Algorithm:
  - Particle : position (codebook) + velocity in R^k
  - Swarm    : n_particles particles
  - Fitness  : mean squared reconstruction error (lower = better)
  - Initialisation: 1 particle = K-means warm start, rest = perturbations
  - Update   : v ← ω·v + c1·r1·(pbest−x) + c2·r2·(gbest−x)
               x ← clip(sort(x + v), w_min, w_max)
  - Inertia  : ω decays linearly from ω_max → ω_min over iterations

Optional hardware-aware fitness (hw_aware=True):
  Adds a penalty for codebook values not representable in signed INT8
  fixed-point, encouraging hardware-friendly centroids.
"""

import numpy as np
from numba import njit


# ── Numba-JIT hot path (shared logic with ga.py) ──────────────────────────

@njit(cache=True)
def _mse_nb(flat: np.ndarray, codebook: np.ndarray) -> float:
    """MSE between weights and nearest codebook entries (JIT)."""
    n = flat.shape[0]
    total = 0.0
    for i in range(n):
        w = flat[i]
        best = (w - codebook[0]) ** 2
        for j in range(1, codebook.shape[0]):
            d = (w - codebook[j]) ** 2
            if d < best:
                best = d
        total += best
    return total / n


@njit(cache=True)
def _assign_nb(flat: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Nearest-centroid index for each weight (JIT)."""
    n = flat.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        w = flat[i]
        best_d = (w - codebook[0]) ** 2
        best_j = 0
        for j in range(1, codebook.shape[0]):
            d = (w - codebook[j]) ** 2
            if d < best_d:
                best_d = d
                best_j = j
        out[i] = best_j
    return out


@njit(cache=True)
def _eval_swarm_nb(flat_fit: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Evaluate MSE fitness for all particles at once (JIT)."""
    n = positions.shape[0]
    fit = np.empty(n, dtype=np.float64)
    for i in range(n):
        fit[i] = _mse_nb(flat_fit, positions[i])
    return fit


# ── Python-side helpers ────────────────────────────────────────────────────

def _hw_penalty(codebook: np.ndarray, scale: float) -> float:
    """Penalty for values that round poorly under INT8 fixed-point."""
    cb_scaled = codebook * scale
    cb_int8   = np.round(cb_scaled).clip(-127, 127)
    return float(np.mean((cb_scaled - cb_int8) ** 2)) / (scale ** 2)


def _fitness(
    flat_fit: np.ndarray,
    codebook: np.ndarray,
    hw_aware: bool,
    hw_lambda: float,
    hw_scale: float,
) -> float:
    mse = _mse_nb(flat_fit, codebook)
    if hw_aware:
        mse += hw_lambda * _hw_penalty(codebook, hw_scale)
    return mse


# ── Public API ─────────────────────────────────────────────────────────────

def pso_codebook(
    weights: np.ndarray,
    k: int = 16,
    warm_start: np.ndarray | None = None,
    n_particles: int = 60,
    n_iterations: int = 200,
    omega_max: float = 0.9,
    omega_min: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    v_max_frac: float = 0.1,
    hw_aware: bool = False,
    hw_lambda: float = 0.1,
    seed: int = 42,
    max_fitness_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Particle Swarm Optimization codebook optimisation.

    Args:
        weights:              2-D float32 weight matrix [M, N]
        k:                    codebook size (≤16 for 4-bit)
        warm_start:           initial codebook from K-means [k] (recommended)
        n_particles:          swarm size
        n_iterations:         number of PSO iterations
        omega_max:            initial inertia weight (exploration)
        omega_min:            final inertia weight   (exploitation)
        c1:                   cognitive acceleration coefficient
        c2:                   social acceleration coefficient
        v_max_frac:           v_max = v_max_frac × weight_range
        hw_aware:             enable hardware-aware fitness penalty
        hw_lambda:            weight of hardware penalty term
        seed:                 RNG seed for reproducibility
        max_fitness_samples:  subsample weights for fitness (speed vs accuracy)

    Returns:
        codebook:  float32 [k] — optimised centroid values
        indices:   uint8  [M, N] — per-weight codebook index
    """
    rng = np.random.default_rng(seed)
    flat = weights.reshape(-1).astype(np.float64)
    w_min, w_max = flat.min(), flat.max()
    w_range  = w_max - w_min + 1e-12
    v_max    = v_max_frac * w_range
    hw_scale = 127.0 / (max(abs(w_min), abs(w_max)) + 1e-12)

    # Subsample for fitness; final assignment uses full flat
    if flat.size > max_fitness_samples:
        idx = rng.choice(flat.size, size=max_fitness_samples, replace=False)
        flat_fit = np.ascontiguousarray(flat[idx])
    else:
        flat_fit = flat

    # ── Initialise positions ───────────────────────────────────────────────
    positions = []
    if warm_start is not None:
        positions.append(np.sort(warm_start.astype(np.float64)))

    while len(positions) < n_particles:
        if warm_start is not None:
            sigma = 0.05 * w_range
            pos = warm_start.astype(np.float64) + rng.normal(0, sigma, size=k)
        else:
            pos = rng.uniform(w_min, w_max, size=k)
        positions.append(np.sort(np.clip(pos, w_min, w_max)))

    positions  = np.ascontiguousarray(positions)                     # [n, k]
    velocities = rng.uniform(-v_max, v_max, size=(n_particles, k))

    # ── Initial fitness ────────────────────────────────────────────────────
    fit = _eval_swarm_nb(flat_fit, positions)
    if hw_aware:
        fit = np.array([
            _fitness(flat_fit, positions[i], True, hw_lambda, hw_scale)
            for i in range(n_particles)
        ])

    pbest_pos = positions.copy()
    pbest_fit = fit.copy()
    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    # ── PSO main loop ──────────────────────────────────────────────────────
    for t in range(n_iterations):
        omega = omega_max - (omega_max - omega_min) * t / max(n_iterations - 1, 1)

        r1 = rng.random(size=(n_particles, k))
        r2 = rng.random(size=(n_particles, k))

        velocities = (
            omega * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos[None, :] - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)
        positions  = np.sort(np.clip(positions + velocities, w_min, w_max), axis=1)
        positions  = np.ascontiguousarray(positions)

        fit = _eval_swarm_nb(flat_fit, positions)
        if hw_aware:
            fit = np.array([
                _fitness(flat_fit, positions[i], True, hw_lambda, hw_scale)
                for i in range(n_particles)
            ])

        improved = fit < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fit[improved]

        gen_best = int(np.argmin(pbest_fit))
        if pbest_fit[gen_best] < gbest_fit:
            gbest_fit = pbest_fit[gen_best]
            gbest_pos = pbest_pos[gen_best].copy()

    codebook = gbest_pos.astype(np.float32)
    indices  = _assign_nb(flat, gbest_pos).astype(np.uint8)
    indices  = indices.reshape(weights.shape)
    return codebook, indices
