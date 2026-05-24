"""
pipeline/ga.py
Genetic Algorithm codebook optimization for CTLE.

Hot path (MSE + assignment) is Numba-JIT compiled — fitness evaluation
over 50K weight samples takes ~microseconds per individual instead of
seconds, making 60 individuals × 200 generations tractable in seconds.

Algorithm:
  - Individual  : sorted codebook of K float64 values
  - Population  : pop_size individuals
  - Fitness     : mean squared reconstruction error (lower = better)
  - Initialisation: 1 individual from K-means warm start, rest = perturbations
  - Selection   : tournament (size=3)
  - Crossover   : arithmetic blend (BLX-α)
  - Mutation    : Gaussian perturbation on each gene with prob p_mut
  - Elitism     : top 1 individual always survives

Optional hardware-aware fitness (hw_aware=True):
  Adds a penalty for codebook values not representable in signed INT8
  fixed-point, encouraging hardware-friendly centroids.
"""

import numpy as np
from numba import njit


# ── Numba-JIT hot path ─────────────────────────────────────────────────────

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
def _eval_pop_nb(flat_fit: np.ndarray, population: np.ndarray) -> np.ndarray:
    """Evaluate MSE fitness for all individuals at once (JIT)."""
    n = population.shape[0]
    fit = np.empty(n, dtype=np.float64)
    for i in range(n):
        fit[i] = _mse_nb(flat_fit, population[i])
    return fit


# ── Python-side helpers (O(k) — negligible) ───────────────────────────────

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

def ga_codebook(
    weights: np.ndarray,
    k: int = 16,
    warm_start: np.ndarray | None = None,
    pop_size: int = 60,
    n_generations: int = 200,
    p_cross: float = 0.8,
    p_mut: float = 0.15,
    mut_sigma_frac: float = 0.05,
    alpha_blx: float = 0.3,
    tournament_k: int = 3,
    hw_aware: bool = False,
    hw_lambda: float = 0.1,
    seed: int = 42,
    max_fitness_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genetic Algorithm codebook optimisation.

    Args:
        weights:              2-D float32 weight matrix [M, N]
        k:                    codebook size (≤16 for 4-bit)
        warm_start:           initial codebook from K-means [k] (recommended)
        pop_size:             population size
        n_generations:        number of GA generations
        p_cross:              crossover probability per pair
        p_mut:                per-gene mutation probability
        mut_sigma_frac:       mutation std = mut_sigma_frac × weight_range
        alpha_blx:            BLX-α crossover extension parameter
        tournament_k:         tournament size for selection
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
    sigma    = mut_sigma_frac * w_range
    hw_scale = 127.0 / (max(abs(w_min), abs(w_max)) + 1e-12)

    # Subsample for fitness; final assignment uses full flat
    if flat.size > max_fitness_samples:
        idx = rng.choice(flat.size, size=max_fitness_samples, replace=False)
        flat_fit = np.ascontiguousarray(flat[idx])
    else:
        flat_fit = flat

    # ── Initialise population ──────────────────────────────────────────────
    population = []
    if warm_start is not None:
        population.append(np.sort(warm_start.astype(np.float64)))

    while len(population) < pop_size:
        if warm_start is not None:
            ind = warm_start.astype(np.float64) + rng.normal(0, sigma, size=k)
        else:
            ind = rng.uniform(w_min, w_max, size=k)
        population.append(np.sort(np.clip(ind, w_min, w_max)))

    population = np.ascontiguousarray(population)   # [pop_size, k]

    # ── Initial fitness ────────────────────────────────────────────────────
    fit = _eval_pop_nb(flat_fit, population)
    if hw_aware:
        fit = np.array([
            _fitness(flat_fit, population[i], True, hw_lambda, hw_scale)
            for i in range(pop_size)
        ])

    best_idx = int(np.argmin(fit))
    best_ind = population[best_idx].copy()
    best_fit = fit[best_idx]

    # ── Evolution loop ─────────────────────────────────────────────────────
    for _ in range(n_generations):
        new_pop = [best_ind.copy()]   # elitism

        while len(new_pop) < pop_size:
            def tournament():
                contenders = rng.integers(0, pop_size, size=tournament_k)
                return population[contenders[np.argmin(fit[contenders])]].copy()

            p1, p2 = tournament(), tournament()

            # BLX-α crossover
            if rng.random() < p_cross:
                lo    = np.minimum(p1, p2) - alpha_blx * np.abs(p1 - p2)
                hi    = np.maximum(p1, p2) + alpha_blx * np.abs(p1 - p2)
                child = rng.uniform(lo, hi)
            else:
                child = p1.copy()

            # Gaussian mutation
            mask = rng.random(size=k) < p_mut
            child[mask] += rng.normal(0, sigma, size=mask.sum())
            child = np.sort(np.clip(child, w_min, w_max))
            new_pop.append(child)

        population = np.ascontiguousarray(new_pop)
        fit = _eval_pop_nb(flat_fit, population)
        if hw_aware:
            fit = np.array([
                _fitness(flat_fit, population[i], True, hw_lambda, hw_scale)
                for i in range(pop_size)
            ])

        gen_best = int(np.argmin(fit))
        if fit[gen_best] < best_fit:
            best_fit = fit[gen_best]
            best_ind = population[gen_best].copy()

    codebook = best_ind.astype(np.float32)
    indices  = _assign_nb(flat, best_ind).astype(np.uint8)
    indices  = indices.reshape(weights.shape)
    return codebook, indices
