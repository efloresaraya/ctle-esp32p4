"""
pipeline/de.py
Differential Evolution codebook optimization for CTLE.

The optimizer uses a classic DE/rand/1/bin strategy over the K centroid
values. Fitness and final assignment are Numba-JIT compiled, matching the GA
and PSO modules while keeping the Python-side DE loop easy to inspect.

Algorithm:
  - Individual     : sorted codebook of K float64 values
  - Population     : pop_size individuals
  - Fitness        : mean squared reconstruction error (lower = better)
  - Initialisation : K-means warm start + perturbations
  - Mutation       : v = x_a + F * (x_b - x_c)
  - Crossover      : binomial crossover with probability CR
  - Selection      : one-to-one greedy replacement

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


# ── Python-side helpers ────────────────────────────────────────────────────

def _hw_penalty(codebook: np.ndarray, scale: float) -> float:
    """Penalty for values that round poorly under INT8 fixed-point."""
    cb_scaled = codebook * scale
    cb_int8 = np.round(cb_scaled).clip(-127, 127)
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


def _evaluate(
    flat_fit: np.ndarray,
    population: np.ndarray,
    hw_aware: bool,
    hw_lambda: float,
    hw_scale: float,
) -> np.ndarray:
    fit = _eval_pop_nb(flat_fit, population)
    if hw_aware:
        fit = np.array([
            _fitness(flat_fit, population[i], True, hw_lambda, hw_scale)
            for i in range(population.shape[0])
        ])
    return fit


# ── Public API ─────────────────────────────────────────────────────────────

def de_codebook(
    weights: np.ndarray,
    k: int = 16,
    warm_start: np.ndarray | None = None,
    pop_size: int = 60,
    n_generations: int = 200,
    mutation_factor: float = 0.7,
    crossover_rate: float = 0.9,
    init_sigma_frac: float = 0.05,
    hw_aware: bool = False,
    hw_lambda: float = 0.1,
    seed: int = 42,
    max_fitness_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Differential Evolution codebook optimisation.

    Args:
        weights:              2-D float32 weight matrix [M, N]
        k:                    codebook size (≤16 for 4-bit)
        warm_start:           initial codebook from K-means [k] (recommended)
        pop_size:             population size (must be at least 4)
        n_generations:        number of DE generations
        mutation_factor:      differential weight F
        crossover_rate:       binomial crossover probability CR
        init_sigma_frac:      warm-start perturbation std as fraction of range
        hw_aware:             enable hardware-aware fitness penalty
        hw_lambda:            weight of hardware penalty term
        seed:                 RNG seed for reproducibility
        max_fitness_samples:  subsample weights for fitness (speed vs accuracy)

    Returns:
        codebook:  float32 [k] — optimised centroid values
        indices:   uint8  [M, N] — per-weight codebook index
    """
    if pop_size < 4:
        raise ValueError("DE requires pop_size >= 4")

    rng = np.random.default_rng(seed)
    flat = weights.reshape(-1).astype(np.float64)
    w_min, w_max = flat.min(), flat.max()
    w_range = w_max - w_min + 1e-12
    sigma = init_sigma_frac * w_range
    hw_scale = 127.0 / (max(abs(w_min), abs(w_max)) + 1e-12)

    # Subsample for fitness; final assignment uses full flat.
    if flat.size > max_fitness_samples:
        idx = rng.choice(flat.size, size=max_fitness_samples, replace=False)
        flat_fit = np.ascontiguousarray(flat[idx])
    else:
        flat_fit = np.ascontiguousarray(flat)

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

    population = np.ascontiguousarray(population)
    fit = _evaluate(flat_fit, population, hw_aware, hw_lambda, hw_scale)

    # ── DE/rand/1/bin loop ─────────────────────────────────────────────────
    all_indices = np.arange(pop_size)
    for _ in range(n_generations):
        trials = np.empty_like(population)

        for i in range(pop_size):
            candidates = all_indices[all_indices != i]
            a, b, c = rng.choice(candidates, size=3, replace=False)

            mutant = population[a] + mutation_factor * (population[b] - population[c])
            mutant = np.clip(mutant, w_min, w_max)

            cross = rng.random(k) < crossover_rate
            cross[rng.integers(0, k)] = True
            trial = np.where(cross, mutant, population[i])
            trials[i] = np.sort(np.clip(trial, w_min, w_max))

        trials = np.ascontiguousarray(trials)
        trial_fit = _evaluate(flat_fit, trials, hw_aware, hw_lambda, hw_scale)
        improved = trial_fit < fit
        population[improved] = trials[improved]
        fit[improved] = trial_fit[improved]

    best_idx = int(np.argmin(fit))
    codebook64 = population[best_idx]
    codebook = codebook64.astype(np.float32)
    indices = _assign_nb(flat, codebook64).astype(np.uint8)
    indices = indices.reshape(weights.shape)
    return codebook, indices
