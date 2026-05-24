#!/usr/bin/env python3
"""
compress.py
Compress a safetensors model into a CTLE binary blob.

Usage:
    python scripts/compress.py \
        --input  models/stories15M/model.safetensors \
        --config models/stories15M/config.json \
        --output weights/stories15M_ctle_k16.bin \
        --k 16 \
        --method kmeans        # default
        --method ga            # Genetic Algorithm (slower, better)
        --method pso           # Particle Swarm (slower, often better)
        --method de            # Differential Evolution (slower, often better)
        --method pctle_de      # P-CTLE with DE-optimised weight codebooks
        --hw-aware             # add INT8 fixed-point penalty (GA/PSO/DE only)

The script:
  1. Loads all tensors from the safetensors file.
  2. Applies codebook quantization (K-means, GA, PSO, or DE) to all large
     weight matrices (embedding + attention + FFN projections).
  3. Keeps small tensors (RMSNorm gains) in full FP32.
  4. Writes the CTLE binary format v2 (see pipeline/export.py for spec).
  5. Prints per-tensor reconstruction error and a final summary table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.quantize import kmeans_codebook, reconstruction_error
from pipeline.ga       import ga_codebook
from pipeline.pso      import pso_codebook
from pipeline.de       import de_codebook
from pipeline.int4     import (int4_uniform, int4_blockwise,
                                reconstruction_error_int4u,
                                reconstruction_error_int4bw,
                                int4u_size_bytes, int4bw_size_bytes,
                                GROUP_SIZE as INT4_GROUP_SIZE)
from pipeline.export   import write_ctle_bin

# Tensors with fewer elements than this threshold are kept as FP32
_SMALL_TENSOR_THRESHOLD = 1_024   # e.g. RMSNorm gains [dim=288]

_METHODS = ("kmeans", "ga", "pso", "de", "int4_uniform", "int4_blockwise", "pctle", "pctle_de")
_INT4_METHODS  = {"int4_uniform", "int4_blockwise"}
_PCTLE_METHODS = {"pctle", "pctle_de"}   # P-CTLE: optimized weights + TAG_PCTLE


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load all tensors from a safetensors file as float32 numpy arrays."""
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).float().numpy()
    return tensors


def _quantize_ctle(
    w2d: np.ndarray,
    k: int,
    method: str,
    hw_aware: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (codebook [k], indices [M,N]) for CTLE methods."""
    if method == "kmeans":
        return kmeans_codebook(w2d, k=k, seed=seed)

    # GA and PSO both benefit from a K-means warm start
    warm_cb, _ = kmeans_codebook(w2d, k=k, seed=seed)

    if method == "ga":
        return ga_codebook(
            w2d,
            k=k,
            warm_start=warm_cb,
            hw_aware=hw_aware,
            seed=seed,
        )
    elif method == "pso":
        return pso_codebook(
            w2d,
            k=k,
            warm_start=warm_cb,
            hw_aware=hw_aware,
            seed=seed,
        )
    elif method == "de":
        return de_codebook(
            w2d,
            k=k,
            warm_start=warm_cb,
            hw_aware=hw_aware,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")


def compress(
    input_path: Path,
    config_path: Path,
    output_path: Path,
    k: int,
    method: str,
    hw_aware: bool,
    seed: int,
) -> None:
    t0 = time.perf_counter()

    is_int4  = method in _INT4_METHODS
    is_pctle = method in _PCTLE_METHODS
    hw_tag = " +HW" if (hw_aware and method not in {"kmeans"} | _INT4_METHODS) else ""
    print(f"\n{'='*64}")
    print(f"  CTLE Compression Pipeline")
    print(f"{'='*64}")
    print(f"  Input   : {input_path}")
    print(f"  Config  : {config_path}")
    print(f"  Output  : {output_path}")
    if not is_int4:
        print(f"  K       : {k} centroids (4-bit)")
    method_label = {
        "pctle": "PCTLE (PSO weight opt + product-LUT tag)",
        "pctle_de": "PCTLE-DE (DE weight opt + product-LUT tag)",
    }.get(method, method.upper())
    print(f"  Method  : {method_label}{hw_tag}")
    if not is_int4:
        print(f"  Seed    : {seed}\n")
    else:
        print()

    config = json.loads(config_path.read_text())
    raw    = load_safetensors(input_path)

    print(f"  Loaded {len(raw)} tensors from safetensors\n")

    processed: dict = {}
    total_fp32_bytes = 0
    total_ctle_bytes = 0

    hdr = f"  {'Tensor':<50} {'Shape':<20} {'Type':<6} {'RelMSE':>10}"
    print(hdr)
    print(f"  {'-'*50} {'-'*20} {'-'*6} {'-'*10}")

    for name, weights in raw.items():
        fp32_bytes = weights.size * 4
        total_fp32_bytes += fp32_bytes

        is_small = weights.ndim < 2 or weights.size < _SMALL_TENSOR_THRESHOLD

        if is_small:
            processed[name] = weights.astype(np.float32)
            total_ctle_bytes += fp32_bytes
            shape_str = str(weights.shape)
            print(f"  {name:<50} {shape_str:<20} {'F32':<6} {'—':>10}")
        elif method == "int4_uniform":
            w2d = weights.reshape(weights.shape[0], -1).astype(np.float32)
            scale, nibbles = int4_uniform(w2d)
            rel_mse = reconstruction_error_int4u(w2d, scale, nibbles)
            processed[name] = ("int4u", scale, nibbles)
            rows, cols = w2d.shape
            q_bytes = int4u_size_bytes(rows, cols)
            total_ctle_bytes += q_bytes
            shape_str = str(weights.shape)
            print(f"  {name:<50} {shape_str:<20} {'INT4U':<6} {rel_mse:>10.6f}")
        elif method == "int4_blockwise":
            w2d = weights.reshape(weights.shape[0], -1).astype(np.float32)
            scales, nibbles = int4_blockwise(w2d)
            rel_mse = reconstruction_error_int4bw(w2d, scales, nibbles)
            processed[name] = ("int4bw", scales, nibbles, INT4_GROUP_SIZE)
            rows, cols = w2d.shape
            q_bytes = int4bw_size_bytes(rows, cols)
            total_ctle_bytes += q_bytes
            shape_str = str(weights.shape)
            print(f"  {name:<50} {shape_str:<20} {'INT4BW':<6} {rel_mse:>10.6f}")
        elif is_pctle:
            # P-CTLE: optimize weight codebook, write TAG_PCTLE for runtime product-LUT.
            w2d = weights.reshape(weights.shape[0], -1).astype(np.float32)
            weight_method = "de" if method == "pctle_de" else "pso"
            codebook, indices = _quantize_ctle(w2d, k, weight_method, hw_aware, seed)
            rel_mse = reconstruction_error(w2d, codebook, indices)
            processed[name] = ("pctle", codebook, indices)
            ctle_bytes = 64 + (w2d.shape[0] * w2d.shape[1] + 1) // 2
            total_ctle_bytes += ctle_bytes
            shape_str = str(weights.shape)
            print(f"  {name:<50} {shape_str:<20} {'PCTLE':<6} {rel_mse:>10.6f}")
        else:
            w2d = weights.reshape(weights.shape[0], -1).astype(np.float32)
            codebook, indices = _quantize_ctle(w2d, k, method, hw_aware, seed)
            rel_mse = reconstruction_error(w2d, codebook, indices)

            processed[name] = (codebook, indices)
            ctle_bytes = 64 + (w2d.shape[0] * w2d.shape[1] + 1) // 2
            total_ctle_bytes += ctle_bytes

            shape_str = str(weights.shape)
            print(f"  {name:<50} {shape_str:<20} {'CTLE':<6} {rel_mse:>10.6f}")

    elapsed = time.perf_counter() - t0
    ratio   = total_fp32_bytes / max(total_ctle_bytes, 1)

    print(f"\n{'='*64}")
    print(f"  FP32 footprint : {total_fp32_bytes / 1e6:>8.2f} MB")
    print(f"  CTLE footprint : {total_ctle_bytes / 1e6:>8.2f} MB")
    print(f"  Compression    : {ratio:>8.2f}×")
    print(f"  Time           : {elapsed:>8.1f} s")
    print(f"{'='*64}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ctle_bin(output_path, config, processed)
    print(f"  Written → {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress safetensors model to CTLE binary"
    )
    parser.add_argument("--input",    required=True,
                        help="Path to model.safetensors")
    parser.add_argument("--config",   required=True,
                        help="Path to config.json")
    parser.add_argument("--output",   required=True,
                        help="Output .bin path")
    parser.add_argument("--k",        type=int, default=16,
                        help="Codebook size (default 16)")
    parser.add_argument("--method",   choices=_METHODS, default="kmeans",
                        help="Codebook optimizer: kmeans | ga | pso | de | int4_uniform | int4_blockwise | pctle | pctle_de  (default: kmeans)")
    parser.add_argument("--hw-aware", action="store_true",
                        help="GA/PSO/DE: add INT8 fixed-point penalty (hardware-aware)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="RNG seed for GA/PSO (default 42)")
    args = parser.parse_args()

    compress(
        input_path  = Path(args.input),
        config_path = Path(args.config),
        output_path = Path(args.output),
        k           = args.k,
        method      = args.method,
        hw_aware    = args.hw_aware,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
