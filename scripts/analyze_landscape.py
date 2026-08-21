#!/usr/bin/env python3
"""
analyze_landscape.py
Regenerate the two offline analyses reported in the ESL manuscript.

1. Optimization-landscape diagnostics (paper Table II)
   -------------------------------------------------
   For each tracked tensor, report the weight-distribution kurtosis, the
   99.9th percentile in units of sigma, and the spread of final relative MSE
   across N randomly-initialised 1-D K-means restarts.

   These explain *why* GA/PSO only improve the token embedding: a heavy tail
   splits the MSE landscape into distinct basins (centroids must be spent on
   the dense core OR on the tail), so a single K-means run lands in a
   different basin depending on initialisation.  The projection matrices are
   near-unimodal and K-means is already near-optimal there.

2. Metadata overhead (paper Fig. 1b)
   ---------------------------------
   Bytes that must be read alongside the weight stream for each format:
   INT4 block-wise carries one FP32 scale per (row, 32-column group), while
   CTLE carries one K-entry FP32 codebook per tensor regardless of size.

Usage
-----
    python -m scripts.analyze_landscape
    python -m scripts.analyze_landscape --restarts 20 --sample 100000

Outputs (written to results/):
    landscape_stats.csv     kurtosis / tail / restart-spread per tensor
    metadata_overhead.csv   scale-table vs codebook bytes per format
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy.stats import kurtosis
from sklearn.cluster import KMeans

# Tensors reported in Table II: the token embedding plus three representative
# projection matrices spanning the attention and feed-forward blocks.
TRACKED = {
    "tok_emb": "tok_embeddings.weight",
    "wq.0": "layers.0.attention.wq.weight",
    "wk.0": "layers.0.attention.wk.weight",
    "w1.0": "layers.0.feed_forward.w1.weight",
}

GROUP_SIZE = 32  # INT4BW columns per scale
BYTES_PER_FP32 = 4


def landscape_stats(model_path: Path, k: int, restarts: int,
                    sample: int, seed: int) -> list[dict]:
    """Kurtosis, tail ratio and K-means restart spread per tracked tensor."""
    rng = np.random.default_rng(seed)
    rows = []

    with safe_open(str(model_path), "numpy") as f:
        for short, key in TRACKED.items():
            w = f.get_tensor(key).astype(np.float64).ravel()

            # Distribution shape. Fisher=False => Gaussian reference is 3.0.
            kurt = float(kurtosis(w, fisher=False))
            tail = float(np.percentile(np.abs(w), 99.9) / w.std())

            # Restart spread: how much does the final RelMSE depend on the
            # random initialisation?  A wide spread implies multiple basins,
            # which is exactly where a warm-started global search can help.
            sub = rng.choice(w, size=min(sample, w.size), replace=False)
            sub = sub.reshape(-1, 1)
            denom = float((sub ** 2).mean())

            mses = []
            for s in range(restarts):
                km = KMeans(n_clusters=k, n_init=1, init="random",
                            random_state=s, max_iter=300).fit(sub)
                c = km.cluster_centers_.ravel()
                assign = np.abs(sub - c[None, :]).argmin(axis=1)
                mses.append(float(((sub.ravel() - c[assign]) ** 2).mean() / denom))

            mses = np.array(mses)
            spread = 100.0 * (mses.max() - mses.min()) / mses.min()

            rows.append({
                "tensor": short,
                "n_weights": int(w.size),
                "kurtosis": round(kurt, 1),
                "p99.9_over_sigma": round(tail, 1),
                "relmse_best": round(float(mses.min()), 5),
                "relmse_worst": round(float(mses.max()), 5),
                "restart_spread_pct": round(float(spread), 1),
            })
            print(f"  {short:9s} N={w.size:>9,d}  kurt={kurt:6.1f}  "
                  f"p99.9={tail:5.1f}s  spread={spread:5.1f}%")
    return rows


def metadata_overhead(model_path: Path) -> list[dict]:
    """Per-model metadata bytes: INT4BW scale tables vs CTLE codebooks."""
    n_scales = 0
    n_tensors = 0
    largest_dense_bytes = 0

    with safe_open(str(model_path), "numpy") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if t.ndim != 2:
                continue
            rows_, cols = t.shape
            n_tensors += 1
            n_scales += rows_ * ((cols + GROUP_SIZE - 1) // GROUP_SIZE)
            largest_dense_bytes = max(largest_dense_bytes, t.size * BYTES_PER_FP32)

    int4bw_bytes = n_scales * BYTES_PER_FP32
    out = [{
        "format": "FP32 expand-then-compute",
        "metadata_desc": "dense reconstructed W in SRAM (largest tensor)",
        "n_tensors": n_tensors,
        "metadata_bytes": largest_dense_bytes,
        "metadata_human": f"{largest_dense_bytes / 1e6:.1f} MB",
        "ratio_vs_int4bw": round(largest_dense_bytes / int4bw_bytes, 1),
    }, {
        "format": "INT4 block-wise",
        "metadata_desc": f"{n_scales:,} FP32 group scales (group={GROUP_SIZE})",
        "n_tensors": n_tensors,
        "metadata_bytes": int4bw_bytes,
        "metadata_human": f"{int4bw_bytes / 1e6:.2f} MB",
        "ratio_vs_int4bw": 1.0,
    }]

    for k in (16, 32):
        cb = n_tensors * k * BYTES_PER_FP32
        out.append({
            "format": f"CTLE (K={k})",
            "metadata_desc": f"{n_tensors} codebooks x {k * BYTES_PER_FP32} B",
            "n_tensors": n_tensors,
            "metadata_bytes": cb,
            "metadata_human": f"{cb / 1024:.1f} kB",
            "ratio_vs_int4bw": round(int4bw_bytes / cb),
        })

    for r in out:
        print(f"  {r['format']:26s} {r['metadata_human']:>10s}  "
              f"({r['metadata_desc']})")
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"  -> {path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path,
                    default=root / "models/stories15M/model.safetensors")
    ap.add_argument("--results", type=Path, default=root / "results")
    ap.add_argument("--k", type=int, default=16,
                    help="codebook size for the restart study (default: 16)")
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--sample", type=int, default=50000,
                    help="weights subsampled per tensor (default: 50000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}\n"
                         "Download it separately (see README).")

    print(f"Optimization-landscape diagnostics (K={args.k}, "
          f"{args.restarts} restarts, {args.sample:,} weights/tensor)")
    write_csv(args.results / "landscape_stats.csv",
              landscape_stats(args.model, args.k, args.restarts,
                              args.sample, args.seed))

    print("\nMetadata overhead per model")
    write_csv(args.results / "metadata_overhead.csv",
              metadata_overhead(args.model))


if __name__ == "__main__":
    main()
