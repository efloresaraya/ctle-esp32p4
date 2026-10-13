"""Per-tensor RelMSE of a compressed .bin against the FP32 checkpoint.

RelMSE = mean((W - W_hat)^2) / mean(W^2), as in pipeline/quantize.py. Reads
the published weight files directly, so Table II can be checked without
re-running the stochastic optimizers.

Usage: python3 -m scripts.relmse_from_bins weights/stories15M_pso_k16.bin [...]
"""
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ctle_reader import read_ctle_bin

MODEL = Path(__file__).resolve().parents[1] / "models/stories15M/model.safetensors"
SHOWN = ["tok_embeddings", "wq.0", "wk.0", "wo.0", "w1.0", "w2.0", "w3.0",
         "wq.5", "w1.5", "w2.5"]

def names():
    m = {"tok_embeddings": "tok_embeddings.weight"}
    for l in range(6):
        for a in ("wq", "wk", "wv", "wo"):
            m[f"{a}.{l}"] = f"layers.{l}.attention.{a}.weight"
        for a in ("w1", "w2", "w3"):
            m[f"{a}.{l}"] = f"layers.{l}.feed_forward.{a}.weight"
    return m

def main(paths):
    f = safe_open(str(MODEL), "np")
    m = names()
    print("file," + ",".join(SHOWN) + ",mean_all,n_tensors")
    for p in paths:
        _, w = read_ctle_bin(Path(p))
        r = {}
        for k, key in m.items():
            W = f.get_tensor(key)
            q = w[k].reshape(W.shape)
            r[k] = float(np.mean((W - q) ** 2) / (np.mean(W ** 2) + 1e-12))
        print(Path(p).name + "," + ",".join(f"{r[k]:.5f}" for k in SHOWN)
              + f",{np.mean(list(r.values())):.5f},{len(r)}")

if __name__ == "__main__":
    main(sys.argv[1:])
