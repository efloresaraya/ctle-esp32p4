"""
pipeline/ctle_reader.py
Read a CTLE binary v2 file and return dequantized float32 weights.
"""

import struct
import numpy as np
from pathlib import Path

MAGIC      = 0x454C5443
VERSION    = 2
TAG_F32    = 0
TAG_CTLE   = 1
TAG_INT4U  = 2
TAG_INT4BW = 3

_INT4_OFFSET = 7  # stored nibble = signed_q + 7


def _unpack_nibbles(packed: np.ndarray, total: int) -> np.ndarray:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    interleaved = np.empty(lo.size + hi.size, dtype=np.uint8)
    interleaved[0::2] = lo
    interleaved[1::2] = hi
    return interleaved[:total]


def read_ctle_bin(path: Path) -> tuple[dict, dict]:
    """
    Read a CTLE binary file.

    Returns:
        config:  dict with model hyperparameters
        weights: dict of name → float32 numpy array (dequantized)

    Weight names:
        "tok_embeddings"        [vocab_size, dim]
        "attn_norm.{l}"        [dim]
        "wq.{l}"               [dim, dim]
        "wk.{l}"               [kv_dim, dim]   (kv_dim = dim * n_kv_heads // n_heads)
        "wv.{l}"               [kv_dim, dim]
        "wo.{l}"               [dim, dim]
        "ffn_norm.{l}"         [dim]
        "w1.{l}"               [hidden_dim, dim]
        "w2.{l}"               [dim, hidden_dim]
        "w3.{l}"               [hidden_dim, dim]
        "norm"                 [dim]
    """
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")

        (dim, hidden_dim, n_layers, n_heads, n_kv_heads,
         vocab_size, max_seq_len, flags) = struct.unpack("<IIIIIIII", f.read(32))

        kv_dim = dim * n_kv_heads // n_heads
        config = {
            "dim": dim, "hidden_dim": hidden_dim,
            "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
            "vocab_size": vocab_size, "max_seq_len": max_seq_len,
            "weight_tied": bool(flags & 1),
            "kv_dim": kv_dim,
        }

        def read_block(name: str, shape: tuple) -> np.ndarray:
            tag = struct.unpack("<B", f.read(1))[0]
            if tag == TAG_F32:
                count = struct.unpack("<I", f.read(4))[0]
                arr = np.frombuffer(f.read(count * 4), dtype=np.float32).copy()
                return arr.reshape(shape)
            elif tag == TAG_CTLE:
                rows, cols = struct.unpack("<II", f.read(8))
                lut = np.frombuffer(f.read(64), dtype=np.float32).copy()
                n_bytes = (rows * cols + 1) // 2
                packed = np.frombuffer(f.read(n_bytes), dtype=np.uint8).copy()
                indices = _unpack_nibbles(packed, rows * cols).reshape(rows, cols)
                return lut[indices].astype(np.float32).reshape(shape)
            elif tag == TAG_INT4U:
                rows, cols = struct.unpack("<II", f.read(8))
                scale = struct.unpack("<f", f.read(4))[0]
                n_bytes = (rows * cols + 1) // 2
                packed = np.frombuffer(f.read(n_bytes), dtype=np.uint8).copy()
                nibbles = _unpack_nibbles(packed, rows * cols).reshape(rows, cols)
                w = (nibbles.astype(np.float32) - _INT4_OFFSET) * scale
                return w.reshape(shape)
            elif tag == TAG_INT4BW:
                rows, cols, group_size = struct.unpack("<III", f.read(12))
                n_groups = (cols + group_size - 1) // group_size
                scales = np.frombuffer(f.read(rows * n_groups * 4), dtype=np.float32).copy()
                scales = scales.reshape(rows, n_groups)
                n_bytes = (rows * cols + 1) // 2
                packed = np.frombuffer(f.read(n_bytes), dtype=np.uint8).copy()
                nibbles = _unpack_nibbles(packed, rows * cols).reshape(rows, cols)
                # Expand scales to [rows, cols] with group broadcasting
                pad = n_groups * group_size - cols
                nib_pad = np.pad(nibbles, ((0, 0), (0, pad))).reshape(rows, n_groups, group_size)
                w_g = (nib_pad.astype(np.float32) - _INT4_OFFSET) * scales[:, :, None]
                w = w_g.reshape(rows, n_groups * group_size)[:, :cols]
                return w.astype(np.float32).reshape(shape)
            else:
                raise ValueError(f"Unknown tag={tag} at tensor '{name}'")

        weights: dict[str, np.ndarray] = {}
        weights["tok_embeddings"] = read_block("tok_embeddings", (vocab_size, dim))

        for l in range(n_layers):
            weights[f"attn_norm.{l}"] = read_block(f"attn_norm.{l}", (dim,))
            weights[f"wq.{l}"]        = read_block(f"wq.{l}",        (dim,    dim))
            weights[f"wk.{l}"]        = read_block(f"wk.{l}",        (kv_dim, dim))
            weights[f"wv.{l}"]        = read_block(f"wv.{l}",        (kv_dim, dim))
            weights[f"wo.{l}"]        = read_block(f"wo.{l}",        (dim,    dim))
            weights[f"ffn_norm.{l}"]  = read_block(f"ffn_norm.{l}", (dim,))
            weights[f"w1.{l}"]        = read_block(f"w1.{l}",        (hidden_dim, dim))
            weights[f"w2.{l}"]        = read_block(f"w2.{l}",        (dim, hidden_dim))
            weights[f"w3.{l}"]        = read_block(f"w3.{l}",        (hidden_dim, dim))

        weights["norm"] = read_block("norm", (dim,))

    return config, weights
