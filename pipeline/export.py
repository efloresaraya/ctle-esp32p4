"""
pipeline/export.py
CTLE binary format v2 writer and reader.

Format layout (little-endian throughout):
─────────────────────────────────────────────────────────────────────
HEADER  (40 bytes)
  magic       u32  = 0x454C5443  ("CTLE")
  version     u32  = 2
  dim         u32
  hidden_dim  u32
  n_layers    u32
  n_heads     u32
  n_kv_heads  u32
  vocab_size  u32
  max_seq_len u32
  flags       u32  bit-0 = weight_tied

TENSOR STREAM  (fixed order, one block per tensor)

  Each block starts with a 1-byte tag:
    tag = 0  →  F32 block   (small tensors kept in full precision)
    tag = 1  →  CTLE block  (large tensors quantized with 4-bit LUT)

  F32 block:
    tag     u8
    count   u32
    data    count × f32  (little-endian)

  CTLE block:
    tag     u8
    rows    u32
    cols    u32
    lut     16 × f32     (64 bytes — the codebook)
    data    ceil(rows*cols/2) bytes  (packed 4-bit nibbles, row-major)
            byte[i] = index[2i] | (index[2i+1] << 4)

Tensor order:
  1.  tok_embeddings            [vocab_size, dim]        CTLE
  2.  For layer l = 0..n_layers-1:
      2a. attention_norm         [dim]                   F32
      2b. wq                     [dim, dim]              CTLE
      2c. wk                     [dim, kv_dim]           CTLE
      2d. wv                     [dim, kv_dim]           CTLE
      2e. wo                     [dim, dim]              CTLE
      2f. ffn_norm               [dim]                   F32
      2g. w1                     [hidden_dim, dim]       CTLE
      2h. w2                     [dim, hidden_dim]       CTLE
      2i. w3                     [hidden_dim, dim]       CTLE
  3.  norm                       [dim]                   F32
─────────────────────────────────────────────────────────────────────
"""

import struct
from pathlib import Path

import numpy as np

MAGIC    = 0x454C5443  # "CTLE"
VERSION  = 2
TAG_F32    = 0
TAG_CTLE   = 1
TAG_INT4U  = 2   # INT4 Uniform  (per-tensor scale)
TAG_INT4BW = 3   # INT4 Block-wise (per-row-group scale, G=32)
TAG_PCTLE  = 4   # Product-LUT CTLE — same data layout as CTLE, tag=4
TAG_CTLE5  = 5   # CTLE with K=32 centroids, 5-bit packed indices


def _pack_nibbles(indices: np.ndarray) -> bytes:
    """Pack [M,N] uint8 indices into ceil(M*N/2) bytes (low nibble first)."""
    flat = indices.reshape(-1).astype(np.uint8)
    if flat.size % 2:
        flat = np.append(flat, 0)
    packed = flat[0::2] | (flat[1::2] << 4)
    return packed.tobytes()


def _pack_5bit(indices: np.ndarray) -> bytes:
    """Pack uint8 indices (0-31) into 5-bit packed bytes.

    Layout: 8 indices are packed into 5 bytes (40 bits).
    Index i occupies bits [5i : 5i+5] within the 40-bit group.
    The array is zero-padded to the next multiple of 8 before packing.
    """
    flat = indices.reshape(-1).astype(np.uint64)
    n = len(flat)
    pad = (-n) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint64)])

    groups = flat.reshape(-1, 8)                          # [G, 8]
    shifts = np.array([0, 5, 10, 15, 20, 25, 30, 35], dtype=np.uint64)
    vals = (groups << shifts).sum(axis=1)                 # [G]  40-bit packed

    byte_shifts = np.array([0, 8, 16, 24, 32], dtype=np.uint64)
    out = ((vals[:, None] >> byte_shifts) & np.uint64(0xFF)).astype(np.uint8)
    return out.reshape(-1).tobytes()


def _write_f32_block(f, data: np.ndarray) -> None:
    flat = data.reshape(-1).astype(np.float32)
    f.write(struct.pack("<B", TAG_F32))
    f.write(struct.pack("<I", flat.size))
    f.write(flat.tobytes())


def _write_ctle_block(f, lut: np.ndarray, indices: np.ndarray) -> None:
    """Write a CTLE block (lut [16] float32, indices [M,N] uint8)."""
    rows, cols = indices.shape
    lut32 = lut.astype(np.float32)

    f.write(struct.pack("<B", TAG_CTLE))
    f.write(struct.pack("<II", rows, cols))
    f.write(lut32.tobytes())              # 64 bytes
    f.write(_pack_nibbles(indices))       # ceil(rows*cols/2) bytes


def _write_ctle5_block(f, lut: np.ndarray, indices: np.ndarray) -> None:
    """Write a CTLE-5b block (K=32 codebook, 5-bit packed indices).

    Format: tag(u8=5) | rows(u32) | cols(u32) | lut[32](f32=128B) | 5bit_packed
    5-bit packing: 8 indices per 5 bytes; n_bytes = ceil(rows*cols/8)*5
    """
    rows, cols = indices.shape
    assert lut.size == 32, f"CTLE-5b requires K=32 codebook, got {lut.size}"
    lut32 = lut.astype(np.float32)
    n_groups = (rows * cols + 7) // 8
    f.write(struct.pack("<B", TAG_CTLE5))
    f.write(struct.pack("<II", rows, cols))
    f.write(lut32.tobytes())          # 128 bytes
    f.write(_pack_5bit(indices))      # n_groups * 5 bytes


def _write_pctle_block(f, lut: np.ndarray, indices: np.ndarray) -> None:
    """Write a P-CTLE block — identical data layout as CTLE but tag=4."""
    rows, cols = indices.shape
    lut32 = lut.astype(np.float32)

    f.write(struct.pack("<B", TAG_PCTLE))
    f.write(struct.pack("<II", rows, cols))
    f.write(lut32.tobytes())              # 64 bytes (weight LUT)
    f.write(_pack_nibbles(indices))       # ceil(rows*cols/2) bytes (weight nibbles)


def _write_int4u_block(f, scale: float, nibbles: np.ndarray) -> None:
    """Write an INT4 Uniform block.
    Format: tag(u8) | rows(u32) | cols(u32) | scale(f32) | packed_nibbles
    """
    rows, cols = nibbles.shape
    f.write(struct.pack("<B", TAG_INT4U))
    f.write(struct.pack("<II", rows, cols))
    f.write(struct.pack("<f", float(scale)))
    f.write(_pack_nibbles(nibbles))


def _write_int4bw_block(
    f,
    scales: np.ndarray,
    nibbles: np.ndarray,
    group_size: int,
) -> None:
    """Write an INT4 Block-wise block.
    Format: tag(u8) | rows(u32) | cols(u32) | group_size(u32)
            | scales_f32[rows * n_groups] | packed_nibbles
    """
    rows, cols = nibbles.shape
    f.write(struct.pack("<B", TAG_INT4BW))
    f.write(struct.pack("<III", rows, cols, group_size))
    f.write(scales.astype(np.float32).tobytes())   # [rows, n_groups] flat
    f.write(_pack_nibbles(nibbles))


def write_ctle_bin(
    output_path: Path,
    config: dict,
    tensors: dict,               # name → (lut, indices) for CTLE or np.ndarray for F32
) -> None:
    """
    Write the CTLE binary file.

    Args:
        output_path:  destination .bin file
        config:       model config dict (dim, hidden_dim, n_layers, n_heads,
                      n_kv_heads, vocab_size, max_seq_len, weight_tied)
        tensors:      mapping from tensor name to:
                        - tuple (lut, indices) for quantized tensors
                        - np.ndarray for F32 tensors
    """
    dim         = config["dim"]
    hidden_dim  = config["hidden_dim"]
    n_layers    = config["n_layers"]
    n_heads     = config["n_heads"]
    n_kv_heads  = config.get("n_kv_heads", n_heads)
    vocab_size  = config["vocab_size"]
    max_seq_len = config.get("max_seq_len", 256)
    weight_tied = int(config.get("weight_tied", True))
    flags       = weight_tied  # bit-0

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<IIIIIIIIII",
            MAGIC, VERSION,
            dim, hidden_dim, n_layers, n_heads, n_kv_heads,
            vocab_size, max_seq_len, flags,
        ))

        def write_tensor(name: str) -> None:
            val = tensors[name]
            if isinstance(val, np.ndarray):
                _write_f32_block(f, val)
            elif isinstance(val, tuple):
                kind = val[0]
                if kind == "int4u":
                    _, scale, nibbles = val
                    _write_int4u_block(f, scale, nibbles)
                elif kind == "int4bw":
                    _, scales, nibbles, group_size = val
                    _write_int4bw_block(f, scales, nibbles, group_size)
                elif kind == "pctle":
                    _, lut, indices = val
                    _write_pctle_block(f, lut, indices)
                elif kind == "ctle5":
                    _, lut, indices = val
                    _write_ctle5_block(f, lut, indices)
                else:
                    # Legacy CTLE tuple: (lut [16], indices [M,N])
                    lut, indices = val
                    _write_ctle_block(f, lut, indices)
            else:
                raise TypeError(f"Unknown tensor type for '{name}': {type(val)}")

        # 1. Token embeddings
        write_tensor("tok_embeddings.weight")

        # 2. Layers
        for l in range(n_layers):
            write_tensor(f"layers.{l}.attention_norm.weight")
            write_tensor(f"layers.{l}.attention.wq.weight")
            write_tensor(f"layers.{l}.attention.wk.weight")
            write_tensor(f"layers.{l}.attention.wv.weight")
            write_tensor(f"layers.{l}.attention.wo.weight")
            write_tensor(f"layers.{l}.ffn_norm.weight")
            write_tensor(f"layers.{l}.feed_forward.w1.weight")
            write_tensor(f"layers.{l}.feed_forward.w2.weight")
            write_tensor(f"layers.{l}.feed_forward.w3.weight")

        # 3. Final norm
        write_tensor("norm.weight")

    size_mb = output_path.stat().st_size / 1e6
    print(f"Written: {output_path}  ({size_mb:.2f} MB)")
