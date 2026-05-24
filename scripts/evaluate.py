#!/usr/bin/env python3
"""
evaluate.py
Compute perplexity (PPL) and mean NLL of a CTLE-compressed model.

Usage:
    # Compare all three methods on the same text
    python scripts/evaluate.py \
        --bins weights/stories15M_kmeans_k16.bin \
               weights/stories15M_ga_k16.bin     \
               weights/stories15M_pso_k16.bin    \
        --tokenizer models/tokenizer.model

    # Use a custom text file (one document per line)
    python scripts/evaluate.py --bins ... --text-file eval_text.txt

The forward pass implements the Llama-2 architecture used by karpathy/tinyllamas:
  RMSNorm → MHA + RoPE → RMSNorm → SwiGLU FFN, all tied-output projection.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ctle_reader import read_ctle_bin

# ── Default evaluation corpus (TinyStories-style sentences) ───────────────
_DEFAULT_TEXT = [
    "Once upon a time there was a little girl named Lily who lived in a big house with her parents.",
    "Tom and his dog Max went to the park every morning to play fetch with a red ball.",
    "The little bunny was scared because it was lost in the dark forest all alone.",
    "Sara loved to paint pictures of flowers and butterflies in her favorite yellow notebook.",
    "One day the kind old man found a small bird with a broken wing under the oak tree.",
    "The children laughed and ran through the meadow chasing colorful butterflies all afternoon.",
    "Ben wanted to learn how to swim so his mother took him to the pool every Saturday.",
    "The dragon was not scary at all — it was friendly and liked to bake cookies for the village.",
    "Every night before bed, Mia asked her father to read her one more story about brave knights.",
    "The little robot did not know how to dance but it tried its best and everyone clapped.",
]


# ── Transformer forward pass (Llama-2 / karpathy tinyllamas) ──────────────

def _t(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _precompute_rope(head_dim: int, max_seq: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 [max_seq, head_dim//2]


def _apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xq, xk: [seq, n_heads, head_dim]
    def rot(x: torch.Tensor) -> torch.Tensor:
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_ * freqs_cis[:x.shape[0], None, :]).flatten(-2)
        return x_out.type_as(x)
    return rot(xq), rot(xk)


def forward(
    tokens: torch.Tensor,     # [seq_len]  int64
    weights: dict,
    config: dict,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Return logits [seq_len, vocab_size]."""
    dim        = config["dim"]
    n_heads    = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    n_layers   = config["n_layers"]
    head_dim   = dim // n_heads
    kv_dim     = config["kv_dim"]
    kv_heads   = n_kv_heads
    n_rep      = n_heads // kv_heads        # GQA repeat factor

    seq = tokens.shape[0]
    w = {k: _t(v) for k, v in weights.items()}

    # Embedding
    x = w["tok_embeddings"][tokens]         # [seq, dim]

    for l in range(n_layers):
        # ── Attention ──────────────────────────────────────────────────
        h = _rmsnorm(x, w[f"attn_norm.{l}"])   # [seq, dim]

        q = h @ w[f"wq.{l}"].T                  # [seq, dim]
        k = h @ w[f"wk.{l}"].T                  # [seq, kv_dim]
        v = h @ w[f"wv.{l}"].T                  # [seq, kv_dim]

        q = q.view(seq, n_heads,  head_dim)
        k = k.view(seq, kv_heads, head_dim)
        v = v.view(seq, kv_heads, head_dim)

        q, k = _apply_rope(q, k, freqs_cis)

        # GQA: repeat k/v if needed
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Causal scaled dot-product attention
        q = q.transpose(0, 1)                    # [n_heads, seq, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.full((seq, seq), float("-inf")).triu(1)
        scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        out = (attn @ v).transpose(0, 1).contiguous().view(seq, dim)

        x = x + out @ w[f"wo.{l}"].T            # [seq, dim]

        # ── FFN (SwiGLU) ───────────────────────────────────────────────
        h = _rmsnorm(x, w[f"ffn_norm.{l}"])
        gate = F.silu(h @ w[f"w1.{l}"].T)
        up   = h @ w[f"w3.{l}"].T
        x    = x + (gate * up) @ w[f"w2.{l}"].T

    x = _rmsnorm(x, w["norm"])
    logits = x @ w["tok_embeddings"].T          # tied weights [seq, vocab]
    return logits


# ── PPL computation ────────────────────────────────────────────────────────

def compute_ppl(
    texts: list[str],
    weights: dict,
    config: dict,
    tokenizer,
    max_seq: int = 256,
) -> tuple[float, float]:
    """Return (mean_nll, ppl) over all texts."""
    freqs_cis = _precompute_rope(config["dim"] // config["n_heads"], max_seq)
    total_nll = 0.0
    total_tok = 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, out_type=int)
            if len(ids) < 2:
                continue
            ids = ids[:max_seq]
            tokens = torch.tensor(ids, dtype=torch.long)
            logits = forward(tokens, weights, config, freqs_cis)  # [T, vocab]
            # Predict token[1..T] from token[0..T-1]
            log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
            targets   = tokens[1:]
            nll = -log_probs[range(len(targets)), targets].sum().item()
            total_nll += nll
            total_tok += len(targets)

    mean_nll = total_nll / max(total_tok, 1)
    ppl      = math.exp(mean_nll)
    return mean_nll, ppl


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CTLE binary PPL")
    parser.add_argument("--bins",       nargs="+", required=True,
                        help="One or more CTLE .bin files to evaluate")
    parser.add_argument("--tokenizer",  default="models/tokenizer.model",
                        help="SentencePiece .model file (default: models/tokenizer.model)")
    parser.add_argument("--text-file",  default=None,
                        help="Text file with one document per line (default: built-in corpus)")
    parser.add_argument("--max-seq",    type=int, default=256,
                        help="Max sequence length (default: 256)")
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        sys.exit("sentencepiece not installed — run: pip install sentencepiece")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    if args.text_file:
        texts = Path(args.text_file).read_text().splitlines()
        texts = [t.strip() for t in texts if t.strip()]
    else:
        texts = _DEFAULT_TEXT

    print(f"\n  Evaluating {len(texts)} sentences  (max_seq={args.max_seq})")
    print(f"  {'File':<45} {'NLL':>8} {'PPL':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8}")

    for bin_path in args.bins:
        config, weights = read_ctle_bin(Path(bin_path))
        nll, ppl = compute_ppl(texts, weights, config, sp, args.max_seq)
        name = Path(bin_path).name
        print(f"  {name:<45} {nll:>8.4f} {ppl:>8.2f}")

    print()


if __name__ == "__main__":
    main()
