#!/usr/bin/env python3
"""
generate.py
Autoregressive text generation for FP32 and CTLE-compressed models.

Usage:
    # Compare FP32 vs all three compressed variants
    python scripts/generate.py \
        --safetensors models/stories15M/model.safetensors \
        --config      models/stories15M/config.json \
        --bins        weights/stories15M_kmeans_k16.bin \
                      weights/stories15M_ga_k16.bin     \
                      weights/stories15M_pso_k16.bin    \
        --tokenizer   models/tokenizer.model \
        --prompt      "Once upon a time" \
        --max-new     80 \
        --temperature 0.8 \
        --top-p       0.9
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ctle_reader import read_ctle_bin


# ── Weight loaders ────────────────────────────────────────────────────────

def load_safetensors_weights(st_path: Path, config: dict) -> dict:
    """Load FP32 weights from safetensors into the same key format as ctle_reader."""
    n_layers = config["n_layers"]
    raw = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            raw[key] = f.get_tensor(key).float().numpy()

    w = {}
    w["tok_embeddings"] = raw["tok_embeddings.weight"]
    for l in range(n_layers):
        w[f"attn_norm.{l}"] = raw[f"layers.{l}.attention_norm.weight"]
        w[f"wq.{l}"]        = raw[f"layers.{l}.attention.wq.weight"]
        w[f"wk.{l}"]        = raw[f"layers.{l}.attention.wk.weight"]
        w[f"wv.{l}"]        = raw[f"layers.{l}.attention.wv.weight"]
        w[f"wo.{l}"]        = raw[f"layers.{l}.attention.wo.weight"]
        w[f"ffn_norm.{l}"]  = raw[f"layers.{l}.ffn_norm.weight"]
        w[f"w1.{l}"]        = raw[f"layers.{l}.feed_forward.w1.weight"]
        w[f"w2.{l}"]        = raw[f"layers.{l}.feed_forward.w2.weight"]
        w[f"w3.{l}"]        = raw[f"layers.{l}.feed_forward.w3.weight"]
    w["norm"] = raw["norm.weight"]
    return w


# ── Transformer forward (single next-token) ───────────────────────────────

def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _precompute_rope(head_dim: int, max_seq: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope(xq, xk, freqs_cis):
    def rot(x):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_ * freqs_cis[:x.shape[0], None, :]).flatten(-2)
        return out.type_as(x)
    return rot(xq), rot(xk)


def forward(tokens: torch.Tensor, tw: dict, config: dict, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Return logits [seq, vocab]."""
    dim, n_heads, n_kv_heads = config["dim"], config["n_heads"], config["n_kv_heads"]
    head_dim = dim // n_heads
    kv_dim   = config["kv_dim"]
    n_rep    = n_heads // n_kv_heads
    seq      = tokens.shape[0]

    x = tw["tok_embeddings"][tokens]

    for l in range(config["n_layers"]):
        h = _rmsnorm(x, tw[f"attn_norm.{l}"])
        q = (h @ tw[f"wq.{l}"].T).view(seq, n_heads,    head_dim)
        k = (h @ tw[f"wk.{l}"].T).view(seq, n_kv_heads, head_dim)
        v = (h @ tw[f"wv.{l}"].T).view(seq, n_kv_heads, head_dim)
        q, k = _apply_rope(q, k, freqs_cis)
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        q, k, v = q.transpose(0,1), k.transpose(0,1), v.transpose(0,1)
        scores = q @ k.transpose(-2,-1) / math.sqrt(head_dim)
        scores += torch.full((seq, seq), float("-inf")).triu(1)
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        out = (attn @ v).transpose(0,1).contiguous().view(seq, dim)
        x = x + out @ tw[f"wo.{l}"].T
        h = _rmsnorm(x, tw[f"ffn_norm.{l}"])
        x = x + (F.silu(h @ tw[f"w1.{l}"].T) * (h @ tw[f"w3.{l}"].T)) @ tw[f"w2.{l}"].T

    return _rmsnorm(x, tw["norm"]) @ tw["tok_embeddings"].T


# ── Sampling ──────────────────────────────────────────────────────────────

def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature == 0.0:
        return int(logits.argmax())
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Remove tokens after cumulative prob exceeds top_p
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
    sorted_probs /= sorted_probs.sum()
    next_tok = sorted_idx[torch.multinomial(sorted_probs, 1)]
    return int(next_tok)


def generate(
    prompt_ids: list[int],
    weights: dict,
    config: dict,
    freqs_cis: torch.Tensor,
    max_new: int = 80,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> list[int]:
    tw = {k: torch.from_numpy(v) for k, v in weights.items()}
    tokens = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new):
            seq = torch.tensor(tokens[-config["max_seq_len"]:], dtype=torch.long)
            logits = forward(seq, tw, config, freqs_cis)
            next_id = _top_p_sample(logits[-1], temperature, top_p)
            tokens.append(next_id)
            # Stop on EOS (id=1 in LLaMA tokenizer) or period+newline
            if next_id == 1:
                break

    return tokens[len(prompt_ids):]


# ── PPL (same corpus as evaluate.py) ─────────────────────────────────────

_EVAL_TEXTS = [
    "Once upon a time there was a little girl named Lily who lived in a big house with her parents.",
    "Tom and his dog Max went to the park every morning to play fetch with a red ball.",
    "The little bunny was scared because it was lost in the dark forest all alone.",
    "Sara loved to paint pictures of flowers and butterflies in her favorite yellow notebook.",
    "One day the kind old man found a small bird with a broken wing under the oak tree.",
]


def compute_ppl(weights: dict, config: dict, sp, freqs_cis: torch.Tensor, max_seq: int) -> tuple[float, float]:
    tw = {k: torch.from_numpy(v) for k, v in weights.items()}
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for text in _EVAL_TEXTS:
            ids = sp.encode(text, out_type=int)[:max_seq]
            if len(ids) < 2:
                continue
            tokens = torch.tensor(ids, dtype=torch.long)
            logits = forward(tokens, tw, config, freqs_cis)
            log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
            nll = -log_probs[range(len(ids)-1), tokens[1:]].sum().item()
            total_nll += nll
            total_tok += len(ids) - 1
    mean_nll = total_nll / max(total_tok, 1)
    return mean_nll, math.exp(mean_nll)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate text and compare FP32 vs CTLE models")
    parser.add_argument("--safetensors", default=None, help="FP32 safetensors model path")
    parser.add_argument("--config",      default=None, help="config.json (needed with --safetensors)")
    parser.add_argument("--bins",        nargs="*",    default=[], help="CTLE .bin files")
    parser.add_argument("--tokenizer",   default="models/tokenizer.model")
    parser.add_argument("--prompt",      default="Once upon a time", help="Generation prompt")
    parser.add_argument("--max-new",     type=int,   default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p",       type=float, default=0.9)
    parser.add_argument("--max-seq",     type=int,   default=256)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    try:
        import sentencepiece as spm
    except ImportError:
        sys.exit("sentencepiece not installed — run: pip install sentencepiece")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    prompt_ids = sp.encode(args.prompt, out_type=int)

    # Collect all (label, config, weights) to evaluate
    models: list[tuple[str, dict, dict]] = []

    if args.safetensors:
        cfg = json.loads(Path(args.config).read_text())
        cfg["kv_dim"] = cfg["dim"] * cfg.get("n_kv_heads", cfg["n_heads"]) // cfg["n_heads"]
        w = load_safetensors_weights(Path(args.safetensors), cfg)
        models.append(("FP32 (baseline)", cfg, w))

    for bin_path in args.bins:
        cfg, w = read_ctle_bin(Path(bin_path))
        label = Path(bin_path).stem.replace("stories15M_", "").replace("_k16", "").upper()
        models.append((label, cfg, w))

    if not models:
        sys.exit("No models specified — use --safetensors and/or --bins")

    head_dim  = models[0][1]["dim"] // models[0][1]["n_heads"]
    freqs_cis = _precompute_rope(head_dim, args.max_seq)

    # ── PPL table ─────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  PPL comparison  ({len(_EVAL_TEXTS)} sentences)")
    print(f"{'='*62}")
    print(f"  {'Model':<20} {'NLL':>8} {'PPL':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    for label, cfg, w in models:
        nll, ppl = compute_ppl(w, cfg, sp, freqs_cis, args.max_seq)
        print(f"  {label:<20} {nll:>8.4f} {ppl:>8.2f}")

    # ── Generation ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Text generation  (prompt: \"{args.prompt}\")")
    print(f"  temperature={args.temperature}  top_p={args.top_p}  max_new={args.max_new}")
    print(f"{'='*62}\n")

    for label, cfg, w in models:
        torch.manual_seed(args.seed)   # same seed → same sampling decisions
        new_ids = generate(
            prompt_ids, w, cfg, freqs_cis,
            max_new=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        text = sp.decode(new_ids)
        print(f"  [{label}]")
        print(f"  {args.prompt}{text}")
        print()


if __name__ == "__main__":
    main()
