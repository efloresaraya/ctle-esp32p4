#!/usr/bin/env python3
"""
evaluate.py
Compute perplexity (PPL) and mean NLL of a CTLE-compressed model.

Two evaluation modes
--------------------
1. TinyStories-style sentences (default, 10 built-in sentences):
       python -m scripts.evaluate --bins weights/stories15M_pso_k16.bin

2. WikiText-2 test set  (standard benchmark, non-overlapping windows):
       python -m scripts.evaluate --bins weights/*.bin --dataset wikitext2
       python -m scripts.evaluate --bins weights/*.bin --dataset wikitext2 --stride 128

   WikiText-2 PPL is computed as:
       exp(Σ NLL_tokens / Σ N_tokens)
   over non-overlapping (or strided) windows of --max-seq tokens from the
   concatenated test split.  This matches the standard LM eval protocol.

   NOTE: TinyStories-15M was trained on children-story text, NOT Wikipedia.
   Its absolute WikiText-2 PPL will therefore be much higher than numbers
   reported for OPT/LLaMA models (GPTQ, SqueezeLLM, LLM.int8()).
   The value of the WikiText-2 column in this paper is to show the
   *relative* PPL ordering across compression methods on a standard corpus.

Usage:
    python -m scripts.evaluate \\
        --bins weights/stories15M_kmeans_k16.bin \\
               weights/stories15M_ga_k16.bin     \\
               weights/stories15M_pso_k16.bin    \\
        --tokenizer models/tokenizer.model

    python -m scripts.evaluate \\
        --bins weights/stories15M_pso_k16.bin   \\
        --dataset wikitext2 --stride 128
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ctle_reader import read_ctle_bin


# ── FP32 loader from safetensors ──────────────────────────────────────────────

def load_fp32_from_safetensors(safetensors_path: Path) -> tuple[dict, dict]:
    """
    Load TinyStories-15M weights from a HuggingFace safetensors checkpoint.
    Returns (config, weights) in the same format as read_ctle_bin.
    """
    from safetensors.torch import load_file
    raw = load_file(str(safetensors_path))

    # karpathy/tinyllamas safetensors key convention:
    #   tok_embeddings.weight, norm.weight,
    #   layers.{l}.attention.wq/wk/wv/wo.weight
    #   layers.{l}.attention_norm.weight
    #   layers.{l}.feed_forward.w1/w2/w3.weight
    #   layers.{l}.ffn_norm.weight
    tok_emb = raw["tok_embeddings.weight"]       # [vocab, dim]
    vocab, dim = tok_emb.shape
    n_layers = sum(1 for k in raw if k.endswith(".attention.wq.weight"))
    n_heads    = 6    # TinyStories-15M fixed
    n_kv_heads = 6
    hidden = raw["layers.0.feed_forward.w1.weight"].shape[0]

    config = {
        "dim": dim, "n_layers": n_layers, "n_heads": n_heads,
        "n_kv_heads": n_kv_heads, "kv_dim": dim,
        "hidden_dim": hidden, "vocab_size": vocab,
    }

    def np32(t: torch.Tensor) -> np.ndarray:
        return t.float().numpy()

    weights: dict = {}
    weights["tok_embeddings"] = np32(tok_emb)
    weights["norm"]           = np32(raw["norm.weight"])

    for l in range(n_layers):
        prefix = f"layers.{l}"
        weights[f"attn_norm.{l}"] = np32(raw[f"{prefix}.attention_norm.weight"])
        weights[f"ffn_norm.{l}"]  = np32(raw[f"{prefix}.ffn_norm.weight"])
        weights[f"wq.{l}"]        = np32(raw[f"{prefix}.attention.wq.weight"])
        weights[f"wk.{l}"]        = np32(raw[f"{prefix}.attention.wk.weight"])
        weights[f"wv.{l}"]        = np32(raw[f"{prefix}.attention.wv.weight"])
        weights[f"wo.{l}"]        = np32(raw[f"{prefix}.attention.wo.weight"])
        weights[f"w1.{l}"]        = np32(raw[f"{prefix}.feed_forward.w1.weight"])
        weights[f"w2.{l}"]        = np32(raw[f"{prefix}.feed_forward.w2.weight"])
        weights[f"w3.{l}"]        = np32(raw[f"{prefix}.feed_forward.w3.weight"])

    return config, weights

# ── Default TinyStories evaluation corpus ─────────────────────────────────────
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


# ── Transformer forward pass (Llama-2 / karpathy tinyllamas) ─────────────────

def _t(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _precompute_rope(head_dim: int, max_seq: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64 [max_seq, head_dim//2]


def _apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rot(x: torch.Tensor) -> torch.Tensor:
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_ * freqs_cis[: x.shape[0], None, :]).flatten(-2)
        return x_out.type_as(x)
    return rot(xq), rot(xk)


def _quantize_vec(x: torch.Tensor) -> torch.Tensor:
    """Uniform 16-level quantization of a 1-D activation vector (P-CTLE step)."""
    x_min = x.min().item()
    x_max = x.max().item()
    step = (x_max - x_min) / 15.0
    if step < 1e-8:
        return x.clone()
    a_idx = torch.clamp(torch.round((x - x_min) / step), 0, 15).long()
    return (x_min + a_idx.float() * step).to(x.dtype)


def _quantize_act(x: torch.Tensor) -> torch.Tensor:
    """Per-token (per-row) uniform 16-level activation quantization for P-CTLE.
    Matches the firmware pctle_matvec() quantization step exactly.
    """
    if x.dim() == 1:
        return _quantize_vec(x)
    # [seq, dim] → quantize each token independently
    return torch.stack([_quantize_vec(x[i]) for i in range(x.shape[0])], dim=0)


def forward(
    tokens: torch.Tensor,      # [seq_len]  int64
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
    n_rep      = n_heads // kv_heads

    seq = tokens.shape[0]
    w = {k: _t(v) for k, v in weights.items()}

    x = w["tok_embeddings"][tokens]              # [seq, dim]

    for l in range(n_layers):
        h = _rmsnorm(x, w[f"attn_norm.{l}"])

        q = h @ w[f"wq.{l}"].T                  # [seq, dim]
        k = h @ w[f"wk.{l}"].T                  # [seq, kv_dim]
        v = h @ w[f"wv.{l}"].T                  # [seq, kv_dim]

        q = q.view(seq, n_heads,  head_dim)
        k = k.view(seq, kv_heads, head_dim)
        v = v.view(seq, kv_heads, head_dim)

        q, k = _apply_rope(q, k, freqs_cis)

        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        mask   = torch.full((seq, seq), float("-inf")).triu(1)
        scores = scores + mask
        attn   = F.softmax(scores.float(), dim=-1).type_as(q)
        out    = (attn @ v).transpose(0, 1).contiguous().view(seq, dim)

        x = x + out @ w[f"wo.{l}"].T

        h    = _rmsnorm(x, w[f"ffn_norm.{l}"])
        gate = F.silu(h @ w[f"w1.{l}"].T)
        up   = h @ w[f"w3.{l}"].T
        x    = x + (gate * up) @ w[f"w2.{l}"].T

    x      = _rmsnorm(x, w["norm"])
    logits = x @ w["tok_embeddings"].T           # tied weights [seq, vocab]
    return logits


def forward_pctle(
    tokens: torch.Tensor,      # [seq_len]  int64
    weights: dict,
    config: dict,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """P-CTLE forward pass.

    Identical to forward() but every activation vector is quantized to 16
    uniform levels before each weight matrix multiply, simulating the
    firmware pctle_matvec() product-LUT kernel.  Weights are already
    approximately quantized (loaded from a P-CTLE bin via ctle_reader).

    No multiplications occur between quantized activations and quantized
    weights in the hot path — the Python emulation uses FP32 matmul for
    convenience, but the quantisation error profile matches the hardware.
    """
    dim        = config["dim"]
    n_heads    = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    n_layers   = config["n_layers"]
    head_dim   = dim // n_heads
    kv_dim     = config["kv_dim"]
    kv_heads   = n_kv_heads
    n_rep      = n_heads // kv_heads

    seq = tokens.shape[0]
    w = {k: _t(v) for k, v in weights.items()}

    x = w["tok_embeddings"][tokens]              # [seq, dim] — embedding row lookup, no quantisation

    for l in range(n_layers):
        h = _rmsnorm(x, w[f"attn_norm.{l}"])
        hq = _quantize_act(h)                    # ← activation quantisation

        q = hq @ w[f"wq.{l}"].T
        k = hq @ w[f"wk.{l}"].T
        v = hq @ w[f"wv.{l}"].T

        q = q.view(seq, n_heads,  head_dim)
        k = k.view(seq, kv_heads, head_dim)
        v = v.view(seq, kv_heads, head_dim)

        q, k = _apply_rope(q, k, freqs_cis)

        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        mask   = torch.full((seq, seq), float("-inf")).triu(1)
        scores = scores + mask
        attn   = F.softmax(scores.float(), dim=-1).type_as(q)
        out    = (attn @ v).transpose(0, 1).contiguous().view(seq, dim)

        outq = _quantize_act(out)                # ← quantise before wo projection
        x = x + outq @ w[f"wo.{l}"].T

        h     = _rmsnorm(x, w[f"ffn_norm.{l}"])
        hq    = _quantize_act(h)                 # ← quantise before FFN projections
        gate  = F.silu(hq @ w[f"w1.{l}"].T)
        up    = hq @ w[f"w3.{l}"].T
        ffnq  = _quantize_act(gate * up)         # ← quantise before w2
        x     = x + ffnq @ w[f"w2.{l}"].T

    x      = _rmsnorm(x, w["norm"])
    xq     = _quantize_act(x)                   # ← quantise before logit projection
    logits = xq @ w["tok_embeddings"].T          # tied weights
    return logits


# ── PPL computation — sentence corpus ────────────────────────────────────────

def compute_ppl_sentences(
    texts: list[str],
    weights: dict,
    config: dict,
    tokenizer,
    max_seq: int = 256,
) -> tuple[float, float]:
    """Return (mean_nll, ppl) over a list of text sentences."""
    freqs_cis  = _precompute_rope(config["dim"] // config["n_heads"], max_seq)
    fwd        = forward_pctle if config.get("pctle") else forward
    total_nll, total_tok = 0.0, 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, out_type=int)
            if len(ids) < 2:
                continue
            ids    = ids[:max_seq]
            tokens = torch.tensor(ids, dtype=torch.long)
            logits = fwd(tokens, weights, config, freqs_cis)
            lp     = F.log_softmax(logits[:-1].float(), dim=-1)
            tgt    = tokens[1:]
            nll    = -lp[range(len(tgt)), tgt].sum().item()
            total_nll += nll
            total_tok += len(tgt)

    mean_nll = total_nll / max(total_tok, 1)
    return mean_nll, math.exp(mean_nll)


# ── PPL computation — sliding-window over a token sequence ───────────────────

def compute_ppl_tokens(
    all_ids: list[int],
    weights: dict,
    config: dict,
    max_seq: int = 256,
    stride: int | None = None,
    verbose: bool = False,
) -> tuple[float, float, int]:
    """
    Compute PPL with a sliding window over a pre-tokenised sequence.

    stride = max_seq  → non-overlapping windows (fast, standard for seq_len<=256)
    stride < max_seq  → overlapping windows (slower, lower PPL variance)

    Returns (mean_nll, ppl, n_tokens_scored).
    """
    if stride is None:
        stride = max_seq            # default: non-overlapping

    freqs_cis  = _precompute_rope(config["dim"] // config["n_heads"], max_seq)
    fwd        = forward_pctle if config.get("pctle") else forward
    total_nll, total_tok = 0.0, 0
    n_all      = len(all_ids)
    n_windows  = max(1, (n_all - 1 - max_seq) // stride + 1)
    t0         = time.time()

    with torch.no_grad():
        win = 0
        pos = 0
        while pos + 1 < n_all:
            end    = min(pos + max_seq, n_all)
            chunk  = all_ids[pos:end]
            if len(chunk) < 2:
                break
            tokens = torch.tensor(chunk, dtype=torch.long)
            logits = fwd(tokens, weights, config, freqs_cis)

            # When overlapping, only score the NEW tokens (stride tokens from end)
            if stride < max_seq and win > 0:
                score_start = max_seq - stride   # first token index to score
            else:
                score_start = 0

            lp  = F.log_softmax(logits[score_start:-1].float(), dim=-1)
            tgt = tokens[score_start + 1:]
            nll = -lp[range(len(tgt)), tgt].sum().item()
            total_nll += nll
            total_tok += len(tgt)

            if verbose and (win % 50 == 0 or win == n_windows - 1):
                elapsed = time.time() - t0
                ppl_so_far = math.exp(total_nll / max(total_tok, 1))
                print(f"    window {win+1:>4}/{n_windows}  "
                      f"tokens={total_tok:>7}  "
                      f"PPL≈{ppl_so_far:.2f}  "
                      f"({elapsed:.0f}s)",
                      flush=True)

            pos += stride
            win += 1

    mean_nll = total_nll / max(total_tok, 1)
    return mean_nll, math.exp(mean_nll), total_tok


# ── WikiText-2 loader ────────────────────────────────────────────────────────

def load_wikitext2_tokens(tokenizer, split: str = "test") -> list[int]:
    """
    Download WikiText-2 and return a single flat list of token IDs
    using the model's own SentencePiece tokenizer.

    The WikiText-2 test split is concatenated into one stream (standard
    protocol), with paragraph boundaries preserved as whitespace.
    """
    from datasets import load_dataset
    print(f"  Loading WikiText-2 ({split} split) …", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split,
                      trust_remote_code=True)

    all_ids: list[int] = []
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, out_type=int)
        all_ids.extend(ids)

    print(f"  WikiText-2 {split}: {len(all_ids):,} tokens "
          f"(with LLaMA SentencePiece tokenizer)", flush=True)
    return all_ids


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CTLE binary PPL")
    parser.add_argument("--bins",      nargs="*", default=[],
                        help="One or more CTLE .bin files to evaluate")
    parser.add_argument("--fp32",      default=None,
                        help="Path to model.safetensors for FP32 baseline")
    parser.add_argument("--tokenizer", default="models/tokenizer.model",
                        help="SentencePiece .model (default: models/tokenizer.model)")
    parser.add_argument("--text-file", default=None,
                        help="Text file, one document per line")
    parser.add_argument("--dataset",   default=None,
                        choices=["wikitext2"],
                        help="Standard dataset to evaluate on (downloads automatically)")
    parser.add_argument("--split",     default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--max-seq",   type=int, default=256,
                        help="Max sequence length / window size (default: 256)")
    parser.add_argument("--stride",    type=int, default=None,
                        help="Sliding-window stride (default: max-seq = non-overlapping)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print per-window progress")
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        sys.exit("sentencepiece not installed — run: pip install sentencepiece")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    # ── Determine evaluation mode ───────────────────────────────────────────
    if args.dataset == "wikitext2":
        mode      = "wikitext2"
        all_ids   = load_wikitext2_tokens(sp, split=args.split)
        stride    = args.stride or args.max_seq
        print(f"\n  WikiText-2 ({args.split}) — sliding window "
              f"size={args.max_seq}  stride={stride}")
        print(f"  NOTE: TinyStories-15M was trained on children-story text, NOT "
              f"Wikipedia.\n"
              f"        Absolute PPL will be higher than OPT/LLaMA numbers in the "
              f"literature.\n"
              f"        The WikiText-2 column shows the *relative* ordering across "
              f"compression methods.\n")
    elif args.text_file:
        mode  = "sentences"
        texts = [t.strip() for t in Path(args.text_file).read_text().splitlines() if t.strip()]
    else:
        mode  = "sentences"
        texts = _DEFAULT_TEXT

    if mode == "sentences":
        print(f"\n  Evaluating {len(texts)} sentences  (max_seq={args.max_seq})")

    print(f"  {'File':<48} {'NLL':>9} {'PPL':>10}  {'Tokens':>9}  {'Time':>7}")
    print(f"  {'-'*48} {'-'*9} {'-'*10}  {'-'*9}  {'-'*7}")

    if not args.bins and not args.fp32:
        sys.exit("Provide at least one --bins file or --fp32 path.")

    # Build list of (label, loader_fn) to evaluate in order
    eval_targets = []
    if args.fp32:
        p = Path(args.fp32)
        eval_targets.append(("FP32  " + p.name, lambda p=p: load_fp32_from_safetensors(p)))
    for bp in args.bins:
        p = Path(bp)
        eval_targets.append((p.name, lambda p=p: read_ctle_bin(p)))

    results = []
    for label, loader in eval_targets:
        config, weights = loader()
        name = label
        t0 = time.time()

        if args.verbose and mode == "wikitext2":
            print(f"\n  ── {name} ──")

        if mode == "wikitext2":
            nll, ppl, n_tok = compute_ppl_tokens(
                all_ids, weights, config,
                max_seq=args.max_seq,
                stride=stride,
                verbose=args.verbose,
            )
        else:
            nll, ppl = compute_ppl_sentences(texts, weights, config, sp, args.max_seq)
            n_tok = sum(len(sp.encode(t, out_type=int)) - 1
                        for t in texts if len(sp.encode(t, out_type=int)) >= 2)

        elapsed = time.time() - t0
        print(f"  {name:<48} {nll:>9.4f} {ppl:>10.2f}  {n_tok:>9,}  {elapsed:>6.1f}s")
        results.append({"file": name, "nll": nll, "ppl": ppl,
                         "n_tokens": n_tok, "elapsed_s": elapsed})

    print()

    # ── Summary CSV ────────────────────────────────────────────────────────
    if len(results) > 1:
        corpus_tag = "wikitext2" if mode == "wikitext2" else "tinystories"
        print(f"  CSV (corpus={corpus_tag},max_seq={args.max_seq}"
              + (f",stride={stride}" if mode == "wikitext2" else "") + "):")
        print("  file,nll,ppl,n_tokens")
        for r in results:
            print(f"  {r['file']},{r['nll']:.4f},{r['ppl']:.2f},{r['n_tokens']}")
        print()


if __name__ == "__main__":
    main()
