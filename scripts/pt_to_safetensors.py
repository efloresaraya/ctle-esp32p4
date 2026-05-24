#!/usr/bin/env python3
"""
pt_to_safetensors.py
Convert a karpathy/tinyllamas .pt checkpoint to safetensors + config.json.

Usage:
    python scripts/pt_to_safetensors.py \
        --input  models/tinyllamas/stories15M.pt \
        --output models/stories15M
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    model_args: dict = ckpt["model_args"]
    state_dict: dict = ckpt["model"]

    # Infer hidden_dim from w1 shape if not present
    if "hidden_dim" not in model_args:
        for k, v in state_dict.items():
            if "feed_forward.w1" in k:
                model_args["hidden_dim"] = v.shape[0]
                break

    # Check weight tying
    weight_tied = torch.equal(
        state_dict["tok_embeddings.weight"],
        state_dict["output.weight"],
    )
    model_args["weight_tied"] = weight_tied

    # Drop output.weight if tied (avoid storing twice)
    if weight_tied:
        state_dict = {k: v for k, v in state_dict.items() if k != "output.weight"}
        print("Weight tying detected — output.weight dropped (same as tok_embeddings).")

    # Ensure all tensors are float32 contiguous
    tensors = {k: v.float().contiguous() for k, v in state_dict.items()}

    # Save safetensors
    st_path = output_dir / "model.safetensors"
    save_file(tensors, st_path)
    print(f"Saved {st_path}  ({st_path.stat().st_size / 1e6:.1f} MB)")

    # Save config
    cfg_path = output_dir / "config.json"
    cfg_path.write_text(json.dumps(model_args, indent=2))
    print(f"Saved {cfg_path}")

    # Summary
    print(f"\nModel config: {model_args}")
    print(f"Tensors: {len(tensors)}")
    total_params = sum(v.numel() for v in tensors.values())
    print(f"Total params: {total_params / 1e6:.2f}M")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .pt checkpoint to safetensors")
    parser.add_argument("--input",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
