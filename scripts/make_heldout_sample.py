"""Rebuild the held-out TinyStories sample used for the round-2 review.

Takes stories from the start of the `validation` split of
roneneldan/TinyStories, in order, skipping empty ones, until their token count
under the model's tokenizer reaches 120,000; joins them with a blank line.
This yields the first 549 validation stories. The split is held out from the
data TinyStories-15M was trained on.

    python3 scripts/make_heldout_sample.py models/tinystories_val_heldout.txt

The dataset is distributed under CDLA-Sharing-1.0; see
models/TINYSTORIES_LICENSE.md.
"""
import sys
import sentencepiece as spm
from datasets import load_dataset

TARGET_TOKENS = 120_000

def main(out_path, tokenizer="models/tokenizer.model"):
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    stories, ntok = [], 0
    for row in load_dataset("roneneldan/TinyStories", split="validation"):
        text = row["text"].strip()
        if not text:
            continue
        stories.append(text)
        ntok += len(sp.encode(text))
        if ntok >= TARGET_TOKENS:
            break
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(stories))
    print(f"{out_path}: {len(stories)} stories, {ntok:,} tokens")

if __name__ == "__main__":
    main(sys.argv[1])
