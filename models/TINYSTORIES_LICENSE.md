# TinyStories held-out sample: source and license

`tinystories_val_heldout.txt` contains the first 549 stories of the
`validation` split of the TinyStories dataset (Eldan and Li, 2023), unmodified
except for stripping surrounding whitespace and joining stories with a blank
line. It is the quality-evaluation sample of the round-2 review.

- Source: https://huggingface.co/datasets/roneneldan/TinyStories
- Dataset revision reported by the Hugging Face API on 12 October 2026:
  `f54c09fd2331`
- License: Community Data License Agreement – Sharing, Version 1.0
  (CDLA-Sharing-1.0), https://cdla.dev/sharing-1-0/

Under CDLA-Sharing-1.0 this subset is redistributed under the same license. It
is not covered by this repository's MIT license, which applies to the code.

Regenerate it with `python3 scripts/make_heldout_sample.py
models/tinystories_val_heldout.txt`; the output is byte-identical
(SHA-256 `8057a8f15b9d9b8d32ba90661d669bd85db1a6fd43e343e0f46ddf01d392cbc3`).
