# Regeneration of the codebook weights (13 October 2026)

Provenance for Table II (per-tensor RelMSE) and the "Comp. s" column of
Table III. Until now those figures existed only in the manuscript and in
`tables/`; no raw record had been kept.

## What was run

`compress.py` with seed 42 for K-means, GA, PSO, P-CTLE and CTLE-5b, the same
recipe as the published files in `weights/`, on an Apple M4 Pro (Mac mini),
one method at a time on an otherwise idle machine. `environment.txt` records
the library versions, the commit and the hash of the FP32 checkpoint.

`compress_<format>.log` is the full output of each run: per-tensor RelMSE,
footprint and wall time. The K-means, GA and PSO runs stopped with a
`ValueError` while writing the file. `pipeline/export.py` compared the first
element of each tensor tuple with a format name, and for plain CTLE tensors
that element is the codebook array. The error had been there since P-CTLE was
added in May; the published files predate it. The values printed before
writing are complete. With the fix, `compress_kmeans_k16_fixcheck.log` and
`compress_pso_k16_fixcheck.log` write their files normally.

## What it shows

- **Table II matches the published weights.** `relmse_published_weights.csv`
  is computed directly from `weights/*.bin` against the FP32 checkpoint with
  `scripts/relmse_from_bins.py`. Every cell of the manuscript's Table II
  (tok_emb, wq.0, wk.0, w1.0, mean; K-means, GA, PSO) agrees. Two cells that
  appear only in `tables/table_relmse.tex` (GA w2.0, PSO wq.5) differ from it
  by one in the last digit: 0.0144346 and 0.0136947, rounded.
- **K-means regenerates byte for byte.** The fixed run produces a file
  identical to `weights/stories15M_kmeans_k16.bin`.
- **GA and PSO regenerate closely but not exactly.** Both are stochastic, and
  the library versions have changed since May; RelMSE differs by at most one
  in the fifth decimal for the tensors in Table II, and the regenerated
  P-CTLE and CTLE-5b files are not byte-identical to the published ones. A
  regenerated PSO file scores TinyStories perplexity 7.67, against 7.79 for
  the published one. All manuscript results use the published files.
- **Compression times, this machine:** K-means 81.2 s, GA 193.2 s, PSO
  196.2 s (192.9 s on the second run), P-CTLE 198.0 s, CTLE-5b 404.7 s.
  The manuscript's 73.6, 207.7, 198.3, 181.4 and 374 s come from an earlier,
  unrecorded run; wall time depends on the machine and its load.
