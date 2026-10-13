# Regeneration of the compressed weights (13 October 2026)

Provenance for Table II (per-tensor RelMSE) and the "Comp. s" column of
Table III. Until now those figures existed only in the manuscript and in
`tables/`; no raw record had been kept.

## Final run: `final/`

`compress.py` with seed 42 for every format in Table III, the same recipe as
the published files in `weights/`, on an Apple M4 Pro (Mac mini, no GPU), one
method at a time on an otherwise idle machine, with the exporter fix of commit
`9eb4df2`. `final/environment.txt` records the library versions, the commit
and the hash of the FP32 checkpoint. The five codebook methods ran as one
sequence (`final/console.txt`); the three INT4 formats right after it. Each
method ran once, and every run completed and wrote its file.

| Format | Wall time | Regenerated file |
|---|---:|---|
| INT4U | 0.6 s | identical to `weights/` |
| INT4BW | 0.5 s | identical |
| INT4BW, G = 288 | 0.5 s | identical |
| CTLE-KM | 85.1 s | identical |
| CTLE-GA | 208.5 s | identical |
| CTLE-PSO | 198.2 s | differs |
| P-CTLE32 | 208.0 s | differs |
| CTLE-5b | 414.6 s | differs |

`final/published_sha256.txt` and `final/regenerated_sha256.txt` hold the
hashes. The three formats that differ are the ones optimized with PSO; their
regenerated files are close but not identical to the published ones. All
results in the manuscript, including Table II, use the published files.

## Table II against the published weights

`relmse_published_weights.csv` is computed directly from `weights/*.bin`
against the FP32 checkpoint with `scripts/relmse_from_bins.py`. Every cell of
the manuscript's Table II (tok_emb, wq.0, wk.0, w1.0, mean; K-means, GA, PSO)
agrees. Two cells that appear only in `tables/table_relmse.tex` (GA w2.0, PSO
wq.5) differ from it by one in the last digit: the values are 0.0144346 and
0.0136947.

## First attempt: the logs in this folder

`compress_<format>.log` in this folder come from a first attempt, before the
fix. The K-means, GA and PSO runs stopped with a `ValueError` while writing
the file: `pipeline/export.py` compared the first element of each tensor tuple
with a format name, and for plain CTLE tensors that element is the codebook
array. The error had been there since P-CTLE was added in May; the published
files predate it. The `_fixcheck` logs confirm that the fix writes the K-means
file (byte-identical to the published one) and the PSO file. These runs are
kept for the record; their times are not used.

The manuscript's earlier compression times (73.6, 207.7, 198.3, 181.4 and
374 s) came from an unrecorded run and are replaced by the final run above.
