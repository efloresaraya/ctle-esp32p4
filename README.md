# CTLE — Compressed Tensor-LUT Engine for IoT TinyLM Inference

**Paper:** *CTLE: A Compressed Tensor-LUT Engine for On-Device TinyLM Inference on IoT Microcontrollers with Metaheuristic Codebook Optimization*  
**Author:** Emilio Flores — Universidad de Valparaíso, Chile · ANID Doctoral Fellowship 21242003  
**Target board:** ESP32-P4 Nano (RISC-V HP @ 360 MHz, 32 MB PSRAM, 16 MB Flash)  
**Model:** TinyStories-15M (Llama-2 architecture, 6 layers, 288 hidden, 32 K vocab)

---

## Repository layout

```
ctle_p4/
├── pipeline/          # Python compression pipeline (import as package)
│   ├── quantize.py    # CTLE core: k-means, _pack_nibbles, RelMSE
│   ├── ga.py          # Genetic Algorithm codebook refinement (BLX-α)
│   ├── pso.py         # Particle Swarm Optimization refinement (ω decay)
│   ├── int4.py        # INT4 Uniform & Block-wise baselines
│   ├── export.py      # Write binary v2 (TAG_F32/CTLE/INT4U/INT4BW)
│   └── ctle_reader.py # Python reader for verification
├── scripts/
│   ├── compress.py    # Main compression entry-point
│   ├── evaluate.py    # PPL / NLL evaluation on a sentence corpus
│   └── generate.py    # CPU-side autoregressive text generation
├── firmware/          # ESP-IDF project (ESP32-P4)
│   ├── main/main.c    # Inference engine (CTLE + INT4 multi-format)
│   ├── partitions.csv # SPIFFS at 0x310000, 13.3 MB
│   ├── sdkconfig.defaults
│   └── build.sh       # Convenience build + flash helper
├── tables/            # LaTeX tables (included by paper)
│   ├── table_ablation.tex  # Table I  — method comparison
│   ├── table_relmse.tex    # Table III — per-tensor RelMSE
│   ├── table_sota.tex      # Table II  — comparison with SOTA
│   └── table_hw.tex        # Table IV  — hardware benchmark
├── paper/             # IEEE journal paper (IEEEtran)
│   ├── ctle_p4.tex    # Main .tex file
│   ├── references.bib
│   └── IEEEtran.cls
├── results/
│   └── hw_benchmark.txt   # Raw hardware benchmark log
├── models/            # (gitignored) place downloaded model here
│   └── .gitkeep
├── weights/           # (gitignored) generated .bin files go here
│   └── .gitkeep
└── requirements.txt
```

---

## Quick-start

### 1 — Install Python dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Download the base model

```bash
python3 scripts/pt_to_safetensors.py   # downloads stories15M from HF Hub
# Produces: models/stories15M/{model.safetensors, tokenizer.model}
```

### 3 — Compress

```bash
# CTLE-PSO (best PPL — used in paper)
python3 -m scripts.compress --model models/stories15M/model.safetensors \
    --method pso --k 16 --output weights/stories15M_pso_k16.bin

# CTLE-GA
python3 -m scripts.compress --method ga  --k 16 \
    --output weights/stories15M_ga_k16.bin

# CTLE-K-means
python3 -m scripts.compress --method kmeans --k 16 \
    --output weights/stories15M_kmeans_k16.bin

# INT4 Block-wise baseline
python3 -m scripts.compress --method int4_blockwise \
    --output weights/stories15M_int4bw.bin

# INT4 Uniform baseline
python3 -m scripts.compress --method int4_uniform \
    --output weights/stories15M_int4u.bin
```

### 4 — Evaluate (PPL / NLL)

```bash
python3 -m scripts.evaluate --bin weights/stories15M_pso_k16.bin \
    --tokenizer models/stories15M/tokenizer.model
```

### 5 — CPU text generation (verify decoding)

```bash
python3 -m scripts.generate --bin weights/stories15M_pso_k16.bin \
    --tokenizer models/stories15M/tokenizer.model \
    --prompt "Once upon a time"
```

---

## ESP32-P4 firmware

### Prerequisites

- ESP-IDF v6.1 (or v5.x with P4 support): `source ~/enki_ESP32/esp-idf/export.sh`
- ESP32-P4 Nano board connected via USB

### Build & flash

```bash
cd firmware

# Full build + flash app + flash SPIFFS model
idf.py build
idf.py -p /dev/cu.usbmodem* flash

# Flash SPIFFS separately (after generating model)
python3 ~/enki_ESP32/esp-idf/components/spiffs/spiffsgen.py \
    13565952 ../spiffs_data spiffs.bin
esptool.py -p /dev/cu.usbmodem* write_flash 0x310000 spiffs.bin
```

Partition layout (see `partitions.csv`):

| Name    | Type | Offset     | Size       |
|---------|------|------------|------------|
| factory | app  | 0x010000   | 3 MB       |
| storage | data/spiffs | 0x310000 | 13.3 MB |

### Hardware results (CTLE-PSO, k=16)

| Prompt             | Tokens | Prefill (ms) | ms/tok | tok/s |
|--------------------|--------|-------------|--------|-------|
| "Once upon a time" | 4      | 4,130       | 1,035  | 0.97  |
| "Tom and his dog…" | 8      | 8,188       | 1,036  | 0.97  |
| "The little girl…" | 6      | 6,091       | 1,037  | 0.96  |
| **Average**        | —      | —           | **1,036** | **0.965** |

CPU: RISC-V HP @ 360 MHz (silicon rev v1.3 hardware cap).  
Memory: 7.61 MB model loaded from Flash → PSRAM in 2,694 ms. Remaining PSRAM: 21,566 KB.

---

## Compression results summary

| Method       | Bits/w | Size (MB) | Ratio   | PPL ↓   | Time (s) |
|--------------|--------|-----------|---------|---------|----------|
| FP32         | 32     | 60.77     | 1.00×   | 6.92    | —        |
| INT4 Uniform | 4      | 7.61      | 7.99×   | 11070   | <1       |
| INT4 Block-wise | ~5  | 9.51      | 6.39×   | 10.71   | <1       |
| CTLE K-means | 4      | 7.61      | 7.98×   | 19.23   | 73.6     |
| CTLE GA      | 4      | 7.61      | 7.98×   | 18.98   | 207.7    |
| **CTLE PSO** | **4**  | **7.61**  | **7.98×** | **15.93** | 198.3 |

PPL evaluated on 10 TinyStories-style sentences (see `scripts/evaluate.py`).  
PSO achieves −14.9 % RelMSE on the 9.2 M-entry embedding matrix → −17.2 % PPL vs. K-means.

---

## Binary format v2

Magic: `0x454C5443` ("CTLE"). 40-byte header + tensor stream.

| Tag | Name       | Payload                                              |
|-----|------------|------------------------------------------------------|
| 0   | TAG_F32    | rows · cols · 4 bytes                               |
| 1   | TAG_CTLE   | LUT[16]×4B + ceil(rows·cols/2) nibble bytes          |
| 2   | TAG_INT4U  | scale×4B + ceil(rows·cols/2) nibble bytes            |
| 3   | TAG_INT4BW | group_size×4B + scales[rows·nG]×4B + nibbles         |

Nibble encoding: `byte = lo | (hi << 4)`, row-major. CTLE index ∈ [0,15]; INT4 value = nibble − 7 ∈ [−7, 7].

---

## Paper

The paper source is in `paper/ctle_p4.tex` (IEEE journal, IEEEtran format).  
Tables are in `tables/` and included with `\input{../tables/table_name}`.

To compile:
```bash
cd paper
pdflatex ctle_p4.tex
bibtex ctle_p4
pdflatex ctle_p4.tex && pdflatex ctle_p4.tex
```

---

## Citation

```bibtex
@article{flores2026ctle,
  author  = {Flores, Emilio},
  title   = {{CTLE}: A Compressed Tensor-{LUT} Engine for On-Device {TinyLM}
             Inference on {IoT} Microcontrollers with Metaheuristic Codebook Optimization},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## Funding

Funded by the National Agency for Research and Development (ANID), Chile,  
Doctoral Fellowship No. 21242003.
