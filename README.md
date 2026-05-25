# CTLE-ESP32P4

**CTLE (Compressed Tensor–LUT Engine)** — Streaming learned-codebook inference for transformer models on ESP32-P4 microcontrollers.

Companion code for:
> E. Flores, "CTLE: Compressed-Domain TinyLM Inference on an ESP32-P4 Microcontroller," *IEEE Embedded Systems Letters*, 2026 (under review).

**Target board:** ESP32-P4 Nano · RISC-V HP @ 360 MHz · 32 MB PSRAM · 16 MB Flash  
**Model:** TinyStories-15M (Llama-2, 6 layers, d=288, 32 K vocab)

---

## Key results

| Method | Bits | MB | PPL TS | PPL WT2 | ms/tok | tok/s |
|---|---|---|---|---|---|---|
| FP32 (reference) | 32 | 60.77 | 9.35 | 6,127 | — | — |
| INT4U | 4 | 7.61 | 11,070 | 533,069 | 1,040.6 | 0.961 |
| INT4BW | ~5 | 9.51 | 10.71 | 7,120 | 1,355.8 | 0.737 |
| CTLE-PSO | 4 | 7.61 | 15.93 | 16,827 | 1,035.9 | 0.965 |
| P-CTLE | 4 | 7.61 | 33.14 | 29,049 | 832.5 | 1.201 |
| P-CTLE32 | W4/A5 | 7.61 | 20.30 | 19,095 | 832.4 | 1.201 |
| **CTLE-5b** | **5** | **9.51** | **10.00** | **8,226** | **1,014.2** | **0.986** |

All hardware numbers measured on ESP32-P4 Nano (3 prompts × 64 tokens, seed 0xCAFEBABE).  
WT2 = WikiText-2 test split, 256-token non-overlapping windows, 325,693 tokens scored.

---

## Repository layout

```
ctle-esp32p4/
├── pipeline/
│   ├── quantize.py      # K-means core + RelMSE, _pack_nibbles, _pack_5bit
│   ├── ctle_reader.py   # Binary format reader (all 6 tags)
│   ├── export.py        # Write CTLE binary from codebook + indices
│   ├── ga.py            # Genetic Algorithm (BLX-α, tournament, elitism)
│   ├── pso.py           # Particle Swarm Optimization (ω-decay, K-means init)
│   ├── int4.py          # INT4 uniform & block-wise baselines
│   └── de.py            # Differential Evolution (experimental)
├── scripts/
│   ├── compress.py      # Main entry-point — generates .bin from safetensors
│   ├── evaluate.py      # PPL/NLL on TinyStories probe and WikiText-2
│   ├── generate.py      # CPU-side autoregressive text generation
│   └── pt_to_safetensors.py  # Convert llama2.c .pt → safetensors
├── firmware/
│   ├── main/main.c      # Full ESP32-P4 inference engine (all 6 formats)
│   ├── main/CMakeLists.txt
│   ├── CMakeLists.txt
│   ├── partitions.csv   # SPIFFS at 0x310000, 13.3 MB
│   ├── sdkconfig.defaults
│   └── build.sh         # Convenience build helper
├── weights/             # Pre-compressed .bin files (tracked, 7–9 MB each)
│   ├── stories15M_pso_k16.bin      # CTLE-PSO  4-bit K=16
│   ├── stories15M_ctle5_k32.bin    # CTLE-5b   5-bit K=32
│   ├── stories15M_pctle_k16.bin    # P-CTLE    W4/A4
│   ├── stories15M_int4bw.bin       # INT4BW baseline
│   ├── stories15M_int4u.bin        # INT4U baseline
│   ├── stories15M_kmeans_k16.bin   # CTLE-KM
│   └── stories15M_ga_k16.bin       # CTLE-GA
├── models/
│   └── tokenizer.model  # LLaMA SentencePiece tokenizer (32 K vocab)
├── results/
│   ├── hw_benchmark.txt # Raw CSV from ESP32-P4 serial log
│   └── wikitext2_ppl.txt
├── tables/              # LaTeX tables (included by paper)
│   ├── table_ablation.tex
│   ├── table_hw.tex
│   ├── table_relmse.tex
│   └── table_sota.tex
├── paper/
│   ├── ctle_esl.tex     # IEEE ESL manuscript (4 pages)
│   └── ctle_esl.pdf     # Compiled PDF
└── requirements.txt
```

---

## Quickstart

### 1. Install Python dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the base model

```bash
# Option A — from HuggingFace Hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('karpathy/tinyllamas', 'stories15M.pt',
                local_dir='models/tinyllamas')
"
python3 scripts/pt_to_safetensors.py \
    models/tinyllamas/stories15M.pt \
    models/stories15M/model.safetensors

# Option B — direct download
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt \
     -O models/tinyllamas/stories15M.pt
```

### 3. Compress

Pre-compressed binaries are already in `weights/`. To regenerate:

```bash
# CTLE-PSO (4-bit, K=16) — 7.61 MB
python3 -m scripts.compress \
    --model models/stories15M/model.safetensors \
    --method pso --k 16 --seed 42 \
    --output weights/stories15M_pso_k16.bin

# CTLE-5b (5-bit, K=32) — 9.51 MB  ← best quality/throughput point
python3 -m scripts.compress \
    --model models/stories15M/model.safetensors \
    --method ctle5 --k 32 --seed 42 \
    --output weights/stories15M_ctle5_k32.bin

# P-CTLE32 (W4 weights, 32-level act quant)
python3 -m scripts.compress \
    --model models/stories15M/model.safetensors \
    --method pctle --k 16 --seed 42 \
    --output weights/stories15M_pctle_k16.bin
```

### 4. Evaluate PPL

```bash
# TinyStories probe (fast)
python3 -m scripts.evaluate \
    --bin weights/stories15M_ctle5_k32.bin \
    --tokenizer models/tokenizer.model

# WikiText-2 test split (requires HuggingFace datasets)
python3 -m scripts.evaluate \
    --bin weights/stories15M_ctle5_k32.bin \
    --tokenizer models/tokenizer.model \
    --wikitext2

# P-CTLE with 32 activation levels
python3 -m scripts.evaluate \
    --bin weights/stories15M_pctle_k16.bin \
    --tokenizer models/tokenizer.model \
    --act-levels 32
```

### 5. CPU text generation (verify decoding)

```bash
python3 -m scripts.generate \
    --bin weights/stories15M_pso_k16.bin \
    --tokenizer models/tokenizer.model \
    --prompt "Once upon a time"
```

---

## ESP32-P4 firmware

### Prerequisites

- ESP-IDF v6.1 with ESP32-P4 support: `source ~/esp/esp-idf/export.sh`
- ESP32-P4 Nano connected via USB-C

### Build and flash

```bash
cd firmware
idf.py build

# Flash firmware
python3 -m esptool --chip esp32p4 --baud 921600 \
    write_flash 0x10000 build/firmware.bin

# Generate SPIFFS image with compressed model, then flash
python3 $IDF_PATH/components/spiffs/spiffsgen.py \
    13565952 <model_dir> spiffs.bin
python3 -m esptool --chip esp32p4 --baud 921600 \
    write_flash 0x310000 spiffs.bin
```

Partition layout (`partitions.csv`):

| Name    | Type        | Offset   | Size   |
|---------|-------------|----------|--------|
| factory | app         | 0x010000 | 3 MB   |
| storage | data/spiffs | 0x310000 | 13.3 MB|

The firmware autodetects the binary tag at load time and selects the corresponding matvec kernel (no recompilation needed to switch between formats).

---

## Binary format

Magic: `0x454C5443` ("CTLE"). 40-byte header + tensor stream.

| Tag | Name      | Index bits | K  | Payload |
|-----|-----------|------------|-----|---------|
| 0   | TAG_F32   | —          | —  | rows·cols·4 B |
| 1   | TAG_CTLE  | 4 (nibble) | 16 | LUT[16]·4 B + ⌈rows·cols/2⌉ B |
| 2   | TAG_INT4U | 4 (uniform)| —  | scale·4 B + ⌈rows·cols/2⌉ B |
| 3   | TAG_INT4BW| 4 (blockwise)| — | group_size + scales + nibbles |
| 4   | TAG_PCTLE | 4 (nibble) | 16 | same as TAG_CTLE; act quant at runtime |
| 5   | TAG_CTLE5 | 5 (packed) | 32 | LUT[32]·4 B + ⌈rows·cols/8⌉·5 B |

CTLE-5b packing: 8 indices per 5-byte group. Index `i` occupies bits `[5i : 5i+5]` in a 40-bit little-endian word. Decoded by `decode5_group()` via uint64 shifts (no lookup table, no PSRAM scale reads).

---

## Citation

```bibtex
@article{flores2026ctle,
  author  = {Flores, Emilio},
  title   = {{CTLE}: Compressed-Domain {TinyLM} Inference on an {ESP32-P4} Microcontroller},
  journal = {IEEE Embedded Systems Letters},
  year    = {2026},
  note    = {Under review}
}
```

---

## Funding

Supported by ANID BECAS/DOCTORADO NACIONAL 21242003,  
Universidad de Valparaíso, Chile.
