/*
 * main.c — CTLE-P4 Inference Engine
 * Compressed Tensor-LUT Engine for ESP32-P4 Nano (RISC-V HP @400MHz)
 *
 * Hardware: ESP32-P4 Nano — 32 MB PSRAM (QSPI 80MHz), 16 MB Flash
 * Model   : TinyStories-15M (Llama-2 architecture)
 *   dim=288, n_layers=6, n_heads=6, vocab=32000, hidden=768, seq=256
 *
 * Memory layout
 *   PSRAM : model weights (7.61 MB), KV cache (3.54 MB), RoPE tables
 *   SRAM  : activations (~12 KB), logits (128 KB)
 *
 * Key innovation — CTLE matvec kernel:
 *   Standard: acc += W[r][c] * x[c]           (weight matrix in RAM)
 *   CTLE:     acc += LUT[nibble[r][c]] * x[c]  (LUT=64B, nibbles streamed)
 *   No full weight matrix ever reconstructed in RAM.
 *
 * Binary format v2 (little-endian):
 *   Header 40B: magic ver dim hidden_dim n_layers n_heads n_kv_heads vocab max_seq flags
 *   Blocks: tag=0 F32 (count u32 + data), tag=1 CTLE (rows u32 + cols u32 + lut[16]f32 + nibbles)
 *
 * Benchmark CSV output (copy to paper):
 *   method, prompt, model_kb, load_ms, prefill_ms, gen_ms_per_tok, tok_per_sec,
 *   tokens_gen, psram_total_kb, psram_used_kb, psram_free_kb, psram_lwm_kb,
 *   sram_total_kb, sram_free_kb
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_spiffs.h"

static const char *TAG = "CTLE";

/* ─── Model constants ───────────────────────────────────────────────────── */
#define MODEL_DIM         288
#define MODEL_HIDDEN      768
#define MODEL_N_LAYERS    6
#define MODEL_N_HEADS     6
#define MODEL_N_KV_HEADS  6
#define MODEL_VOCAB       32000
#define MODEL_MAX_SEQ     256
#define MODEL_HEAD_DIM    (MODEL_DIM / MODEL_N_HEADS)   /* 48 */
#define MODEL_KV_DIM      (MODEL_DIM * MODEL_N_KV_HEADS / MODEL_N_HEADS) /* 288 */

#define CTLE_MAGIC    0x454C5443u
#define CTLE_VERSION  2u
#define TAG_F32       0u
#define TAG_CTLE      1u
#define TAG_INT4U     2u   /* INT4 Uniform  — one scale per tensor   */
#define TAG_INT4BW    3u   /* INT4 Block-wise — one scale per G=32 cols */
#define TAG_PCTLE     4u   /* Product-LUT CTLE — same data as CTLE, no mul in inner loop */
#define TAG_CTLE5     5u   /* CTLE-5b: K=32 centroids, 5-bit packed indices */
#define INT4_OFFSET   7    /* stored nibble = signed_q + 7 */
/* P-CTLE activation quantisation levels.
 * 16 = 4-bit (original, 256-entry product LUT)
 * 32 = 5-bit (experimental, 512-entry product LUT, PPL 33→20) */
#define PCTLE_LEVELS  32

/* ─── Hardcoded benchmark prompts (pre-tokenized via SentencePiece) ──────── */
/* "Once upon a time" */
static const int32_t PROMPT_ONCE_UPON[] = {9038, 2501, 263, 931};
#define PROMPT_ONCE_UPON_LEN 4

/* "Tom and his dog went to the park" */
static const int32_t PROMPT_TOM_DOG[] = {4335, 322, 670, 11203, 3512, 304, 278, 14089};
#define PROMPT_TOM_DOG_LEN 8

/* "The little girl named Lily" */
static const int32_t PROMPT_LILY[] = {450, 2217, 7826, 4257, 365, 2354};
#define PROMPT_LILY_LEN 6

#define GEN_TOKENS   64
#define TEMPERATURE  0.8f
#define TOP_P        0.9f

/* ─── Weight block descriptor ───────────────────────────────────────────── */
typedef struct {
    uint8_t  tag;
    uint32_t rows, cols;
    /* TAG_F32 */
    float   *f32;
    /* TAG_CTLE / TAG_PCTLE: 16 entries (64B)
     * TAG_CTLE5:            32 entries (128B) */
    float    lut[32];
    /* TAG_CTLE / TAG_PCTLE / TAG_INT4U / TAG_INT4BW: 4-bit nibbles */
    uint8_t *nibbles;
    /* TAG_CTLE5: 5-bit packed index stream (8 indices per 5 bytes) */
    uint8_t *pack5;
    /* TAG_INT4U */
    float    scale;
    /* TAG_INT4BW */
    float   *scales;    /* [rows * n_groups] flat, n_groups = ceil(cols/group_size) */
    uint32_t group_size;
} WeightBlock;

/* ─── Full model weight table ───────────────────────────────────────────── */
typedef struct {
    WeightBlock tok_emb;
    WeightBlock attn_norm[MODEL_N_LAYERS];
    WeightBlock wq[MODEL_N_LAYERS];
    WeightBlock wk[MODEL_N_LAYERS];
    WeightBlock wv[MODEL_N_LAYERS];
    WeightBlock wo[MODEL_N_LAYERS];
    WeightBlock ffn_norm[MODEL_N_LAYERS];
    WeightBlock w1[MODEL_N_LAYERS];
    WeightBlock w2[MODEL_N_LAYERS];
    WeightBlock w3[MODEL_N_LAYERS];
    WeightBlock norm;
} Model;

/* ─── KV cache + activations ────────────────────────────────────────────── */
typedef struct {
    float *k_cache[MODEL_N_LAYERS];  /* [MODEL_MAX_SEQ * MODEL_KV_DIM] each */
    float *v_cache[MODEL_N_LAYERS];
    float *rope_cos;  /* [MODEL_MAX_SEQ * MODEL_HEAD_DIM/2] */
    float *rope_sin;
    /* Small activation buffers — live in SRAM via struct fields */
    float x[MODEL_DIM];
    float xb[MODEL_DIM];
    float q[MODEL_DIM];
    float k[MODEL_KV_DIM];
    float v[MODEL_KV_DIM];
    float att[MODEL_N_HEADS * MODEL_MAX_SEQ];
    float hb[MODEL_HIDDEN];
    float hb2[MODEL_HIDDEN];
    float logits[MODEL_VOCAB];
} RunState;

static Model     g_model;
static RunState *g_state;  /* allocated in PSRAM */

/* ─── Profiling globals (set during init, read in every run_benchmark) ──────── */
static char    g_method_name[32] = "unknown"; /* auto-detected from tensor tags   */
static size_t  g_model_bytes     = 0;         /* bytes read from SPIFFS            */
static int64_t g_load_ms         = 0;         /* Flash→PSRAM load time (ms)        */
static size_t  g_psram_total_kb  = 0;         /* captured once at boot             */
static size_t  g_psram_runstate_kb = 0;       /* free after KV-cache + RoPE alloc  */
static size_t  g_psram_model_kb  = 0;         /* free after full model loaded       */
static size_t  g_sram_total_kb   = 0;         /* internal SRAM total (at boot)      */

/* P-CTLE scratch buffers — static to avoid stack overflow.
 * Inference is single-threaded; no mutex needed.
 * Max cols = hidden_dim = 768.                                              */
static uint8_t s_pctle_a_idx[768];              /* activation indices per GEMV call   */
static float   s_pctle_lut_p[PCTLE_LEVELS*16]; /* product LUT = lut_a[P] ⊗ lut_w[16]
                                                   P=16→256 entries (1KB)
                                                   P=32→512 entries (2KB)             */

/* Auto-detect compression format from the tok_emb tensor tag.
 * NOTE: CTLE variants (K-means / GA / PSO) all write TAG_CTLE and are
 * indistinguishable at runtime — their inference timings are therefore
 * identical by construction (same byte layout, same matvec kernel).     */
static void detect_method(void)
{
    switch (g_model.tok_emb.tag) {
        case TAG_CTLE:   snprintf(g_method_name, sizeof(g_method_name), "CTLE");   break;
        case TAG_PCTLE:  snprintf(g_method_name, sizeof(g_method_name),
                                 "PCTLE%d", PCTLE_LEVELS);                         break;
        case TAG_CTLE5:  snprintf(g_method_name, sizeof(g_method_name), "CTLE5");  break;
        case TAG_INT4U:  snprintf(g_method_name, sizeof(g_method_name), "INT4U");  break;
        case TAG_INT4BW: snprintf(g_method_name, sizeof(g_method_name), "INT4BW"); break;
        case TAG_F32:    snprintf(g_method_name, sizeof(g_method_name), "FP32");   break;
        default:
            snprintf(g_method_name, sizeof(g_method_name), "TAG%u",
                     (unsigned)g_model.tok_emb.tag);
            break;
    }
    ESP_LOGI(TAG, "Detected method : %s  (tok_emb tag=%u, model=%zu B)",
             g_method_name, (unsigned)g_model.tok_emb.tag, g_model_bytes);
}

/* ════════════════════════════════════════════════════════════════════════════
 * SPIFFS + binary loader
 * ════════════════════════════════════════════════════════════════════════════ */

static esp_err_t read_block(FILE *f, WeightBlock *b)
{
    uint8_t tag;
    if (fread(&tag, 1, 1, f) != 1) return ESP_FAIL;
    b->tag = tag;

    if (tag == TAG_F32) {
        uint32_t count;
        if (fread(&count, 4, 1, f) != 1) return ESP_FAIL;
        b->rows = count; b->cols = 1;
        b->f32 = heap_caps_malloc(count * sizeof(float), MALLOC_CAP_SPIRAM);
        if (!b->f32) { ESP_LOGE(TAG, "OOM F32 %lu", (unsigned long)count); return ESP_ERR_NO_MEM; }
        fread(b->f32, sizeof(float), count, f);
        b->nibbles = NULL; b->scales = NULL;
    } else if (tag == TAG_CTLE || tag == TAG_PCTLE) {
        /* P-CTLE has identical data layout to CTLE (lut[16] + packed nibbles).
         * The tag byte is the only difference; runtime kernel is selected later. */
        if (fread(&b->rows, 4, 1, f) != 1) return ESP_FAIL;
        if (fread(&b->cols, 4, 1, f) != 1) return ESP_FAIL;
        fread(b->lut, sizeof(float), 16, f);
        uint32_t n_bytes = (b->rows * b->cols + 1) / 2;
        b->nibbles = heap_caps_malloc(n_bytes, MALLOC_CAP_SPIRAM);
        if (!b->nibbles) { ESP_LOGE(TAG, "OOM nibbles %lu", (unsigned long)n_bytes); return ESP_ERR_NO_MEM; }
        fread(b->nibbles, 1, n_bytes, f);
        b->f32 = NULL; b->scales = NULL; b->pack5 = NULL;
    } else if (tag == TAG_CTLE5) {
        /* CTLE-5b: K=32 codebook (128 B) + 5-bit packed index stream */
        if (fread(&b->rows, 4, 1, f) != 1) return ESP_FAIL;
        if (fread(&b->cols, 4, 1, f) != 1) return ESP_FAIL;
        fread(b->lut, sizeof(float), 32, f);          /* 128 bytes */
        uint32_t n_groups = ((uint32_t)b->rows * b->cols + 7) / 8;
        uint32_t n_bytes  = n_groups * 5;
        b->pack5 = heap_caps_malloc(n_bytes, MALLOC_CAP_SPIRAM);
        if (!b->pack5) { ESP_LOGE(TAG, "OOM CTLE5 pack5 %lu", (unsigned long)n_bytes); return ESP_ERR_NO_MEM; }
        fread(b->pack5, 1, n_bytes, f);
        b->f32 = NULL; b->nibbles = NULL; b->scales = NULL;
    } else if (tag == TAG_INT4U) {
        if (fread(&b->rows, 4, 1, f) != 1) return ESP_FAIL;
        if (fread(&b->cols, 4, 1, f) != 1) return ESP_FAIL;
        fread(&b->scale, sizeof(float), 1, f);
        uint32_t n_bytes = (b->rows * b->cols + 1) / 2;
        b->nibbles = heap_caps_malloc(n_bytes, MALLOC_CAP_SPIRAM);
        if (!b->nibbles) { ESP_LOGE(TAG, "OOM INT4U nibbles %lu", (unsigned long)n_bytes); return ESP_ERR_NO_MEM; }
        fread(b->nibbles, 1, n_bytes, f);
        b->f32 = NULL; b->scales = NULL; b->pack5 = NULL;
    } else if (tag == TAG_INT4BW) {
        if (fread(&b->rows,       4, 1, f) != 1) return ESP_FAIL;
        if (fread(&b->cols,       4, 1, f) != 1) return ESP_FAIL;
        if (fread(&b->group_size, 4, 1, f) != 1) return ESP_FAIL;
        uint32_t n_groups = (b->cols + b->group_size - 1) / b->group_size;
        uint32_t n_scales = b->rows * n_groups;
        b->scales = heap_caps_malloc(n_scales * sizeof(float), MALLOC_CAP_SPIRAM);
        if (!b->scales) { ESP_LOGE(TAG, "OOM INT4BW scales %lu", (unsigned long)n_scales); return ESP_ERR_NO_MEM; }
        fread(b->scales, sizeof(float), n_scales, f);
        uint32_t n_bytes = (b->rows * b->cols + 1) / 2;
        b->nibbles = heap_caps_malloc(n_bytes, MALLOC_CAP_SPIRAM);
        if (!b->nibbles) { ESP_LOGE(TAG, "OOM INT4BW nibbles %lu", (unsigned long)n_bytes); return ESP_ERR_NO_MEM; }
        fread(b->nibbles, 1, n_bytes, f);
        b->f32 = NULL; b->pack5 = NULL;
    } else {
        ESP_LOGE(TAG, "Unknown block tag %u", tag);
        return ESP_FAIL;
    }
    return ESP_OK;
}

static esp_err_t load_model(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { ESP_LOGE(TAG, "Cannot open %s", path); return ESP_FAIL; }

    uint32_t magic, version;
    fread(&magic,   4, 1, f);
    fread(&version, 4, 1, f);
    if (magic != CTLE_MAGIC || version != CTLE_VERSION) {
        ESP_LOGE(TAG, "Bad header: magic=0x%08lX ver=%lu",
                 (unsigned long)magic, (unsigned long)version);
        fclose(f); return ESP_FAIL;
    }
    uint32_t hdr[8];
    fread(hdr, 4, 8, f);
    ESP_LOGI(TAG, "CTLE v2 — dim=%lu layers=%lu vocab=%lu",
             (unsigned long)hdr[0], (unsigned long)hdr[2], (unsigned long)hdr[5]);

    if (read_block(f, &g_model.tok_emb) != ESP_OK) goto fail;

    for (int l = 0; l < MODEL_N_LAYERS; l++) {
        if (read_block(f, &g_model.attn_norm[l]) != ESP_OK) goto fail;
        if (read_block(f, &g_model.wq[l])        != ESP_OK) goto fail;
        if (read_block(f, &g_model.wk[l])        != ESP_OK) goto fail;
        if (read_block(f, &g_model.wv[l])        != ESP_OK) goto fail;
        if (read_block(f, &g_model.wo[l])        != ESP_OK) goto fail;
        if (read_block(f, &g_model.ffn_norm[l])  != ESP_OK) goto fail;
        if (read_block(f, &g_model.w1[l])        != ESP_OK) goto fail;
        if (read_block(f, &g_model.w2[l])        != ESP_OK) goto fail;
        if (read_block(f, &g_model.w3[l])        != ESP_OK) goto fail;
    }
    if (read_block(f, &g_model.norm) != ESP_OK) goto fail;

    g_model_bytes = (size_t)ftell(f);   /* total bytes read from SPIFFS */
    fclose(f);
    ESP_LOGI(TAG, "Model loaded OK  (%zu KB)", g_model_bytes / 1024);
    return ESP_OK;
fail:
    fclose(f); return ESP_FAIL;
}

static esp_err_t init_spiffs(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path              = "/spiffs",
        .partition_label        = "storage",
        .max_files              = 4,
        .format_if_mount_failed = false,
    };
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SPIFFS mount failed: %s", esp_err_to_name(ret));
        return ret;
    }
    size_t total, used;
    esp_spiffs_info("storage", &total, &used);
    ESP_LOGI(TAG, "SPIFFS: %zu KB total, %zu KB used", total / 1024, used / 1024);
    return ESP_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Math
 * ════════════════════════════════════════════════════════════════════════════ */

static void rmsnorm(float *out, const float *x, const float *w, int size)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)size + 1e-5f);
    for (int i = 0; i < size; i++) out[i] = w[i] * (ss * x[i]);
}

static void softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* ════════════════════════════════════════════════════════════════════════════
 * CTLE matvec — the core kernel
 *
 * y[r] = Σ_c  LUT[ nibble[r,c] ] * x[c]
 *
 * Two 4-bit indices per byte, low nibble first (matching Python pack_nibbles).
 * LUT fits in 16 float registers; nibbles streamed sequentially from PSRAM.
 * ════════════════════════════════════════════════════════════════════════════ */
static void ctle_matvec(const WeightBlock *wb, const float *x, float *y,
                        int rows, int cols)
{
    const float   *lut     = wb->lut;
    const uint8_t *nibbles = wb->nibbles;
    int byte_idx = 0;

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        int c = 0;
        for (; c + 1 < cols; c += 2) {
            uint8_t b = nibbles[byte_idx++];
            acc += lut[b & 0x0F] * x[c];
            acc += lut[b >>   4] * x[c + 1];
        }
        if (c < cols) {
            acc += lut[nibbles[byte_idx++] & 0x0F] * x[c];
        }
        y[r] = acc;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * CTLE-5b matvec — K=32 codebook, 5-bit packed indices
 *
 * Packing layout (matches Python _pack_5bit):
 *   8 indices per 5 bytes (40 bits).  Index i occupies bits [5i : 5i+5].
 *   cols must be a multiple of 8 (satisfied for all TinyStories-15M weights:
 *   dim=288=8×36, hidden=768=8×96).
 * ════════════════════════════════════════════════════════════════════════════ */
static inline void decode5_group(const uint8_t *p, uint8_t *out)
{
    /* Load 5 bytes into a 64-bit register and extract 8 × 5-bit fields. */
    uint64_t v = (uint64_t)p[0]
               | ((uint64_t)p[1] << 8)
               | ((uint64_t)p[2] << 16)
               | ((uint64_t)p[3] << 24)
               | ((uint64_t)p[4] << 32);
    out[0] = (uint8_t)((v >>  0) & 0x1F);
    out[1] = (uint8_t)((v >>  5) & 0x1F);
    out[2] = (uint8_t)((v >> 10) & 0x1F);
    out[3] = (uint8_t)((v >> 15) & 0x1F);
    out[4] = (uint8_t)((v >> 20) & 0x1F);
    out[5] = (uint8_t)((v >> 25) & 0x1F);
    out[6] = (uint8_t)((v >> 30) & 0x1F);
    out[7] = (uint8_t)((v >> 35) & 0x1F);
}

static void ctle5_matvec(const WeightBlock *wb, const float *x, float *y,
                         int rows, int cols)
{
    const float   *lut = wb->lut;   /* 32-entry codebook (128 B) */
    const uint8_t *p5  = wb->pack5;
    int groups_per_row = cols / 8;  /* cols is a multiple of 8 */

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        uint8_t idx[8];
        for (int g = 0; g < groups_per_row; g++) {
            decode5_group(p5, idx);
            p5 += 5;
            int base = g << 3;
            acc += lut[idx[0]] * x[base + 0];
            acc += lut[idx[1]] * x[base + 1];
            acc += lut[idx[2]] * x[base + 2];
            acc += lut[idx[3]] * x[base + 3];
            acc += lut[idx[4]] * x[base + 4];
            acc += lut[idx[5]] * x[base + 5];
            acc += lut[idx[6]] * x[base + 6];
            acc += lut[idx[7]] * x[base + 7];
        }
        y[r] = acc;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * P-CTLE matvec — Product-LUT Compressed Tensor-Linear Engine
 *
 * Replaces MAC (acc += x * w) with Product-Lookup-Accumulate:
 *   acc += LUT_p[(a_idx << 4) | w_idx]
 *
 * Steps per GEMV call:
 *   1. Quantise x[cols] → a_idx[cols] uniform 16-level (O(N), no mul)
 *   2. Build LUT_p[256] = outer_product(LUT_a, LUT_w)  (256 muls, amortised)
 *   3. Inner loop: addr = (a_idx[c]<<4)|w_idx → acc += LUT_p[addr]  (0 muls)
 *
 * Static scratch buffers used (no stack overflow risk):
 *   s_pctle_a_idx[768]   activation indices
 *   s_pctle_lut_p[256]   product LUT (float32, 1 KB)
 * ════════════════════════════════════════════════════════════════════════════ */
static void pctle_matvec(const WeightBlock *wb, const float *x, float *y,
                         int rows, int cols)
{
    /* 1 ── Quantise activations → a_idx + LUT_a (uniform min-max, 16 levels) */
    float x_min = x[0], x_max = x[0];
    for (int c = 1; c < cols; c++) {
        if (x[c] < x_min) x_min = x[c];
        if (x[c] > x_max) x_max = x[c];
    }
    float step = (x_max - x_min) / (float)(PCTLE_LEVELS - 1);
    if (step < 1e-8f) step = 1e-8f;
    float inv_step = 1.0f / step;

    float lut_a[PCTLE_LEVELS];
    for (int k = 0; k < PCTLE_LEVELS; k++)
        lut_a[k] = x_min + k * step;
    for (int c = 0; c < cols; c++) {
        int idx = (int)((x[c] - x_min) * inv_step + 0.5f);
        if (idx < 0)             idx = 0;
        if (idx >= PCTLE_LEVELS) idx = PCTLE_LEVELS - 1;
        s_pctle_a_idx[c] = (uint8_t)idx;
    }

    /* 2 ── Build product LUT: PCTLE_LEVELS×16 multiplications, once per GEMV
     *       Weight codebook always has 16 entries (K=16, 4-bit nibbles).
     *       LUT index: a_idx * 16 + w_idx  (avoids shift for PCTLE_LEVELS>16) */
    const float *lut_w = wb->lut;
    for (int a = 0; a < PCTLE_LEVELS; a++)
        for (int w = 0; w < 16; w++)
            s_pctle_lut_p[a * 16 + w] = lut_a[a] * lut_w[w];

    /* 3 ── Inner loop: lookup-accumulate — zero multiplications */
    const uint8_t *nibbles = wb->nibbles;
    int byte_idx = 0;
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        int c = 0;
        for (; c + 1 < cols; c += 2) {
            uint8_t b  = nibbles[byte_idx++];
            acc += s_pctle_lut_p[s_pctle_a_idx[c]   * 16 + (b & 0x0F)];
            acc += s_pctle_lut_p[s_pctle_a_idx[c+1] * 16 + (b >>   4)];
        }
        if (c < cols) {
            uint8_t b = nibbles[byte_idx++];
            acc += s_pctle_lut_p[s_pctle_a_idx[c] * 16 + (b & 0x0F)];
        }
        y[r] = acc;
    }
}

static void f32_matvec(const float *W, const float *x, float *y, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const float *row = W + (size_t)r * cols;
        for (int c = 0; c < cols; c++) acc += row[c] * x[c];
        y[r] = acc;
    }
}

/* INT4 Uniform matvec:  y[r] = scale * Σ_c (nibble[r,c] - 7) * x[c]  */
static void int4u_matvec(const WeightBlock *wb, const float *x, float *y,
                         int rows, int cols)
{
    const float   scale   = wb->scale;
    const uint8_t *nibbles = wb->nibbles;
    int byte_idx = 0;

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        int c = 0;
        for (; c + 1 < cols; c += 2) {
            uint8_t b = nibbles[byte_idx++];
            acc += ((float)(b & 0x0F) - INT4_OFFSET) * x[c];
            acc += ((float)(b >>   4) - INT4_OFFSET) * x[c + 1];
        }
        if (c < cols) {
            acc += ((float)(nibbles[byte_idx++] & 0x0F) - INT4_OFFSET) * x[c];
        }
        y[r] = acc * scale;
    }
}

/* INT4 Block-wise matvec:  y[r] = Σ_g scale_g * Σ_{c in g} (nibble[r,c]-7)*x[c]  */
static void int4bw_matvec(const WeightBlock *wb, const float *x, float *y,
                          int rows, int cols)
{
    const uint8_t  *nibbles    = wb->nibbles;
    const float    *scales     = wb->scales;
    const uint32_t  group_size = wb->group_size;
    const uint32_t  n_groups   = (cols + group_size - 1) / group_size;
    int byte_idx = 0;

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const float *row_scales = scales + (size_t)r * n_groups;
        for (uint32_t g = 0; g < n_groups; g++) {
            float gs = row_scales[g];
            uint32_t c_start = g * group_size;
            uint32_t c_end   = c_start + group_size;
            if ((uint32_t)c_end > (uint32_t)cols) c_end = (uint32_t)cols;
            float gacc = 0.0f;
            uint32_t c = c_start;
            for (; c + 1 < c_end; c += 2) {
                uint8_t b = nibbles[byte_idx++];
                gacc += ((float)(b & 0x0F) - INT4_OFFSET) * x[c];
                gacc += ((float)(b >>   4) - INT4_OFFSET) * x[c + 1];
            }
            if (c < c_end) {
                gacc += ((float)(nibbles[byte_idx++] & 0x0F) - INT4_OFFSET) * x[c];
            }
            acc += gs * gacc;
        }
        y[r] = acc;
    }
}

static void matvec(const WeightBlock *wb, const float *x, float *y)
{
    int rows = (int)wb->rows, cols = (int)wb->cols;
    switch (wb->tag) {
        case TAG_CTLE:   ctle_matvec(wb, x, y, rows, cols);      break;
        case TAG_CTLE5:  ctle5_matvec(wb, x, y, rows, cols);     break;
        case TAG_PCTLE:  pctle_matvec(wb, x, y, rows, cols);     break;
        case TAG_INT4U:  int4u_matvec(wb, x, y, rows, cols);     break;
        case TAG_INT4BW: int4bw_matvec(wb, x, y, rows, cols);    break;
        default:         f32_matvec(wb->f32, x, y, rows, cols);  break;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * RoPE
 * ════════════════════════════════════════════════════════════════════════════ */
static void precompute_rope(float *cos_out, float *sin_out)
{
    for (int pos = 0; pos < MODEL_MAX_SEQ; pos++) {
        for (int i = 0; i < MODEL_HEAD_DIM / 2; i++) {
            float freq  = 1.0f / powf(10000.0f, (2.0f * i) / MODEL_HEAD_DIM);
            float angle = (float)pos * freq;
            cos_out[pos * (MODEL_HEAD_DIM / 2) + i] = cosf(angle);
            sin_out[pos * (MODEL_HEAD_DIM / 2) + i] = sinf(angle);
        }
    }
}

static void apply_rope(float *vec, int n_heads, int pos,
                       const float *cos_t, const float *sin_t)
{
    const float *c = cos_t + pos * (MODEL_HEAD_DIM / 2);
    const float *s = sin_t + pos * (MODEL_HEAD_DIM / 2);
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * MODEL_HEAD_DIM;
        for (int i = 0; i < MODEL_HEAD_DIM / 2; i++) {
            float v0 = v[2 * i], v1 = v[2 * i + 1];
            v[2 * i]     = v0 * c[i] - v1 * s[i];
            v[2 * i + 1] = v0 * s[i] + v1 * c[i];
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Transformer forward — one token at position `pos`
 * ════════════════════════════════════════════════════════════════════════════ */
static void transformer_forward(int token, int pos)
{
    RunState   *s = g_state;
    const float *rc = s->rope_cos;
    const float *rs = s->rope_sin;

    /* ── Embedding lookup ──────────────────────────────────────────────── */
    {
        const WeightBlock *emb = &g_model.tok_emb;
        int cols = (int)emb->cols;
        int base = token * cols;
        if (emb->tag == TAG_CTLE || emb->tag == TAG_PCTLE) {
            /* Embedding lookup is a row-select — no activation to quantise */
            const float   *lut = emb->lut;
            const uint8_t *nb  = emb->nibbles;
            for (int c = 0; c < cols; c++) {
                int     bi  = (base + c) / 2;
                uint8_t b   = nb[bi];
                uint8_t idx = ((base + c) & 1) ? (b >> 4) : (b & 0x0F);
                s->x[c] = lut[idx];
            }
        } else if (emb->tag == TAG_CTLE5) {
            /* CTLE5 embedding: row token starts at byte (token * cols/8 * 5).
             * cols=288 is a multiple of 8, so rows are 5-byte-group aligned. */
            const float   *lut   = emb->lut;
            const uint8_t *p5row = emb->pack5 + (size_t)token * (cols / 8) * 5;
            uint8_t idx8[8];
            for (int g = 0; g < cols / 8; g++) {
                decode5_group(p5row, idx8);
                p5row += 5;
                int b0 = g << 3;
                for (int k = 0; k < 8; k++) s->x[b0 + k] = lut[idx8[k]];
            }
        } else if (emb->tag == TAG_INT4U) {
            const uint8_t *nb  = emb->nibbles;
            float sc = emb->scale;
            for (int c = 0; c < cols; c++) {
                int     bi  = (base + c) / 2;
                uint8_t b   = nb[bi];
                uint8_t nib = ((base + c) & 1) ? (b >> 4) : (b & 0x0F);
                s->x[c] = ((float)nib - INT4_OFFSET) * sc;
            }
        } else if (emb->tag == TAG_INT4BW) {
            const uint8_t *nb  = emb->nibbles;
            uint32_t gs = emb->group_size;
            uint32_t ng = (cols + gs - 1) / gs;
            const float *row_sc = emb->scales + (size_t)token * ng;
            for (int c = 0; c < cols; c++) {
                int     bi  = (base + c) / 2;
                uint8_t b   = nb[bi];
                uint8_t nib = ((base + c) & 1) ? (b >> 4) : (b & 0x0F);
                s->x[c] = ((float)nib - INT4_OFFSET) * row_sc[c / gs];
            }
        } else {
            memcpy(s->x, emb->f32 + (size_t)token * MODEL_DIM,
                   MODEL_DIM * sizeof(float));
        }
    }

    /* ── Layers ────────────────────────────────────────────────────────── */
    for (int l = 0; l < MODEL_N_LAYERS; l++) {

        rmsnorm(s->xb, s->x, g_model.attn_norm[l].f32, MODEL_DIM);

        matvec(&g_model.wq[l], s->xb, s->q);
        matvec(&g_model.wk[l], s->xb, s->k);
        matvec(&g_model.wv[l], s->xb, s->v);

        apply_rope(s->q, MODEL_N_HEADS,    pos, rc, rs);
        apply_rope(s->k, MODEL_N_KV_HEADS, pos, rc, rs);

        /* Store into KV cache */
        float *kc = s->k_cache[l] + (size_t)pos * MODEL_KV_DIM;
        float *vc = s->v_cache[l] + (size_t)pos * MODEL_KV_DIM;
        memcpy(kc, s->k, MODEL_KV_DIM * sizeof(float));
        memcpy(vc, s->v, MODEL_KV_DIM * sizeof(float));

        /* Causal scaled dot-product attention */
        float scale = 1.0f / sqrtf((float)MODEL_HEAD_DIM);
        for (int h = 0; h < MODEL_N_HEADS; h++) {
            const float *q_h  = s->q   + h * MODEL_HEAD_DIM;
            float       *att  = s->att + h * MODEL_MAX_SEQ;
            for (int t = 0; t <= pos; t++) {
                const float *k_t = s->k_cache[l] + (size_t)t * MODEL_KV_DIM
                                   + h * MODEL_HEAD_DIM;
                float dot = 0.0f;
                for (int i = 0; i < MODEL_HEAD_DIM; i++) dot += q_h[i] * k_t[i];
                att[t] = dot * scale;
            }
            softmax(att, pos + 1);

            float *xb_h = s->xb + h * MODEL_HEAD_DIM;
            memset(xb_h, 0, MODEL_HEAD_DIM * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *v_t = s->v_cache[l] + (size_t)t * MODEL_KV_DIM
                                   + h * MODEL_HEAD_DIM;
                for (int i = 0; i < MODEL_HEAD_DIM; i++) xb_h[i] += att[t] * v_t[i];
            }
        }

        /* Output projection + residual (reuse s->q as temp) */
        matvec(&g_model.wo[l], s->xb, s->q);
        for (int i = 0; i < MODEL_DIM; i++) s->x[i] += s->q[i];

        /* SwiGLU FFN */
        rmsnorm(s->xb, s->x, g_model.ffn_norm[l].f32, MODEL_DIM);
        matvec(&g_model.w1[l], s->xb, s->hb);
        matvec(&g_model.w3[l], s->xb, s->hb2);
        for (int i = 0; i < MODEL_HIDDEN; i++) s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        matvec(&g_model.w2[l], s->hb, s->xb);
        for (int i = 0; i < MODEL_DIM; i++) s->x[i] += s->xb[i];
    }

    /* ── Output logits (tied weights = tok_emb.T * xb) ────────────────── */
    rmsnorm(s->xb, s->x, g_model.norm.f32, MODEL_DIM);

    {
        const WeightBlock *emb = &g_model.tok_emb;
        int cols = (int)emb->cols;
        if (emb->tag == TAG_CTLE) {
            const float   *lut = emb->lut;
            const uint8_t *nb  = emb->nibbles;
            for (int v = 0; v < MODEL_VOCAB; v++) {
                float acc  = 0.0f;
                int   base = v * cols;
                for (int c = 0; c < cols; c++) {
                    int     bi  = (base + c) / 2;
                    uint8_t b   = nb[bi];
                    uint8_t idx = ((base + c) & 1) ? (b >> 4) : (b & 0x0F);
                    acc += lut[idx] * s->xb[c];
                }
                s->logits[v] = acc;
            }
        } else if (emb->tag == TAG_CTLE5) {
            /* CTLE5 logit projection: sequential 5-bit decode, 8 per group */
            const float   *lut = emb->lut;
            const uint8_t *p5  = emb->pack5;
            int groups_per_row = cols / 8;
            for (int v = 0; v < MODEL_VOCAB; v++) {
                float acc = 0.0f;
                uint8_t idx8[8];
                for (int g = 0; g < groups_per_row; g++) {
                    decode5_group(p5, idx8);
                    p5 += 5;
                    int b0 = g << 3;
                    acc += lut[idx8[0]] * s->xb[b0 + 0];
                    acc += lut[idx8[1]] * s->xb[b0 + 1];
                    acc += lut[idx8[2]] * s->xb[b0 + 2];
                    acc += lut[idx8[3]] * s->xb[b0 + 3];
                    acc += lut[idx8[4]] * s->xb[b0 + 4];
                    acc += lut[idx8[5]] * s->xb[b0 + 5];
                    acc += lut[idx8[6]] * s->xb[b0 + 6];
                    acc += lut[idx8[7]] * s->xb[b0 + 7];
                }
                s->logits[v] = acc;
            }
        } else if (emb->tag == TAG_PCTLE) {
            /* Logits via P-CTLE: quantise xb[cols] to PCTLE_LEVELS uniform levels,
             * build product LUT (PCTLE_LEVELS×16 entries), then lookup-accumulate. */
            float x_min = s->xb[0], x_max = s->xb[0];
            for (int c = 1; c < cols; c++) {
                if (s->xb[c] < x_min) x_min = s->xb[c];
                if (s->xb[c] > x_max) x_max = s->xb[c];
            }
            float step = (x_max - x_min) / (float)(PCTLE_LEVELS - 1);
            if (step < 1e-8f) step = 1e-8f;
            float inv_step = 1.0f / step;
            float lut_a[PCTLE_LEVELS];
            for (int k = 0; k < PCTLE_LEVELS; k++)
                lut_a[k] = x_min + k * step;
            for (int c = 0; c < cols; c++) {
                int idx = (int)((s->xb[c] - x_min) * inv_step + 0.5f);
                if (idx < 0)             idx = 0;
                if (idx >= PCTLE_LEVELS) idx = PCTLE_LEVELS - 1;
                s_pctle_a_idx[c] = (uint8_t)idx;
            }
            const float *lut_w = emb->lut;
            for (int a = 0; a < PCTLE_LEVELS; a++)
                for (int w = 0; w < 16; w++)
                    s_pctle_lut_p[a * 16 + w] = lut_a[a] * lut_w[w];
            const uint8_t *nb = emb->nibbles;
            for (int v = 0; v < MODEL_VOCAB; v++) {
                float acc = 0.0f;
                int   base = v * cols;
                int   c    = 0;
                for (; c + 1 < cols; c += 2) {
                    int     bi = (base + c) / 2;
                    uint8_t b  = nb[bi];
                    acc += s_pctle_lut_p[s_pctle_a_idx[c]   * 16 + (b & 0x0F)];
                    acc += s_pctle_lut_p[s_pctle_a_idx[c+1] * 16 + (b >>   4)];
                }
                if (c < cols) {
                    uint8_t b = nb[(base + c) / 2];
                    acc += s_pctle_lut_p[s_pctle_a_idx[c] * 16 + (b & 0x0F)];
                }
                s->logits[v] = acc;
            }
        } else if (emb->tag == TAG_INT4U) {
            const uint8_t *nb = emb->nibbles;
            float sc = emb->scale;
            for (int v = 0; v < MODEL_VOCAB; v++) {
                float acc  = 0.0f;
                int   base = v * cols;
                for (int c = 0; c < cols; c++) {
                    int     bi  = (base + c) / 2;
                    uint8_t b   = nb[bi];
                    uint8_t nib = ((base + c) & 1) ? (b >> 4) : (b & 0x0F);
                    acc += ((float)nib - INT4_OFFSET) * s->xb[c];
                }
                s->logits[v] = acc * sc;
            }
        } else if (emb->tag == TAG_INT4BW) {
            const uint8_t *nb = emb->nibbles;
            uint32_t gs = emb->group_size;
            uint32_t ng = (cols + gs - 1) / gs;
            for (int v = 0; v < MODEL_VOCAB; v++) {
                float acc  = 0.0f;
                int   base = v * cols;
                const float *row_sc = emb->scales + (size_t)v * ng;
                for (int c = 0; c < cols; c++) {
                    int     bi  = (base + c) / 2;
                    uint8_t b   = nb[bi];
                    uint8_t nib = ((base + c) & 1) ? (b >> 4) : (b & 0x0F);
                    acc += ((float)nib - INT4_OFFSET) * row_sc[c / gs] * s->xb[c];
                }
                s->logits[v] = acc;
            }
        } else {
            f32_matvec(emb->f32, s->xb, s->logits, MODEL_VOCAB, MODEL_DIM);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Sampling (temperature + top-p, xorshift RNG)
 * ════════════════════════════════════════════════════════════════════════════ */
static uint32_t g_rng = 0xDEADBEEFu;
static uint32_t xorshift32(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static int sample_argmax(void)
{
    int best = 0;
    for (int i = 1; i < MODEL_VOCAB; i++)
        if (g_state->logits[i] > g_state->logits[best]) best = i;
    return best;
}

static int sample_top_p(float temperature)
{
    float *logits = g_state->logits;

    /* Softmax with temperature */
    float mx = logits[0];
    for (int i = 1; i < MODEL_VOCAB; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < MODEL_VOCAB; i++) {
        logits[i] = expf((logits[i] - mx) / temperature);
        sum += logits[i];
    }
    for (int i = 0; i < MODEL_VOCAB; i++) logits[i] /= sum;

    /* Uniform random in [0,1) */
    float coin = (float)(xorshift32() & 0xFFFFFFu) / (float)0x1000000u;

    /* Linear top-p scan — two passes O(V), no extra alloc */
    float cumul = 0.0f;
    float threshold = 1.0f - TOP_P;
    for (int i = 0; i < MODEL_VOCAB; i++) if (logits[i] >= threshold) cumul += logits[i];
    float target = coin * cumul;
    cumul = 0.0f;
    for (int i = 0; i < MODEL_VOCAB; i++) {
        if (logits[i] < threshold) continue;
        cumul += logits[i];
        if (cumul >= target) return i;
    }
    return sample_argmax();
}

/* ════════════════════════════════════════════════════════════════════════════
 * Benchmark harness
 * ════════════════════════════════════════════════════════════════════════════ */
static void reset_kv_cache(void)
{
    for (int l = 0; l < MODEL_N_LAYERS; l++) {
        memset(g_state->k_cache[l], 0, MODEL_MAX_SEQ * MODEL_KV_DIM * sizeof(float));
        memset(g_state->v_cache[l], 0, MODEL_MAX_SEQ * MODEL_KV_DIM * sizeof(float));
    }
}

static void run_benchmark(const char *name, const int32_t *prompt, int plen)
{
    reset_kv_cache();

    int32_t buf[MODEL_MAX_SEQ];
    memcpy(buf, prompt, plen * sizeof(int32_t));
    int n = plen;

    /* Prefill */
    int64_t t0 = esp_timer_get_time();
    for (int pos = 0; pos < plen; pos++)
        transformer_forward(buf[pos], pos);
    int64_t t_prefill = esp_timer_get_time() - t0;

    /* Generation.
     *
     * The MARK lines bracket the generation window for the external power
     * logger. They are printed at the boundaries only, never inside the loop:
     * UART activity during the window would add current that correlates with
     * token rate and would therefore bias the per-format energy comparison.
     * Flushed immediately so the host timestamp reflects the boundary rather
     * than when the stdout buffer happens to drain. */
    printf("MARK,gen_start,%s,%s\n", g_method_name, name);
    fflush(stdout);

    int64_t t1 = esp_timer_get_time();
    int next = sample_top_p(TEMPERATURE);
    buf[n++] = next;

    for (int g = 1; g < GEN_TOKENS && n < MODEL_MAX_SEQ; g++) {
        transformer_forward(next, plen + g - 1);
        next = sample_top_p(TEMPERATURE);
        buf[n++] = next;
        if (next == 1) break;   /* EOS */
    }
    int64_t t_gen = esp_timer_get_time() - t1;

    printf("MARK,gen_end,%s,%s,%lld\n", g_method_name, name, (long long)t_gen);
    fflush(stdout);

    int gen_toks      = n - plen;
    float prefill_ms  = (float)t_prefill / 1000.0f;
    float gen_ms_tok  = (float)t_gen / 1000.0f / (float)gen_toks;
    float tok_per_sec = 1000.0f / gen_ms_tok;

    /* Memory snapshot at inference time (after all allocs are live) */
    size_t psram_free_kb = heap_caps_get_free_size(MALLOC_CAP_SPIRAM)    / 1024;
    size_t psram_used_kb = g_psram_total_kb - psram_free_kb;
    size_t sram_free_kb  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL)  / 1024;
    /* Watermark: minimum free PSRAM ever seen (catches any transient spike) */
    size_t psram_lwm_kb  = heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM) / 1024;

    ESP_LOGI(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "Method    : %s", g_method_name);
    ESP_LOGI(TAG, "Prompt    : %s (%d tokens)", name, plen);
    ESP_LOGI(TAG, "Prefill   : %.1f ms", prefill_ms);
    ESP_LOGI(TAG, "Gen speed : %.1f ms/tok  (%.3f tok/s)", gen_ms_tok, tok_per_sec);
    ESP_LOGI(TAG, "Generated : %d tokens", gen_toks);
    ESP_LOGI(TAG, "Model size: %zu KB  (loaded in %lld ms)", g_model_bytes/1024, g_load_ms);
    ESP_LOGI(TAG, "PSRAM     : total=%zu KB  used=%zu KB  free=%zu KB  lwm=%zu KB",
             g_psram_total_kb, psram_used_kb, psram_free_kb, psram_lwm_kb);
    ESP_LOGI(TAG, "SRAM      : total=%zu KB  free=%zu KB",
             g_sram_total_kb, sram_free_kb);

    /* Print generated token IDs for offline decoding in Python */
    printf("TOKENS[%s]:", name);
    for (int i = plen; i < n; i++) printf(" %ld", (long)buf[i]);
    printf("\n");

    /* ── CSV row (columns match header printed in app_main) ──────────────────
     * method, prompt, model_kb, load_ms, prefill_ms, gen_ms_tok, tok_per_sec,
     * tokens_gen, psram_total_kb, psram_used_kb, psram_free_kb, psram_lwm_kb,
     * sram_total_kb, sram_free_kb
     * ───────────────────────────────────────────────────────────────────────── */
    printf("CSV,%s,%s,%zu,%lld,%.1f,%.2f,%.3f,%d,%zu,%zu,%zu,%zu,%zu,%zu\n",
           g_method_name, name,
           g_model_bytes / 1024, g_load_ms,
           prefill_ms, gen_ms_tok, tok_per_sec,
           gen_toks,
           g_psram_total_kb, psram_used_kb, psram_free_kb, psram_lwm_kb,
           g_sram_total_kb, sram_free_kb);
}

/* ════════════════════════════════════════════════════════════════════════════
 * RunState PSRAM allocation + RoPE init
 * ════════════════════════════════════════════════════════════════════════════ */
static esp_err_t alloc_run_state(void)
{
    g_state = heap_caps_calloc(1, sizeof(RunState), MALLOC_CAP_SPIRAM);
    if (!g_state) { ESP_LOGE(TAG, "OOM: RunState"); return ESP_ERR_NO_MEM; }

    for (int l = 0; l < MODEL_N_LAYERS; l++) {
        size_t kv_sz = MODEL_MAX_SEQ * MODEL_KV_DIM * sizeof(float);
        g_state->k_cache[l] = heap_caps_calloc(1, kv_sz, MALLOC_CAP_SPIRAM);
        g_state->v_cache[l] = heap_caps_calloc(1, kv_sz, MALLOC_CAP_SPIRAM);
        if (!g_state->k_cache[l] || !g_state->v_cache[l]) {
            ESP_LOGE(TAG, "OOM: KV cache layer %d", l); return ESP_ERR_NO_MEM;
        }
    }

    size_t rope_sz = MODEL_MAX_SEQ * (MODEL_HEAD_DIM / 2) * sizeof(float);
    g_state->rope_cos = heap_caps_malloc(rope_sz, MALLOC_CAP_SPIRAM);
    g_state->rope_sin = heap_caps_malloc(rope_sz, MALLOC_CAP_SPIRAM);
    if (!g_state->rope_cos || !g_state->rope_sin) {
        ESP_LOGE(TAG, "OOM: RoPE tables"); return ESP_ERR_NO_MEM;
    }
    precompute_rope(g_state->rope_cos, g_state->rope_sin);
    return ESP_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * app_main
 * ════════════════════════════════════════════════════════════════════════════ */
void app_main(void)
{
    /* ── 0. Boot memory snapshot (before any heap allocation) ─────────────── */
    g_psram_total_kb = heap_caps_get_total_size(MALLOC_CAP_SPIRAM)    / 1024;
    g_sram_total_kb  = heap_caps_get_total_size(MALLOC_CAP_INTERNAL)  / 1024;
    size_t psram_free_boot = heap_caps_get_free_size(MALLOC_CAP_SPIRAM)   / 1024;
    size_t sram_free_boot  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024;

    ESP_LOGI(TAG, "CTLE-P4 Engine  —  ESP32-P4 Nano (RISC-V HP @360MHz, rev 1.3)");
    ESP_LOGI(TAG, "PSRAM : total=%zu KB  free=%zu KB", g_psram_total_kb, psram_free_boot);
    ESP_LOGI(TAG, "SRAM  : total=%zu KB  free=%zu KB", g_sram_total_kb,  sram_free_boot);

    /* ── 1. SPIFFS ────────────────────────────────────────────────────────── */
    if (init_spiffs() != ESP_OK) { ESP_LOGE(TAG, "SPIFFS failed"); return; }

    /* ── 2. KV-cache + RoPE tables + RunState struct (PSRAM) ─────────────── */
    if (alloc_run_state() != ESP_OK) { ESP_LOGE(TAG, "RunState alloc failed"); return; }
    g_psram_runstate_kb = heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024;
    ESP_LOGI(TAG, "PSRAM after RunState: %zu KB free  (RunState used ~%zu KB)",
             g_psram_runstate_kb, psram_free_boot - g_psram_runstate_kb);

    /* ── 3. Load model from SPIFFS → PSRAM ───────────────────────────────── */
    int64_t t0 = esp_timer_get_time();
    if (load_model("/spiffs/model.bin") != ESP_OK) {
        ESP_LOGE(TAG, "Model load failed"); return;
    }
    g_load_ms       = (esp_timer_get_time() - t0) / 1000;
    g_psram_model_kb = heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024;
    ESP_LOGI(TAG, "Load time  : %lld ms", g_load_ms);
    ESP_LOGI(TAG, "PSRAM after model: %zu KB free  (model used ~%zu KB)",
             g_psram_model_kb, g_psram_runstate_kb - g_psram_model_kb);

    /* ── 4. Detect compression format and report ──────────────────────────── */
    detect_method();

    /* ── 5. Print CSV header ──────────────────────────────────────────────── */
    printf("\nCSV_HEADER:method,prompt,model_kb,load_ms,prefill_ms,"
           "gen_ms_tok,tok_per_sec,tokens_gen,"
           "psram_total_kb,psram_used_kb,psram_free_kb,psram_lwm_kb,"
           "sram_total_kb,sram_free_kb\n\n");

    /* ── 6. Benchmark — three prompts, fixed seed for reproducibility ─────── */
    g_rng = 0xCAFEBABEu;
    run_benchmark("once_upon", PROMPT_ONCE_UPON, PROMPT_ONCE_UPON_LEN);
    run_benchmark("tom_dog",   PROMPT_TOM_DOG,   PROMPT_TOM_DOG_LEN);
    run_benchmark("lily",      PROMPT_LILY,       PROMPT_LILY_LEN);

    ESP_LOGI(TAG, "All benchmarks complete. Idle.");
    while (1) vTaskDelay(pdMS_TO_TICKS(30000));
}
