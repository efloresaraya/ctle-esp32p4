"""Summarize the controlled latency runs into results/latency_controlled.csv.

Reads the per-prompt CSV rows the firmware prints (results/rev2_logs/{equalized,
original}/*.log) and reports, per configuration, the mean, minimum and maximum
of gen_ms_tok over the prompts in each run.

    python3 scripts/summarize_latency.py results/rev2_logs > results/latency_controlled.csv
"""
import glob, os, sys

HEADER = """# Controlled latency comparison, review round 2.
#
# 'original'  = per-format output-projection loops as published in revision 1
#               (build with -DLMHEAD_EQUALIZED=0).
# 'equalized' = revised output-projection loops use common access principles:
#               sequential byte pointer, no per-element division, scale hoisted
#               to the group (default build).
#
# int4bw_g288 uses G=288 instead of 32: one scale per 288-column row,
# three per 768-column row. The kernel source is shared, but group arithmetic
# and loop counts also shrink. This does not isolate scale traffic. Its third
# prompt ends at EOS after 44 tokens; all other runs produce 64.
#
# ESP32-P4 rev v1.3 @360 MHz, three prompts x 64 generated tokens each.
#
variant,weights,model_kb,mean_ms_per_tok,min_ms,max_ms,n_prompts"""

def rows(root):
    for variant in ("equalized", "original"):
        for path in sorted(glob.glob(os.path.join(root, variant, "*.log"))):
            ms, kb = [], None
            for line in open(path, encoding="utf-8", errors="replace"):
                if line.startswith("CSV,"):
                    c = line.strip().split(",")
                    ms.append(float(c[6])); kb = c[3]
            if ms:
                yield (variant, os.path.basename(path)[:-4], kb,
                       sum(ms) / len(ms), min(ms), max(ms), len(ms))

if __name__ == "__main__":
    print(HEADER)
    for v, w, kb, mean, lo, hi, n in rows(sys.argv[1]):
        print(f"{v},{w},{kb},{mean:.2f},{lo:.2f},{hi:.2f},{n}")
