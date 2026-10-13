"""Summarize the -Og/-O2 campaign into results/opt_levels.csv.

Reads results/rev2_logs/opt_levels/<format>__<build>_{serial.log,ocr.csv}.
Latency is the firmware's own gen_ms_tok per prompt (mean and population SD
over the three prompts). Energy uses energy_ocr/energy_analyze.py: same-run
idle reference, edges trimmed, frames near watchdog dumps masked.
same_tokens_as_Og says whether the -O2 build generated exactly the tokens of
the -Og build for all three prompts.

Usage: python3 scripts/summarize_opt_levels.py
"""
import csv, os, re, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "energy_ocr"))
from energy_analyze import summarize

LOGS = os.path.join(ROOT, "results", "rev2_logs", "opt_levels")
OUT = os.path.join(ROOT, "results", "opt_levels.csv")
FORMATS = [("int4u", "INT4U"), ("int4bw", "INT4BW"), ("int4bw_g288", "INT4BW-g288"),
           ("kmeans_k16", "CTLE-KM"), ("ga_k16", "CTLE-GA"), ("pso_k16", "CTLE-PSO"),
           ("pctle_k16", "P-CTLE32"), ("ctle5_k32", "CTLE-5b")]

HEADER = """\
# Latency and incremental energy at two compiler optimization levels.
#
# Every earlier hardware figure was measured with ESP-IDF's default
# optimization level, -Og (CONFIG_COMPILER_OPTIMIZATION_DEBUG). This campaign
# builds the same source twice, at -Og and at -O2
# (CONFIG_COMPILER_OPTIMIZATION_PERF; firmware/sdkconfig.O2); the two
# sdkconfig files differ only in the optimization level and in the FreeRTOS
# task-function wrapper, which ESP-IDF enables only in debug builds.
#
# One thermal session, 12-13 Oct 2026, ESP32-P4 rev v1.3 at 360 MHz, revised
# output-projection loops. Each model image is flashed once and both builds
# run back to back, in alternating order. Three prompts x 64 generated tokens,
# except INT4BW-g288's third prompt (44 tokens, EOS) in both builds.
# Latency includes sampling and watchdog activity and excludes prefill.
#
# delta_p_mw: board power above the idle window of the same run, read by OCR
# from an inline USB meter that resolves 0.5 mA (about 2.45 mW).
# mj_per_token = delta_p_mw * ms_per_token / 1000.
#"""

def latencies(path):
    out = []
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.partition("\t")[2]
        if line.startswith("CSV,"):
            c = line.strip().split(",")
            out.append((float(c[6]), int(c[8])))
    return out

def tokens(path):
    d = {}
    for raw in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"TOKENS\[(\w+)\]: (.*)", raw.partition("\t")[2].strip())
        if m:
            d[m.group(1)] = m.group(2).split()
    return d

def model_kb(path):
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.partition("\t")[2]
        if line.startswith("CSV,"):
            return int(line.split(",")[3])

rows = []
for tag, name in FORMATS:
    toks = {}
    for build in ("Og", "O2"):
        t = f"{tag}__{build}"
        serial = os.path.join(LOGS, f"{t}_serial.log")
        lat = latencies(serial)
        ms = [m for m, _ in lat]
        e = summarize(LOGS, t)
        toks[build] = tokens(serial)
        rows.append(dict(format=name, build=f"-{build}", model_kb=model_kb(serial),
                         ms_per_token=f"{statistics.mean(ms):.1f}",
                         ms_sd=f"{statistics.pstdev(ms):.1f}",
                         tokens=" ".join(str(n) for _, n in lat),
                         idle_mw=f"{e['idle_mw']:.2f}", idle_frames=e["idle_n"],
                         active_frames=e["active_n"],
                         delta_p_mw=f"{e['dp_mw']:.1f}",
                         mj_per_token=f"{e['mj_tok']:.1f}",
                         watchdog_dumps=open(serial, errors="replace").read().count("Task watchdog got triggered"), same_tokens_as_Og=""))
    rows[-1]["same_tokens_as_Og"] = str(toks["Og"] == toks["O2"])

with open(OUT, "w", newline="") as f:
    f.write(HEADER + "\n")
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(open(OUT).read())
