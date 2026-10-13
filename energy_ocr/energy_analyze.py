"""Turn the serial logs and meter OCR series of one campaign into delta-P and mJ/token.

For each format:
  idle   = meter samples inside MARK,idle_start .. MARK,idle_end
  active = meter samples inside each MARK,gen_start .. MARK,gen_end
Both windows are trimmed by EDGE_S at each end, because the meter display
lags the load step. Samples near a task-watchdog dump are masked: the dump is a
burst of UART traffic, not inference, and the display averages over it.

delta_p = mean(active) - mean(idle), per prompt, then averaged over the three.
mJ/token = delta_p * ms/token / 1000, with ms/token from the same run.

Usage: python3 energy_analyze.py <campaign_dir>
"""
import csv, glob, os, re, statistics, sys

EDGE_S = 2.0
WDT_MASK = (-0.5, 1.0)      # seconds before/after a watchdog line

def load(d, tag):
    lines = []
    for raw in open(os.path.join(d, f"{tag}_serial.log"), encoding="utf-8", errors="replace"):
        t, _, line = raw.partition("\t")
        lines.append((float(t), line.rstrip("\n")))
    ocr = []
    with open(os.path.join(d, f"{tag}_ocr.csv")) as f:
        for r in csv.DictReader(f):
            if r["ok"] == "1":
                ocr.append((float(r["t_capture"]), float(r["w"])))
    return lines, ocr

def windows(lines):
    idle, gens, wdt, ms = None, [], [], []
    open_t = {}
    for t, l in lines:
        if l.startswith("MARK,idle_start"): open_t["idle"] = t
        elif l.startswith("MARK,idle_end"):  idle = (open_t.pop("idle"), t)
        elif l.startswith("MARK,gen_start"): open_t["gen"] = t
        elif l.startswith("MARK,gen_end"):   gens.append((open_t.pop("gen"), t))
        elif "task_wdt" in l:                wdt.append(t)
        elif l.startswith("CSV,"):           ms.append(float(l.split(",")[6]))
    return idle, gens, wdt, ms

def inside(ocr, win, wdt, mask=True):
    a, b = win[0] + EDGE_S, win[1] - EDGE_S
    out = []
    for t, w in ocr:
        if not (a <= t <= b):
            continue
        if mask and any(tw + WDT_MASK[0] <= t <= tw + WDT_MASK[1] for tw in wdt):
            continue
        out.append(w)
    return out

def summarize(d, tag):
    lines, ocr = load(d, tag)
    idle, gens, wdt, ms = windows(lines)
    if not idle or not gens:
        return None
    iw = inside(ocr, idle, wdt)
    per = []
    for g in gens:
        aw = inside(ocr, g, wdt)
        if aw and iw:
            per.append((statistics.mean(aw), statistics.pstdev(aw), len(aw)))
    if not per:
        return None
    i_mean = statistics.mean(iw)
    dps = [1000 * (m - i_mean) for m, _, _ in per]
    dp = statistics.mean(dps)
    mean_ms = statistics.mean(ms) if ms else float("nan")
    return dict(tag=tag, idle_mw=1000 * i_mean, idle_sd=1000 * statistics.pstdev(iw), idle_n=len(iw),
                active_mw=1000 * statistics.mean(m for m, _, _ in per),
                active_sd=1000 * statistics.mean(s for _, s, _ in per),
                active_n=sum(n for _, _, n in per),
                dp_mw=dp, dp_spread=max(dps) - min(dps), ms_tok=mean_ms,
                mj_tok=dp * mean_ms / 1000, n_wdt=len(wdt), n_prompts=len(per))

NAMES = {"int4u": ("INT4U", 7431), "int4bw": ("INT4BW", 9285),
         "pso_k16": ("CTLE-PSO", 7433), "ctle5_k32": ("CTLE-5b", 9290),
         "pctle_k16": ("P-CTLE32", 7433)}

if __name__ == "__main__":
    d = sys.argv[1]
    tags = [t for t in NAMES
            if os.path.exists(os.path.join(d, f"{t}_serial.log"))]
    rows, pooled = [], []
    for t in tags:
        lines, ocr = load(d, t)
        idle, gens, wdt, ms = windows(lines)
        iw = inside(ocr, idle, wdt)
        pooled += iw
        act = [statistics.mean(inside(ocr, g, wdt)) for g in gens]
        rows.append((t, iw, act, statistics.mean(ms)))
    # Primary: idle window of the same run (idle drifts with temperature).
    # Sensitivity: one reference pooled over every format's idle window.
    ref = statistics.mean(pooled)
    out = os.path.join(d, "energy_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["format", "model_kb", "ms_per_token", "idle_mw", "idle_n",
                    "active_mw", "delta_p_mw", "mj_per_token",
                    "delta_p_common_mw", "mj_per_token_common"])
        for t, iw, act, ms in rows:
            a, i = statistics.mean(act), statistics.mean(iw)
            dp, dpc = 1000 * (a - i), 1000 * (a - ref)
            name, kb = NAMES[t]
            w.writerow([name, kb, f"{ms:.1f}", f"{1000*i:.2f}", len(iw),
                        f"{1000*a:.2f}", f"{dp:.2f}", f"{dp*ms/1000:.1f}",
                        f"{dpc:.2f}", f"{dpc*ms/1000:.1f}"])
            print(f"{name:9s} delta_p {dp:6.2f} mW  {dp*ms/1000:5.1f} mJ/tok   "
                  f"pooled {dpc:6.2f} mW  {dpc*ms/1000:5.1f} mJ/tok")
    print(f"pooled idle reference {1000*ref:.2f} mW over {len(pooled)} frames -> {out}")
