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

if __name__ == "__main__":
    d = sys.argv[1]
    tags = sorted({os.path.basename(p).rsplit("_serial", 1)[0]
                   for p in glob.glob(os.path.join(d, "*_serial.log"))})
    rows = [r for r in (summarize(d, t) for t in tags) if r]
    print(f"{'formato':10s} {'reposo mW':>16s} {'activo mW':>16s} {'ΔP mW':>12s} {'ms/tok':>8s} {'mJ/tok':>7s}")
    for r in rows:
        print(f"{r['tag']:10s} {r['idle_mw']:7.2f}±{r['idle_sd']:4.2f} n={r['idle_n']:<3d} "
              f"{r['active_mw']:7.2f}±{r['active_sd']:4.2f} n={r['active_n']:<4d} "
              f"{r['dp_mw']:6.2f} (±{r['dp_spread']/2:.2f}) {r['ms_tok']:8.1f} {r['mj_tok']:7.1f}")
    with open(os.path.join(d, "energy_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["tag"])
        w.writeheader(); w.writerows(rows)
