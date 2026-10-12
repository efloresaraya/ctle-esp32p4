#!/usr/bin/env python3
"""
capture_energy.py — synchronous capture of the ESP32-P4 inference log and the
ESP32-S3 power log, with host timestamps accurate enough to align them.

Two serial ports are polled in one loop. Every line is stamped the moment its
terminating newline becomes visible to the host, not when a fixed-size read
happens to return: pyserial's read(n) blocks until n bytes OR the timeout, so
reading with a 1 s timeout batches up to a second of traffic and back-dates
every line in the batch. Polling `in_waiting` instead keeps the stamp within a
few milliseconds of arrival, which is what makes the two streams comparable.

Windowing follows the MARK lines the P4 prints around its generation loop:

    MARK,gen_start,<method>,<prompt>
    MARK,gen_end,<method>,<prompt>,<microseconds>

Watchdog masking
----------------
The generation loop runs for ~65 s without yielding, so the task watchdog fires
about every 5 s and dumps ~18 lines of registers over the UART. That traffic is
inside the measured window and its volume scales with how long the run takes,
so a slow format accumulates more of it than a fast one -- exactly the
speed-correlated bias the MARK design was meant to avoid. Rather than disable
the watchdog (which would change the throughput numbers already reported), the
contaminated intervals are masked out of the power average here: any sample
within MASK_BEFORE_MS..MASK_AFTER_MS of a task_wdt line is dropped.

Usage
-----
    python capture_energy.py --p4 /dev/cu.usbmodemXXXX --s3 /dev/cu.usbmodemYYYY
    python capture_energy.py --analyze runs/2026-08-16T14-30-00
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import serial
except ImportError:
    sys.exit("pyserial is required: pip install pyserial")

BAUD = 115200

# A watchdog dump is ~18 lines at 115200 baud, so roughly 90 ms of UART traffic
# plus the CPU cost of formatting it. The window is padded on both sides
# because the current disturbance starts slightly before the first line is
# fully transmitted and decays after the last one.
MASK_BEFORE_MS = 150
MASK_AFTER_MS = 250

# Discard this much of each end of the generation window. The first samples
# still carry the tail of whatever preceded the run, and the last ones overlap
# the MARK line being transmitted.
EDGE_TRIM_S = 2.0


@dataclass
class Line:
    t: float          # seconds since capture start
    wall: float       # unix timestamp
    src: str          # "p4" or "s3"
    text: str


@dataclass
class Window:
    method: str
    prompt: str
    t_start: float
    t_end: float
    device_us: int
    masked: list[tuple[float, float]] = field(default_factory=list)


class Reader:
    """Line assembler that stamps each line when its newline first appears."""

    def __init__(self, port: str, name: str):
        self.ser = serial.Serial(port, BAUD, timeout=0)
        self.name = name
        self.buf = b""

    def reset_target(self) -> None:
        self.ser.setDTR(False)
        self.ser.setRTS(True)
        time.sleep(0.12)
        self.ser.setRTS(False)

    def poll(self, t0: float) -> list[Line]:
        n = self.ser.in_waiting
        if not n:
            return []
        self.buf += self.ser.read(n)
        now = time.time()
        out = []
        while b"\n" in self.buf:
            raw, self.buf = self.buf.split(b"\n", 1)
            out.append(Line(now - t0, now, self.name,
                            raw.decode("utf-8", "replace").rstrip("\r")))
        return out

    def close(self) -> None:
        self.ser.close()


def capture(p4_port: str, s3_port: str, outdir: Path, max_s: float,
            want_windows: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    readers = [Reader(p4_port, "p4")]
    if s3_port:
        readers.append(Reader(s3_port, "s3"))

    for r in readers:
        r.reset_target()

    t0 = time.time()
    raw_path = outdir / "raw.tsv"
    closed = 0
    print(f"capturing -> {raw_path}")

    with raw_path.open("w") as f:
        f.write("t_s\twall\tsrc\ttext\n")
        while time.time() - t0 < max_s:
            got = False
            for r in readers:
                for ln in r.poll(t0):
                    got = True
                    f.write(f"{ln.t:.4f}\t{ln.wall:.4f}\t{ln.src}\t{ln.text}\n")
                    if ln.text.startswith("MARK,gen_end"):
                        closed += 1
                        print(f"  window {closed} closed at t={ln.t:.2f}s")
                    elif ln.text.startswith("MARK,gen_start"):
                        print(f"  window opening at t={ln.t:.2f}s"
                              f"  ({ln.text.split(',', 2)[-1]})")
            if closed >= want_windows:
                break
            if not got:
                time.sleep(0.002)   # keep the poll tight; do not batch
            f.flush()

    for r in readers:
        r.close()
    print(f"done: {closed} window(s)")


def load(raw_path: Path) -> list[Line]:
    out = []
    with raw_path.open() as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd, None)
        for row in rd:
            if len(row) >= 4:
                out.append(Line(float(row[0]), float(row[1]), row[2],
                                "\t".join(row[3:])))
    return out


def find_windows(lines: list[Line]) -> list[Window]:
    windows, open_w = [], None
    for ln in lines:
        if ln.src != "p4":
            continue
        if ln.text.startswith("MARK,gen_start"):
            p = ln.text.split(",")
            open_w = Window(p[2], p[3], ln.t, 0.0, 0)
        elif ln.text.startswith("MARK,gen_end") and open_w:
            p = ln.text.split(",")
            open_w.t_end = ln.t
            open_w.device_us = int(p[4])
            windows.append(open_w)
            open_w = None
    # mask every watchdog burst that lands inside a window
    for w in windows:
        for ln in lines:
            if ln.src == "p4" and "task_wdt" in ln.text \
                    and w.t_start <= ln.t <= w.t_end:
                a, b = ln.t - MASK_BEFORE_MS / 1000, ln.t + MASK_AFTER_MS / 1000
                if w.masked and a <= w.masked[-1][1]:
                    w.masked[-1] = (w.masked[-1][0], max(w.masked[-1][1], b))
                else:
                    w.masked.append((a, b))
    return windows


def power_samples(lines: list[Line]) -> list[tuple[float, float, float]]:
    """(t, bus_mV, current_uA) from the S3's JSON stream."""
    out = []
    for ln in lines:
        if ln.src != "s3" or not ln.text.lstrip().startswith("{"):
            continue
        try:
            d = json.loads(ln.text)
        except json.JSONDecodeError:
            continue
        if "current_ma" in d and "bus_voltage_v" in d:
            out.append((ln.t, d["bus_voltage_v"] * 1000.0,
                        d["current_ma"] * 1000.0))
    return out


def analyze(outdir: Path) -> None:
    lines = load(outdir / "raw.tsv")
    windows = find_windows(lines)
    samples = power_samples(lines)

    if not windows:
        sys.exit("no MARK windows found in raw.tsv")

    print(f"\n{len(windows)} window(s), {len(samples)} power sample(s)\n")
    rows = []

    # Idle reference: everything outside every window, edge-trimmed.
    def in_any_window(t: float) -> bool:
        return any(w.t_start - 1 <= t <= w.t_end + 1 for w in windows)

    idle = [(v * i / 1e6) for t, v, i in samples if not in_any_window(t)]
    p_idle = sum(idle) / len(idle) if idle else float("nan")
    print(f"idle mean power: {p_idle*1000:.1f} mW  ({len(idle)} samples)")

    for w in windows:
        host_s = w.t_end - w.t_start
        dev_s = w.device_us / 1e6
        lo, hi = w.t_start + EDGE_TRIM_S, w.t_end - EDGE_TRIM_S

        kept, dropped = [], 0
        for t, v, i in samples:
            if not (lo <= t <= hi):
                continue
            if any(a <= t <= b for a, b in w.masked):
                dropped += 1
                continue
            kept.append(v * i / 1e6)          # watts

        p_act = sum(kept) / len(kept) if kept else float("nan")
        d_p = p_act - p_idle
        toks = 64
        e_tok = d_p * dev_s / toks * 1000.0   # mJ per token

        print(f"{w.method:>10s} {w.prompt:<10s} "
              f"host={host_s:6.2f}s dev={dev_s:6.2f}s drift={host_s-dev_s:+.3f}s "
              f"| kept={len(kept):4d} masked={dropped:3d} wdt={len(w.masked):2d} "
              f"| P={p_act*1000:7.1f} mW dP={d_p*1000:6.1f} mW  {e_tok:6.1f} mJ/tok")
        rows.append(dict(method=w.method, prompt=w.prompt,
                         host_s=round(host_s, 3), device_s=round(dev_s, 3),
                         samples_kept=len(kept), samples_masked=dropped,
                         wdt_bursts=len(w.masked),
                         p_idle_mW=round(p_idle * 1000, 1),
                         p_active_mW=round(p_act * 1000, 1),
                         delta_p_mW=round(d_p * 1000, 1),
                         mJ_per_token=round(e_tok, 1)))

    if rows:
        out = outdir / "energy.csv"
        with out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p4", help="serial port of the ESP32-P4")
    ap.add_argument("--s3", default="", help="serial port of the ESP32-S3 logger")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--seconds", type=float, default=400.0)
    ap.add_argument("--windows", type=int, default=3,
                    help="stop after this many completed generations")
    ap.add_argument("--analyze", type=Path, help="analyze an existing run directory")
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return
    if not args.p4:
        sys.exit("--p4 is required (or use --analyze)")

    outdir = args.out or Path("runs") / time.strftime("%Y-%m-%dT%H-%M-%S")
    capture(args.p4, args.s3, outdir, args.seconds, args.windows)
    analyze(outdir)


if __name__ == "__main__":
    main()
