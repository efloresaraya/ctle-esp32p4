"""One run of the -Og/-O2 campaign: flash app (and SPIFFS when given), reset,
record serial and meter OCR together, stamped with the host clock.

Usage: python3 opt_levels_run.py <port> <app.bin> <spiffs.bin|-> <tag> <outdir> <camera_url>
Writes <outdir>/<tag>_serial.log, <tag>_ocr.csv, <tag>_ocr_raw.tsv (OCR text of
every frame, so the readings can be re-parsed), <tag>_flash.log
"""
import csv, os, re, subprocess, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meter_ocr import ocr_jpeg_bytes, parse

port, app, spiffs, tag, outdir, cam = sys.argv[1:7]
os.makedirs(outdir, exist_ok=True)

def flash():
    parts = ["0x10000", app] + (["0x310000", spiffs] if spiffs != "-" else [])
    need = len(parts) // 2
    for baud in ("921600", "460800"):
        r = subprocess.run(["python", "-m", "esptool", "--chip", "esp32p4", "--port", port,
                            "--baud", baud, "write-flash"] + parts,
                           capture_output=True, text=True, timeout=1200)
        out = r.stdout + r.stderr
        open(os.path.join(outdir, f"{tag}_flash.log"), "a").write(out)
        if out.count("Hash of data verified") >= need:
            return baud
    return None

baud = flash()
if not baud:
    print(f"FLASH-FAILED {tag}", flush=True); sys.exit(1)
print(f"[{tag}] flashed at {baud}", flush=True)

stop = threading.Event()
ocr_rows, raw_rows, st = [], [], {"n": 0, "ok": 0, "bad": 0, "none": 0}

def ocr_loop():
    while not stop.is_set():
        t_req = time.time()
        try:
            b = urllib.request.urlopen(cam, timeout=3).read()
        except Exception:
            st["none"] += 1; time.sleep(0.2); continue
        t_cap = (t_req + time.time()) / 2
        obs = ocr_jpeg_bytes(b); st["n"] += 1
        raw_rows.append((t_cap, " | ".join(t for t, _ in obs)))
        r = parse(obs)
        if r is None:
            st["none"] += 1; continue
        st["ok" if r["ok"] else "bad"] += 1
        ocr_rows.append((t_cap, r["v"], r["a"], r["w"], r["residual"], int(r["ok"])))

import serial
s = serial.Serial(port, 115200, timeout=1)
th = threading.Thread(target=ocr_loop, daemon=True); th.start()
time.sleep(1.0)
s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False

t0, n_csv, elf, method = time.time(), 0, None, None
with open(os.path.join(outdir, f"{tag}_serial.log"), "w", encoding="utf-8", errors="replace") as f:
    while time.time() - t0 < 600:
        line = s.readline().decode("utf-8", "replace")
        if not line:
            continue
        now = time.time()
        f.write(f"{now:.3f}\t{line}"); f.flush()
        m = re.search(r"ELF file SHA256:\s*(\w+)", line)
        if m: elf = m.group(1)
        m = re.search(r"Detected method : (\w+)\s+\(tok_emb tag=\d+, model=(\d+) B\)", line)
        if m: method = f"{m.group(1)} {m.group(2)}B"
        if line.startswith("CSV,"):
            n_csv += 1
            c = line.strip().split(",")
            print(f"[{tag}] {c[2]:9s} {c[6]} ms/tok  tokens={c[8]}", flush=True)
            if n_csv >= 3:
                time.sleep(3.0); break
s.close()
stop.set(); th.join(timeout=5)
with open(os.path.join(outdir, f"{tag}_ocr.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_capture", "v", "a", "w", "residual_w", "ok"]); w.writerows(ocr_rows)
with open(os.path.join(outdir, f"{tag}_ocr_raw.tsv"), "w") as f:
    f.writelines(f"{t:.3f}\t{txt}\n" for t, txt in raw_rows)
print(f"[{tag}] elf={elf} method={method} ocr n={st['n']} ok={st['ok']} rejected={st['bad']} unread={st['none']}", flush=True)
