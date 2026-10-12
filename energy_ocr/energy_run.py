"""One energy run: flash a model image, reset, and record serial and meter together.

The serial log and the OCR readings of the inline USB meter are both stamped
with the host clock (time.time()), so the MARK lines the firmware prints
delimit the idle and generation windows directly in the meter series.

Usage: python3 energy_run.py <port> <spiffs.bin> <tag> <outdir> <camera_url>
Writes <outdir>/<tag>_serial.log  (host_t <TAB> line)
       <outdir>/<tag>_ocr.csv     (t_capture,v,a,w,residual_w,ok)
"""
import csv, os, re, subprocess, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meter_ocr import ocr_jpeg_bytes, parse

port, img, tag, outdir, cam = sys.argv[1:6]
os.makedirs(outdir, exist_ok=True)

def flash():
    for baud in ("921600", "460800"):
        r = subprocess.run(["python", "-m", "esptool", "--chip", "esp32p4", "--port", port,
                            "--baud", baud, "write-flash", "0x310000", img],
                           capture_output=True, text=True, timeout=900)
        if "Hash of data verified" in r.stdout + r.stderr:
            return baud
    return None

baud = flash()
if not baud:
    print(f"FLASH-FAILED {tag}", flush=True); sys.exit(1)
print(f"[{tag}] flashed at {baud}", flush=True)

stop = threading.Event()
ocr_rows, ocr_stats = [], {"n": 0, "ok": 0, "bad": 0, "none": 0}

def ocr_loop():
    while not stop.is_set():
        t_req = time.time()
        try:
            b = urllib.request.urlopen(cam, timeout=3).read()
        except Exception:
            ocr_stats["none"] += 1; time.sleep(0.2); continue
        t_cap = (t_req + time.time()) / 2          # frame fetched mid-request
        r = parse(ocr_jpeg_bytes(b)); ocr_stats["n"] += 1
        if r is None:
            ocr_stats["none"] += 1; continue
        ocr_stats["ok" if r["ok"] else "bad"] += 1
        ocr_rows.append((t_cap, r["v"], r["a"], r["w"], r["residual"], int(r["ok"])))

import serial
s = serial.Serial(port, 115200, timeout=1)
th = threading.Thread(target=ocr_loop, daemon=True); th.start()
time.sleep(1.0)                                    # meter series starts before reset
s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False

t0, csv_rows, size_kb = time.time(), 0, None
with open(os.path.join(outdir, f"{tag}_serial.log"), "w", encoding="utf-8", errors="replace") as f:
    while time.time() - t0 < 480:
        line = s.readline().decode("utf-8", "replace")
        if not line:
            continue
        now = time.time()
        f.write(f"{now:.3f}\t{line}"); f.flush()
        m = re.search(r"Model size:\s*(\d+) KB", line)
        if m:
            size_kb = int(m.group(1))
        if line.startswith("MARK,"):
            print(f"[{tag}] {now - t0:6.1f}s {line.strip()[:60]}", flush=True)
        if line.startswith("CSV,"):
            csv_rows += 1
            c = line.strip().split(",")
            print(f"[{tag}] {c[1]} {c[2]} {c[3]}KB {c[6]} ms/tok", flush=True)
            if csv_rows >= 3:
                time.sleep(3.0)                    # a little tail after the last window
                break
s.close()
stop.set(); th.join(timeout=5)

with open(os.path.join(outdir, f"{tag}_ocr.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_capture", "v", "a", "w", "residual_w", "ok"])
    w.writerows(ocr_rows)
print(f"[{tag}] model={size_kb}KB  ocr frames={ocr_stats['n']} ok={ocr_stats['ok']} "
      f"rejected={ocr_stats['bad']} unread={ocr_stats['none']}", flush=True)
