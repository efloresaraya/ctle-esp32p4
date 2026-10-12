#!/bin/zsh
# All five formats in one thermal session, power never removed: only the SPIFFS
# model image changes between runs, so the idle reference is comparable.
set -e
PORT=/dev/cu.usbmodem5ABA0764091
CAM=http://192.168.1.137:8080/shot.jpg
B=/Users/emilioylore/CTLE_v1_v2/bench_rev2
OUT=$B/runs/energy_$(date +%Y%m%d_%H%M)
source /Users/emilioylore/enki_ESP32/esp-idf/export.sh >/dev/null 2>&1
echo "════ salida: $OUT"
echo "════ grabando firmware igualado con ventana de reposo"
python -m esptool --chip esp32p4 --port $PORT --baud 921600 \
  write-flash 0x10000 $B/fw_equalized_idle.bin 2>&1 | grep -q "Hash of data verified" \
  && echo "     ok" || { echo "     FALLO al grabar el firmware"; exit 1; }
for m in int4u int4bw pso_k16 ctle5_k32 pctle_k16; do
  /opt/anaconda3/bin/python3 $B/energy_run.py $PORT $B/spiffs/$m.bin $m $OUT $CAM
done
echo "════ campaña de energía completa"
