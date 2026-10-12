#!/bin/zsh
# Energy campaign: all five formats in one thermal session, power never removed;
# only the SPIFFS model image changes between runs, so idle references compare.
#
# Required environment:
#   PORT     serial port of the ESP32-P4          e.g. /dev/cu.usbmodemXXXX
#   CAM      URL returning one JPEG of the meter  e.g. http://<phone>:8080/shot.jpg
#   SPIFFS   directory with <format>.bin images built by spiffsgen.py
#   FW       equalized firmware binary (build/firmware.bin)
# Optional:
#   PY       Python with pyobjc-framework-Vision and pyserial (default: python3)
#   OUT      output directory (default: runs/energy_<date>)
#
# ESP-IDF must already be sourced (`. $IDF_PATH/export.sh`) so that `python`
# provides esptool. Its Python usually lacks Vision, hence PY is separate.
set -e
: ${PORT:?set PORT} ${CAM:?set CAM} ${SPIFFS:?set SPIFFS} ${FW:?set FW}
PY=${PY:-python3}
HERE=${0:A:h}
OUT=${OUT:-runs/energy_$(date +%Y%m%d_%H%M)}
command -v esptool.py >/dev/null || python -m esptool version >/dev/null \
  || { echo "source ESP-IDF first"; exit 1; }

echo "== output: $OUT"
python -m esptool --chip esp32p4 --port $PORT --baud 921600 \
  write-flash 0x10000 $FW 2>&1 | grep -q "Hash of data verified" \
  || { echo "firmware flash failed"; exit 1; }
for m in int4u int4bw pso_k16 ctle5_k32 pctle_k16; do
  $PY $HERE/energy_run.py $PORT $SPIFFS/$m.bin $m $OUT $CAM
done
$PY $HERE/energy_analyze.py $OUT
