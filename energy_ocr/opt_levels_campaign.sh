#!/bin/zsh
# -Og versus -O2 for every format in the paper, in one thermal session.
#
# The two firmware builds come from the same source and differ only in the
# compiler optimization level (see README, "Optimization level"). Each model
# image is flashed once, then both builds run back to back; their order
# alternates between formats so that warm-up drift favours neither.
#
# Required environment:
#   PORT     serial port of the ESP32-P4          e.g. /dev/cu.usbmodemXXXX
#   CAM      URL returning one JPEG of the meter  e.g. http://<phone>:8080/shot.jpg
#   SPIFFS   directory with <format>.bin images built by spiffsgen.py
#   FW_OG    build directory of the -Og firmware  (contains firmware.bin)
#   FW_O2    build directory of the -O2 firmware
# Optional:
#   PY       Python with pyobjc-framework-Vision and pyserial (default: python3)
#   OUT      output directory (default: runs/opt_levels_<date>)
#
# ESP-IDF must already be sourced so that `python` provides esptool.
set -u
: ${PORT:?set PORT} ${CAM:?set CAM} ${SPIFFS:?set SPIFFS} ${FW_OG:?set FW_OG} ${FW_O2:?set FW_O2}
PY=${PY:-python3}
HERE=${0:A:h}
OUT=${OUT:-runs/opt_levels_$(date +%Y%m%d_%H%M)}
mkdir -p $OUT
python -m esptool --chip esp32p4 --port $PORT --baud 921600 write-flash \
  0x2000 $FW_OG/bootloader/bootloader.bin 0x8000 $FW_OG/partition_table/partition-table.bin \
  > $OUT/base_flash.log 2>&1 || { echo "bootloader/partition flash failed"; exit 1; }
typeset -A FW; FW=(Og $FW_OG O2 $FW_O2)
i=0
for f in int4u int4bw int4bw_g288 kmeans_k16 ga_k16 pso_k16 pctle_k16 ctle5_k32; do
  if (( i % 2 == 0 )); then order=(Og O2); else order=(O2 Og); fi
  sp=$SPIFFS/$f.bin
  for b in $order; do
    $PY $HERE/opt_levels_run.py $PORT $FW[$b]/firmware.bin $sp ${f}__$b $OUT $CAM \
      || echo "RUN FAILED ${f}__$b"
    sp=-
  done
  i=$((i+1))
done
echo "campaign done: $OUT"
