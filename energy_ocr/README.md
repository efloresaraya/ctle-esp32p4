# Energy measurement by OCR of an inline USB meter

How the `delta_p_mW` and `mJ_per_token` values in
`results/energy_controlled.csv` were obtained.

The ESP32-P4 is powered through an inline USB meter (KWS-X1) that passes the
data lines through, so the serial console keeps working on the same cable. The
meter has no data interface: it only shows voltage, current and power on its
display. These scripts read that display with a camera and the macOS Vision
OCR, about three frames per second, while the host logs the firmware's serial
output. Both are stamped with the host clock, so the firmware's `MARK` lines
delimit the idle and generation windows directly in the meter series.

## Why a frame can be trusted

Each frame carries three numbers, and the meter computes power from the other
two. A frame is kept only if the displayed `V × A` equals the displayed `W`
within 0.2 mW. A misread digit in the current moves the product by 0.5 mW or
more, and a frame captured while the display is half-refreshed does not
satisfy the identity either, so both are rejected. The roles of the three
numbers are assigned by the same identity, not by their unit letters or
on-screen position, because the OCR misreads `W` as `v` when the display is
rotated relative to the camera. In the first session 3,202 of 3,465 frames
(92 %) were kept; in the `-Og`/`-O2` session, 8,064 of 8,744 (92 %).

In the `-Og`/`-O2` session the OCR at first kept only about a third of the
frames: it often read the large voltage digits with a comma or without their
decimal point. The parser now accepts both (a five-digit voltage can only mean
d.dddd near a 5 V bus); the `V × A = W` check still decides every frame. That
session was restarted from the beginning with the new parser, so all of its
runs use it, and `opt_levels_run.py` stores the OCR text of every frame
(`*_ocr_raw.tsv`) so that the readings can be parsed again.

## What the meter can and cannot resolve

The display shows 0.1 mW, but every current value read was a multiple of
0.5 mA, about 2.45 mW at the 4.9 V bus. Each reported figure is therefore an
average of hundreds of frames, and idle readings, which carry little load noise,
sit on one or two adjacent codes.

Board power at idle is about 86 % of the total and drifted upward by 2.6 mW over
the first session (2.2 mW over the `-Og`/`-O2` session) as the board warmed. `delta_p` is therefore taken against the idle
window of the same run. Pooling the five idle windows instead changes the
CTLE-5b versus INT4BW energy difference from +16.6 % to +18.4 % and changes no
ordering; `energy_analyze.py` reports both.

## Running it

Requirements: macOS with `pip install pyobjc-framework-Vision pyserial`; a
phone camera serving single JPEG frames over HTTP (the Android app IP Webcam
serves them at `/shot.jpg`), fixed above the meter with no glare; ESP-IDF
sourced, for esptool; the firmware built with its default, equalized loops.

```
. $IDF_PATH/export.sh
PORT=/dev/cu.usbmodemXXXX CAM=http://<phone>:8080/shot.jpg \
SPIFFS=<dir with int4u.bin int4bw.bin pso_k16.bin ctle5_k32.bin pctle_k16.bin> \
FW=firmware/build/firmware.bin PY=<python with Vision> \
zsh energy_ocr/energy_campaign.sh
```

Build each SPIFFS image from the matching `weights/stories15M_<format>.bin`,
renamed to `model.bin`, with
`python $IDF_PATH/components/spiffs/spiffsgen.py 0xCF0000 <dir> <format>.bin`.

Before trusting a run, check that the model size the firmware reports matches
the format flashed: a failed SPIFFS write leaves the previous model in place and
the run otherwise looks normal. `energy_run.py` prints it for each format.

Close anything else that holds the serial port. A second process reading it
makes esptool fail with "The chip stopped responding" or "Invalid head of
packet", which looks like a baud-rate or wiring fault but is not.

| File | Role |
|---|---|
| `meter_ocr.py` | OCR of one frame and the `V × A = W` check |
| `energy_run.py` | One format: flash, reset, record serial and meter together |
| `energy_analyze.py` | Windows, trimming, watchdog masking, delta-P and mJ/token |
| `energy_campaign.sh` | The five formats in one thermal session (first session) |
| `opt_levels_run.py` | One run: flash app (and model), reset, record serial, meter and OCR text |
| `opt_levels_campaign.sh` | Eight formats × two builds (`-Og`, `-O2`) in one thermal session |
