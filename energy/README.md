# Energy measurement

> **Status (v1.2.0).** This folder documents the energy method of the first
> revision: the meter read by eye from its display, with an optional INA219
> logger. The published v1.2.0 energy figures come from `energy_ocr/` instead,
> which reads the same meter by OCR, with the output-projection loops revised.
> The 473.5 mW idle reference and the per-format values quoted below belong to
> that earlier session and to the original loops; see `energy_ocr/README.md`
> and `results/energy_controlled.csv` for the current ones. The INA219 path was
> never used for any published number.

How the `delta_p_mW` and `mJ_per_token` columns of Table IV, recorded in
`results/energy_per_token.csv`, were obtained, and how to reproduce them.

## What was actually used

An **inline USB power meter** on the ESP32-P4's own 5 V supply cable. The meter
sits between the USB hub and the board and passes the data lines through, so the
serial console keeps working on the same cable. Readings were taken from its
display.

That is the whole instrument. An alternative automated path using an INA219
shunt monitor and a second microcontroller is described at the end, and its
firmware is included, but it is **not** how the published numbers were measured.

```
Mac ──> USB hub ──> inline USB power meter ──> ESP32-P4
```

Requirements for the meter: it must pass USB data (most do; if the P4 stops
enumerating, yours does not), and resolve about 1 mW. The differences between
formats are a few mW, so a meter quantised to 10 mW cannot see them.

## What is reported, and why it is a difference

`delta_p_mW` is board power during generation minus board power in an idle
window, both from the same run.

The differential form is not a refinement, it is what makes the measurement
work at all. Fixed board overhead — regulator, USB bridge, LEDs, PSRAM holding
the model — is **473.5 mW, about 85 % of total draw, and identical for every
format**. Reported as absolute power, all five formats read between 551 and
561 mW and nothing is distinguishable. The differential isolates the 78–88 mW
that inference actually costs.

## Windows

The firmware brackets both measurement windows with printed markers, so the two
readings come from a defined state rather than from whenever the meter happens
to be observed:

```
MARK,idle_start                          20 s, model resident, not generating
MARK,idle_end
MARK,gen_start,<method>,<prompt>          one prompt, 64 tokens
MARK,gen_end,<method>,<prompt>,<microseconds>
```

Nothing is printed between `gen_start` and `gen_end` by the application. The
task watchdog does fire during the window, roughly every 5 s, because the
generation loop runs for about a minute without yielding; that traffic is left
in place rather than disabling the watchdog, since the throughput figures in
Table IV were measured with it enabled. With a display-read measurement it is
averaged over anyway. It matters only for a sampling logger, and
`capture_energy.py` masks those intervals.

## Procedure

Connect everything **with power off**, then power up. Do not hot-plug the board
onto a live meter: inrush into its discharged input capacitance during a
partially-mated connector is a real hazard to anything else sharing the ground.

For each format:

1. Write the model image to the SPIFFS partition at `0x310000`:

   ```
   python $IDF_PATH/components/spiffs/spiffsgen.py 0xCF0000 <dir-with-model.bin> spiffs_<fmt>.bin
   python -m esptool --chip esp32p4 --port <port> write-flash 0x310000 spiffs_<fmt>.bin
   ```

   The firmware opens `/spiffs/model.bin`, so the file inside must have that
   name. It detects the format from the file's own tag; check the reported size
   against Table IV before recording anything.

2. Run the capture, which prints when to read:

   ```
   python energy/capture_energy.py --p4 <port> --windows 1 --seconds 130
   ```

3. Record the meter during the idle window and during generation.

**Measure all formats in one session, without removing power between them.**
Across separate sessions the idle reference drifted by up to 7.5 mW, presumably
thermal, while `delta_p` for a given format stayed within 1.1 mW. The
differential absorbs that drift, but differences between formats are of the same
order, so a comparison spanning sessions is not trustworthy. In the published
run the idle reference came out identical (473.5 mW) for all five formats, which
is what makes a 2.5 mW difference between them meaningful.

## Checks before trusting a number

1. The model size the firmware reports must match Table IV. A failed SPIFFS
   write leaves the previous format in place and the run looks normal.
2. `drift` printed by `capture_energy.py` is host window length minus device
   window length. Expect a few milliseconds.
3. Bus voltage should stay above about 4.6 V. Below that the hub is not keeping
   up and the board can reset mid-run.
4. `mJ_per_token` must equal `delta_p_mW` × `ms_per_token` / 1000. The CSV uses
   the Table IV latency so this multiplication verifies.

## Optional: automated logging with an INA219

`ina219_monitor/` is firmware for an ESP32-S3 that reads an INA219 shunt monitor
over I2C and emits one JSON sample per interval. `capture_energy.py` will read
that stream alongside the P4's, window it by the MARK lines, mask watchdog
intervals and compute delta_p automatically.

This path yields a time series rather than two display readings, so it gives a
standard deviation and lets the current step be checked against the markers. It
was not used for the published numbers.

Wiring, if you take this route: the INA219 goes in series on the high side of
the P4's 5 V supply (`VIN+` to the supply, `VIN-` to the board), powered from
the S3's 3V3 so the instrument is outside the shunt, with SDA on GPIO6 and SCL
on GPIO7. Only the S3's ground reaches the INA219; the P4's ground returns
through its own USB cable, and both are already common at the host. Do not add a
second ground wire between them.
