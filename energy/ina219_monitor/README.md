# INA219 monitor on ESP32-S3 with Enki

This project measures voltage, current and power with an INA219 and emits one
JSON line per second at 115200 baud.

## Wiring

| INA219 | ESP32-S3 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SDA | GPIO6 |
| SCL | GPIO7 |
| VIN+ | Supply positive |
| VIN- | Load positive |

The supply negative, the load negative and the ESP32 GND must share a common
reference. The INA219 goes in series on the positive side; do not connect VIN+
or VIN- directly to a GPIO.

GPIO 6 and 7 are the defaults. If your ESP32-S3 board reserves those pins,
change `INA219_SDA_GPIO` and `INA219_SCL_GPIO` at the top of `main/main.c`. The
default I2C address is `0x40` (A0 and A1 unbridged).

The bundled calibration matches the common module with an `R100` shunt
resistor (0.1 ohm): roughly +/-3.2 A range and a bus up to 26 V, which is the
INA219's specified limit. A different shunt value requires recomputing
`INA219_CALIBRATION_VALUE`.

## Build, flash and watch the measurements

From a terminal:

```bash
. /path/to/esp-idf/export.sh
cd energy/ina219_monitor
idf.py set-target esp32s3
idf.py -p /dev/tty.usbmodemXXXX flash monitor
```

The `enki_runtime` component ships inside the project, so no external tooling
is needed: ESP-IDF is enough. The version used here was v6.1.

Example output:

```json
{"sample":1,"bus_voltage_v":4.980,"shunt_voltage_mv":12.40,"source_voltage_v":4.992,"current_ma":124.0,"power_mw":618.0,"conversion_ready":true,"overflow":false}
```

- `bus_voltage_v`: voltage at VIN- (load side) with respect to GND.
- `source_voltage_v`: estimated voltage at VIN+ with respect to GND.
- `current_ma`: positive current from VIN+ towards VIN-.
- `overflow`: `true` means the measurement exceeded the configured range.
