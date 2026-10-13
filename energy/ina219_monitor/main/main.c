#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "enki.h"
#include "esp_err.h"

/* Change these values if your board uses different I2C pins. */
#ifndef INA219_SDA_GPIO
#define INA219_SDA_GPIO 6
#endif

#ifndef INA219_SCL_GPIO
#define INA219_SCL_GPIO 7
#endif

#ifndef INA219_I2C_ADDRESS
#define INA219_I2C_ADDRESS 0x40
#endif

#define INA219_I2C_FREQUENCY_HZ 100000
#define INA219_SAMPLE_PERIOD_MS 100
#define INA219_I2C_TIMEOUT_MS 100
/* 0 enables I2C and validates the INA219 through its real registers. */
#define INA219_GPIO_RELEASE_TEST 0

#define INA219_REG_CONFIG 0x00
#define INA219_REG_SHUNT_VOLTAGE 0x01
#define INA219_REG_BUS_VOLTAGE 0x02
#define INA219_REG_POWER 0x03
#define INA219_REG_CURRENT 0x04
#define INA219_REG_CALIBRATION 0x05

/*
 * Settings for the common INA219 module with a 0.1 ohm shunt:
 * - 32 V bus
 * - +/-320 mV shunt range
 * - ADC averaging over 64 samples, continuous bus and shunt conversion
 * - current LSB = 100 uA; power LSB = 2 mW
 *
 * 0x3F77 keeps BRNG and PG from 0x399F and changes only BADC and SADC from
 * 0011 (12-bit, single sample) to 1110 (64 samples).
 *
 * 64 was chosen over 128 because the noise of the mean depends on how many
 * ADC conversions fit inside the window, and that is the same either way
 * (64 samples every 68 ms = 128 every 136 ms). At 64, every reading is a
 * fresh conversion: no duplicates, the standard deviation is honest, and the
 * current step at the start of inference is resolved at twice the detail,
 * which is the independent check that the window is placed correctly.
 */
#define INA219_CONFIG_VALUE 0x3F77
#define INA219_CALIBRATION_VALUE 0x1000

typedef struct {
    float bus_voltage_v;
    float shunt_voltage_mv;
    float source_voltage_v;
    float current_ma;
    float power_mw;
    bool conversion_ready;
    bool overflow;
} ina219_measurement_t;

static i2c_master_bus_handle_t i2c_bus;
static i2c_master_dev_handle_t ina219_device;
static uint16_t ina219_last_calibration_readback;

static esp_err_t ina219_write_register(uint8_t reg, uint16_t value)
{
    const uint8_t data[] = {
        reg,
        (uint8_t)(value >> 8),
        (uint8_t)(value & 0xFF),
    };
    return i2c_master_transmit(ina219_device, data, sizeof(data),
                               INA219_I2C_TIMEOUT_MS);
}

static esp_err_t ina219_read_register(uint8_t reg, uint16_t *value)
{
    uint8_t data[2];
    esp_err_t err = i2c_master_transmit_receive(
        ina219_device, &reg, 1, data, sizeof(data), INA219_I2C_TIMEOUT_MS);
    if (err == ESP_OK) {
        *value = ((uint16_t)data[0] << 8) | data[1];
    }
    return err;
}

static esp_err_t ina219_configure(void)
{
    /*
     * i2c_master_probe() is not used: that API has had regressions in the
     * new ESP-IDF driver. A real register write and read validates both the
     * ACK and the protocol the INA219 requires.
     */
    esp_err_t err = ina219_write_register(INA219_REG_CONFIG,
                                          INA219_CONFIG_VALUE);
    if (err != ESP_OK) {
        return err;
    }
    err = ina219_write_register(INA219_REG_CALIBRATION,
                                INA219_CALIBRATION_VALUE);
    if (err != ESP_OK) {
        return err;
    }

    uint16_t calibration_readback = 0;
    err = ina219_read_register(INA219_REG_CALIBRATION,
                               &calibration_readback);
    if (err != ESP_OK) {
        return err;
    }
    ina219_last_calibration_readback = calibration_readback;
    return calibration_readback == INA219_CALIBRATION_VALUE
               ? ESP_OK
               : ESP_ERR_INVALID_RESPONSE;
}

static esp_err_t ina219_read(ina219_measurement_t *measurement)
{
    uint16_t bus_raw;
    uint16_t shunt_raw;
    uint16_t current_raw;
    uint16_t power_raw;

    esp_err_t err = ina219_read_register(INA219_REG_BUS_VOLTAGE, &bus_raw);
    if (err != ESP_OK) {
        return err;
    }
    err = ina219_read_register(INA219_REG_SHUNT_VOLTAGE, &shunt_raw);
    if (err != ESP_OK) {
        return err;
    }
    err = ina219_read_register(INA219_REG_CURRENT, &current_raw);
    if (err != ESP_OK) {
        return err;
    }
    err = ina219_read_register(INA219_REG_POWER, &power_raw);
    if (err != ESP_OK) {
        return err;
    }

    measurement->bus_voltage_v = (float)(bus_raw >> 3) * 0.004f;
    measurement->shunt_voltage_mv = (float)(int16_t)shunt_raw * 0.01f;
    measurement->source_voltage_v = measurement->bus_voltage_v +
                                    measurement->shunt_voltage_mv / 1000.0f;
    measurement->current_ma = (float)(int16_t)current_raw * 0.1f;
    measurement->power_mw = (float)power_raw * 2.0f;
    measurement->conversion_ready = (bus_raw & 0x0002) != 0;
    measurement->overflow = (bus_raw & 0x0001) != 0;
    return ESP_OK;
}

static void print_measurement_json(uint32_t sample,
                                   const ina219_measurement_t *measurement)
{
    printf("{\"sample\":%" PRIu32
           ",\"bus_voltage_v\":%.3f"
           ",\"shunt_voltage_mv\":%.2f"
           ",\"source_voltage_v\":%.3f"
           ",\"current_ma\":%.1f"
           ",\"power_mw\":%.1f"
           ",\"conversion_ready\":%s"
           ",\"overflow\":%s}\n",
           sample,
           measurement->bus_voltage_v,
           measurement->shunt_voltage_mv,
           measurement->source_voltage_v,
           measurement->current_ma,
           measurement->power_mw,
           measurement->conversion_ready ? "true" : "false",
           measurement->overflow ? "true" : "false");
    fflush(stdout);
}

void app_main(void)
{
    const gpio_config_t diagnostic_gpio_config = {
        .pin_bit_mask = (1ULL << INA219_SDA_GPIO) |
                        (1ULL << INA219_SCL_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&diagnostic_gpio_config));
    enki_delay_ms(10);
    const int pre_i2c_sda_level = gpio_get_level(INA219_SDA_GPIO);
    const int pre_i2c_scl_level = gpio_get_level(INA219_SCL_GPIO);

#if INA219_GPIO_RELEASE_TEST
    printf("{\"status\":\"gpio_release_test\","
           "\"instruction\":\"measure_GPIO8_and_GPIO9_to_GND\"}\n");
    while (true) {
        printf("{\"sda_gpio4\":%d,\"scl_gpio5\":%d,"
               "\"control_gpio6\":%d,\"control_gpio7\":%d}\n",
               gpio_get_level(INA219_SDA_GPIO),
               gpio_get_level(INA219_SCL_GPIO),
               gpio_get_level(DIAGNOSTIC_CONTROL_GPIO_A),
               gpio_get_level(DIAGNOSTIC_CONTROL_GPIO_B));
        fflush(stdout);
        enki_delay_ms(1000);
    }
#endif

    /* gpio_config() claims the outputs in IDF 6.1; release them for I2C. */
    ESP_ERROR_CHECK(gpio_reset_pin(INA219_SDA_GPIO));
    ESP_ERROR_CHECK(gpio_reset_pin(INA219_SCL_GPIO));

    const i2c_master_bus_config_t bus_config = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = INA219_SDA_GPIO,
        .scl_io_num = INA219_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_config, &i2c_bus));

    const i2c_device_config_t device_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = INA219_I2C_ADDRESS,
        .scl_speed_hz = INA219_I2C_FREQUENCY_HZ,
    };
    ESP_ERROR_CHECK(i2c_master_bus_add_device(i2c_bus, &device_config,
                                               &ina219_device));

    printf("{\"status\":\"starting\",\"sensor\":\"INA219\","
           "\"address\":\"0x%02X\",\"sda_gpio\":%d,\"scl_gpio\":%d,"
           "\"pre_i2c_sda_released\":%d,\"pre_i2c_scl_released\":%d}\n",
           INA219_I2C_ADDRESS, INA219_SDA_GPIO, INA219_SCL_GPIO,
           pre_i2c_sda_level, pre_i2c_scl_level);

    esp_err_t init_err;
    while ((init_err = ina219_configure()) != ESP_OK) {
        printf("{\"error\":\"INA219_not_found\","
               "\"esp_err\":\"%s\","
               "\"calibration_readback\":\"0x%04X\","
               "\"sda_level\":%d,\"scl_level\":%d,"
               "\"hint\":\"check_power_address_and_i2c_wiring\"}\n",
               esp_err_to_name(init_err),
               ina219_last_calibration_readback,
               gpio_get_level(INA219_SDA_GPIO),
               gpio_get_level(INA219_SCL_GPIO));
        i2c_master_bus_reset(i2c_bus);
        fflush(stdout);
        enki_delay_ms(2000);
    }

    printf("{\"status\":\"ready\",\"sample_period_ms\":%d}\n",
           INA219_SAMPLE_PERIOD_MS);

    uint32_t sample = 0;
    while (true) {
        ina219_measurement_t measurement;
        esp_err_t err = ina219_read(&measurement);
        if (err == ESP_OK) {
            print_measurement_json(++sample, &measurement);
        } else {
            printf("{\"error\":\"read_failed\",\"esp_err\":\"%s\"}\n",
                   esp_err_to_name(err));
            fflush(stdout);
        }
        enki_delay_ms(INA219_SAMPLE_PERIOD_MS);
    }
}
