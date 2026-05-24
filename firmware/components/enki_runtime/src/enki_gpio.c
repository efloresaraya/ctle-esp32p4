#include "enki.h"

#include "driver/gpio.h"

void enki_gpio_output(int pin)
{
    gpio_reset_pin((gpio_num_t)pin);
    gpio_set_direction((gpio_num_t)pin, GPIO_MODE_OUTPUT);
}

void enki_gpio_write(int pin, int value)
{
    gpio_set_level((gpio_num_t)pin, value ? 1 : 0);
}

void enki_gpio_toggle(int pin)
{
    int value = gpio_get_level((gpio_num_t)pin);
    gpio_set_level((gpio_num_t)pin, value ? 0 : 1);
}
