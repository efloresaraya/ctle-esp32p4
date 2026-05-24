#pragma once

#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifdef __cplusplus
extern "C" {
#endif

void enki_delay_ms(uint32_t ms);
void enki_gpio_output(int pin);
void enki_gpio_write(int pin, int value);
void enki_gpio_toggle(int pin);
void enki_log_info(const char *tag, const char *message);

BaseType_t enki_task_create(
    TaskFunction_t task_fn,
    const char *name,
    uint32_t stack_depth,
    void *params,
    UBaseType_t priority
);

#ifdef __cplusplus
}
#endif
