idf_component_register(
    SRCS
        "src/enki_gpio.c"
        "src/enki_time.c"
        "src/enki_log.c"
        "src/enki_task.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_driver_gpio freertos log
)
