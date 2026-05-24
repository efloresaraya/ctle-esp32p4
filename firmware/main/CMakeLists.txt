idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES enki_runtime spiffs esp_timer heap log
)
