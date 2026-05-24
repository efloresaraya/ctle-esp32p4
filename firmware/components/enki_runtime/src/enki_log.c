#include "enki.h"

#include "esp_log.h"

void enki_log_info(const char *tag, const char *message)
{
    ESP_LOGI(tag, "%s", message);
}
