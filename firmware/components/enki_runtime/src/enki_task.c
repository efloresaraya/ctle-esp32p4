#include "enki.h"

BaseType_t enki_task_create(
    TaskFunction_t task_fn,
    const char *name,
    uint32_t stack_depth,
    void *params,
    UBaseType_t priority
)
{
    return xTaskCreate(task_fn, name, stack_depth, params, priority, NULL);
}
