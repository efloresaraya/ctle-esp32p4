idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES enki_runtime spiffs esp_timer heap log
)

# Output-projection loop structure, for the controlled comparison requested in
# review round 2. Default (1) equalizes the loops across formats; build with
#   idf.py -DLMHEAD_EQUALIZED=0 build
# to restore the original per-format loops.
if(DEFINED LMHEAD_EQUALIZED)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
                               LMHEAD_EQUALIZED=${LMHEAD_EQUALIZED})
endif()
