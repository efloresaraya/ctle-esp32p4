#!/bin/bash
# Convenience wrapper — sets IDF_PATH and runs enki build
# Usage: ./build.sh [flash]
set -e
export IDF_PATH="/Users/emilioylore/enki_ESP32/esp-idf"
if [ "$1" = "flash" ]; then
    IDF_PATH="$IDF_PATH" python3 -m enki.cli flash
else
    IDF_PATH="$IDF_PATH" python3 -m enki.cli build
fi
