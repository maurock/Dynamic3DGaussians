#!/bin/bash
DIRECTORY="data/refnerf-blend"
mkdir -p $DIRECTORY/obj
BLEND_FILES=$(find "$DIRECTORY" -type f -name "*.blend")
for BLEND_FILE in $BLEND_FILES; do
    blender -b $BLEND_FILE -P data_making/_extract_obj_from_blend.py
done