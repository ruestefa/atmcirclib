#!/bin/bash

paths=("${@}")
for path in "${paths[@]}"; do
    if [ ! -f "${path}" ]; then
        echo "error: file not found at '{path}'" >&2
        exit 1
    fi
done

for path in "${paths[@]}"; do
    echo "clear ${path}"
    python -m jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "${path}" || exit 1
done
