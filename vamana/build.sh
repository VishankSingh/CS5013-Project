#!/bin/bash
set -e 

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if cmake --build build; then
    ./build/fresh_vamana ../data/sift10k/sift10k_randomgraph.bin . .
else
    exit 1
fi
