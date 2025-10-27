#!/bin/bash

# Helper function to print and run a command
run_cmd() {
    echo -e "\e[1;31m[ $* ]\e[0m"  # bold red
    eval "$@"                      # execute command
    echo -e ""
}

run_cmd cd vamana
run_cmd make
run_cmd ./bin/vamana ../data/sift10k/sift10k_randomgraph.bin _ ./build/vamana.out
run_cmd python3 scripts/bang-preprocess.py build/vamana.out test_data/test_sift10k_index
run_cmd cd ../bang_exact/build
run_cmd ./bang_exact _ _ ../../vamana/test_data/test_sift10k_index_disk.bin ../../data/sift10k/siftsmall_query.bin _ _ ../../data/sift10k/sift10k_groundtruth.bin 100 1 256 512 256 10 8 1
