#!/usr/bin/env bash

# make

cd build/

./bang_exact \
. \
. \
../siftfiles/sift10k_index_disk.bin \
../siftfiles/siftsmall_query.bin \
. \
. \
../siftfiles/sift10k_groundtruth.bin \
100 \
1 \
256 \
512 \
256 \
10 \
8 \
1
