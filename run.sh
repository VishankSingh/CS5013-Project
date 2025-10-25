#!/usr/bin/env bash

# make

cd build/

./bang_exact \
[sift10k_index_pq_pivots.bin] \
[sift10k_index_pq_compressed.bin] \
sift10k_index_disk.bin \
siftsmall_query.bin \
[MISSING_CHUNK_OFFSETS_FILE.bin] \
[MISSING_CENTROID_FILE.bin] \
sift10k_groundtruth.bin \
100 \
1 \
256 \
512 \
256 \
10 \
8 \
1
