#!/bin/sh

rm -f output.txt

for L in 40 #10 20 30 40 60 80 120 160
do
    ./compile.sh SIFT10K "$L" 64

    
    ./build/bang_exact \
        /mnt/karthik_hdd_4tb/sift100m128/DiskANNsift100m64_pq_pivots.bin \
        /mnt/karthik_hdd_4tb/sift100m128/DiskANNsift100m64_pq_compressed.bin \
        ./sift10k_randomgraph.bin \
        ./siftsmall_query.bin \
        /mnt/karthik_hdd_4tb/sift100m128/DiskANNsift100m64_pq_pivots.bin_chunk_offsets.bin \
        /mnt/karthik_hdd_4tb/sift100m128/DiskANNsift100m64_pq_pivots.bin_centroid.bin \
        ./sift10k_groundtruth.bin \
        10000 1 256 512 256 10 64 0 << EOM >> output.txt
y
y
y
y
y
EOM

done
