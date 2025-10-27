

cd vamana

make

./bin/vamana ../data/sift10k/sift10k_randomgraph.bin . ./build/vamana.out

python3 scripts/bang-preprocess.py build/vamana.out test_data/test_sift10k_index

cd ../bang_exact/build

./bang_exact \
. \
. \
../../vamana/test_data/test_sift10k_index_disk.bin \
../../data/sift10k/siftsmall_query.bin \
. \
. \
../../data/sift10k/sift10k_groundtruth.bin \
100 \
1 \
256 \
512 \
256 \
10 \
8 \
1