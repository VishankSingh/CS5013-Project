cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

./build/fresh_vamana ../data/sift10k/sift10k_randomgraph.bin . . 
