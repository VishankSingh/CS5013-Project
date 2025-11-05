#include "graphT.cuh"
#include "utils.cuh"
#include "vamana.cuh"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

__global__ void pp() {
    printf("\n\n%d\n\n", FreshVamana::Globals::d_graph_size);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <graph> <basepoints> <output>\n", argv[0]);
        return 1;
    }

    // we have the graph_struct with all data. we just need to implement insert/delete methods.
    // add workfloat handler here;

    std::string random_graph_bin_path = argv[1];

    std::unique_ptr<GraphT<dtype_g>> graph = initGraph<dtype_g>(random_graph_bin_path);
    // std::cout << graph->h_graph_capacity << " " << graph->h_graph_size << '\n';
    std::cout << FreshVamana::Globals::d_graph_capacity << " " << FreshVamana::Globals::d_graph_size
              << '\n';

    Vamana<dtype_g> index(std::move(graph));

    // ADDING TEST QUERIES
    float* d_queryVecs;
    uint   num_vecs = 1;
    gpuErrchk(cudaMalloc(&d_queryVecs, num_vecs * FreshVamana::Consts::D_g * sizeof(float)));

    for (uint i = 0; i < num_vecs; i++) {
        float* src = (float*)(index.graph_->d_graph + (i)*FreshVamana::Consts::graph_entry_bytes_g);
        float* dst = (float*)(d_queryVecs + i * FreshVamana::Consts::D_g);
        cudaMemcpy(dst, src, FreshVamana::Consts::D_g * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // SEARCH TEST
    uint* d_worklist = index.search(d_queryVecs, num_vecs);

    std::vector<uint> h_worklist(num_vecs * FreshVamana::Consts::L_g);
    cudaMemcpy(
        h_worklist.data(), d_worklist, h_worklist.size() * sizeof(uint), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 0; ++i) {
    //     printf("Worklist[%d] = %u\n", i, h_worklist[i]);

    //     const size_t vecDim    = 128;
    //     const size_t entrySize = FreshVamana::Consts::graph_entry_bytes_g;

    //     std::vector<float> h_vec(vecDim);

    //     size_t offset = entrySize * hcl_worklist[i];
    //     cudaMemcpy(h_vec.data(),
    //                index.graph_->d_graph + offset,
    //                vecDim * sizeof(float),
    //                cudaMemcpyDeviceToHost);

    //     for (size_t j = 0; j < vecDim; ++j)
    //         printf("[%3zu] %f\n", j, h_vec[j]);
    // }

    // DELETE TEST
    index.deletePoint(d_queryVecs, num_vecs);

    std::cout << isNodeInDeleteList(index.delete_list_.data(), index.delete_list_.size(), 1)
              << "\n";

    // pp<<<1, 1>>>();

    return 0;
}
