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

__global__ void pqv(float* dd) {
    printf("\n\n");

    for (int i = 0; i < 128; i++) {
        printf("%f ", dd[i]);
    }
    printf("\n\n");
}

__global__ void pqpinc(float* dd) {
    dd[0] += 1.0f;
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
    dtype_g* d_queryVecs;
    uint     num_vecs = 10;
    gpuErrchk(cudaMalloc(&d_queryVecs, num_vecs * FreshVamana::Consts::D_g * sizeof(dtype_g)));

    for (uint i = 0; i < num_vecs; i++) {
        dtype_g* src =
            (dtype_g*)(index.graph_->d_graph + (i)*FreshVamana::Consts::graph_entry_bytes_g);
        dtype_g* dst = (dtype_g*)(d_queryVecs + i * FreshVamana::Consts::D_g);
        cudaMemcpy(dst, src, FreshVamana::Consts::D_g * sizeof(dtype_g), cudaMemcpyDeviceToDevice);
    }

    // SEARCH TEST
    std::cout << '\n';
    uint* d_worklist = index.searchPoints(d_queryVecs, 1);

    std::vector<uint> h_worklist(1 * FreshVamana::Consts::L_g);
    cudaMemcpy(
        h_worklist.data(), d_worklist, h_worklist.size() * sizeof(uint), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; ++i) {
        printf("Worklist[%d] = %u\n", i, h_worklist[i]);

        const size_t vecDim    = 128;
        const size_t entrySize = FreshVamana::Consts::graph_entry_bytes_g;

        std::vector<dtype_g> h_vec(vecDim);

        size_t offset = entrySize * h_worklist[i];
        cudaMemcpy(h_vec.data(),
                   index.graph_->d_graph + offset,
                   vecDim * sizeof(dtype_g),
                   cudaMemcpyDeviceToHost);

        // for (size_t j = 0; j < vecDim; ++j)
        //     printf("[%3zu] %f\n", j, h_vec[j]);
    }

    // pqv<<<1, 1>>>(d_queryVecs);

    pqpinc<<<1, 1>>>(d_queryVecs);

    // pqv<<<1, 1>>>(d_queryVecs);

    std::cout << '\n';
    index.insertPoints(d_queryVecs, 1);

    cudaDeviceSynchronize();

    std::cout << '\n';
    uint* d_worklist4 = index.searchPoints(d_queryVecs, 1);

    std::vector<uint> h_worklist4(1 * FreshVamana::Consts::L_g);
    cudaMemcpy(
        h_worklist4.data(), d_worklist4, h_worklist4.size() * sizeof(uint), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; ++i) {
        printf("Worklist[%d] = %u\n", i, h_worklist4[i]);

        const size_t vecDim    = 128;
        const size_t entrySize = FreshVamana::Consts::graph_entry_bytes_g;

        std::vector<dtype_g> h_vec(vecDim);

        size_t offset = entrySize * h_worklist4[i];
        cudaMemcpy(h_vec.data(),
                   index.graph_->d_graph + offset,
                   vecDim * sizeof(dtype_g),
                   cudaMemcpyDeviceToHost);

        // for (size_t j = 0; j < vecDim; ++j)
        //     printf("[%3zu] %f\n", j, h_vec[j]);
    }

    // // DELETE TEST
    // std::cout << '\n';

    // index.deletePoints(
    //     (dtype_g*)(index.graph_->d_graph + (2) * FreshVamana::Consts::graph_entry_bytes_g), 1);

    // // std::cout << isNodeInDeleteList(index.delete_list_.data(), index.delete_list_.size(), 2)
    // //           << "\n";

    // std::cout << '\n';

    // uint* d_worklist2 = index.searchPoints(d_queryVecs, 1);

    // std::vector<uint> h_worklist2(1 * FreshVamana::Consts::L_g);
    // cudaMemcpy(
    //     h_worklist2.data(), d_worklist2, h_worklist2.size() * sizeof(uint),
    //     cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 5; ++i) {
    //     printf("Worklist[%d] = %u\n", i, h_worklist2[i]);

    //     const size_t vecDim    = 128;
    //     const size_t entrySize = FreshVamana::Consts::graph_entry_bytes_g;

    //     std::vector<dtype_g> h_vec(vecDim);

    //     size_t offset = entrySize * h_worklist2[i];
    //     cudaMemcpy(h_vec.data(),
    //                index.graph_->d_graph + offset,
    //                vecDim * sizeof(dtype_g),
    //                cudaMemcpyDeviceToHost);

    //     // for (size_t j = 0; j < vecDim; ++j)
    //     //     printf("[%3zu] %f\n", j, h_vec[j]);
    // }

    // index.deletePoints(
    //     (dtype_g*)(index.graph_->d_graph + (6) * FreshVamana::Consts::graph_entry_bytes_g), 1);

    // for (int i = 7; i < 4400; i++) {
    //     index.deletePoints(
    //         (dtype_g*)(index.graph_->d_graph + (i)*FreshVamana::Consts::graph_entry_bytes_g), 1);
    // }

    // index.deletePoints(
    //     (dtype_g*)(index.graph_->d_graph + (4585) * FreshVamana::Consts::graph_entry_bytes_g),
    //     1);
    // index.deletePoints(
    //     (dtype_g*)(index.graph_->d_graph + (9256) * FreshVamana::Consts::graph_entry_bytes_g),
    //     1);

    // std::cout << '\n';

    // uint* d_worklist3 = index.searchPoints(d_queryVecs, 1);

    // std::vector<uint> h_worklist3(1 * FreshVamana::Consts::L_g);
    // cudaMemcpy(
    //     h_worklist3.data(), d_worklist3, h_worklist3.size() * sizeof(uint),
    //     cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 5; ++i) {
    //     printf("Worklist[%d] = %u\n", i, h_worklist3[i]);

    //     const size_t vecDim    = 128;
    //     const size_t entrySize = FreshVamana::Consts::graph_entry_bytes_g;

    //     std::vector<dtype_g> h_vec(vecDim);

    //     size_t offset = entrySize * h_worklist3[i];
    //     cudaMemcpy(h_vec.data(),
    //                index.graph_->d_graph + offset,
    //                vecDim * sizeof(dtype_g),
    //                cudaMemcpyDeviceToHost);

    //     // for (size_t j = 0; j < vecDim; ++j)
    //     //     printf("[%3zu] %f\n", j, h_vec[j]);
    // }

    // // pp<<<1, 1>>>();

    return 0;
}
