#pragma once
#include "constants.cuh"
#include "graphT.cuh"
#include "kernels.cuh"
#include "search.cuh"
#include "timer.h"
#include "utils.cuh"
#include "vamana.cuh"

template <typename T__>
Vamana<T__>::Vamana(std::unique_ptr<GraphT<T__>> graph_arg) {
    graph_ = std::move(graph_arg);

    CPUTimer cputimer;

    cputimer.Start();
    runVamana();
    cputimer.Stop();
    printf("Vamana<T__>::Vamana: %f sec\n", cputimer.Elapsed());
};

template <typename T__>
void Vamana<T__>::insertPoints(T__* d_queryVecs, size_t num) {
    if (FreshVamana::Globals::d_graph_size + num > FreshVamana::Globals::d_graph_capacity) {
    }
}

template <typename T__>
void Vamana<T__>::deletePoints(T__* d_queryVecs, size_t num) {
    std::vector<int> h_results(num);
    CPUTimer         cputimer;

    cputimer.Start();

    findPointsInGraph(
        h_results.data(), graph_->d_graph, FreshVamana::Globals::d_graph_size, d_queryVecs, num);

    cputimer.Stop();
    // printf("findPointsInGraph(%lu points): %f sec\n", num, cputimer.Elapsed());

    for (uint i = 0; i < num; ++i) {
        // printf("Query %u found at node index: %d\n", i, h_results[i]);
        delete_list_.addNode(h_results[i]);
    }
}

template <typename T__>
[[nodiscard]] uint* Vamana<T__>::searchPoints(T__* d_queryVecs, size_t num) {
    uint*    d_visitedSets;
    uint*    d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;
    std::cout << "[ Searching ]\n";

    CPUTimer cputimer;
    CPUTimer cputimermain;
    cputimermain.Start();

    cputimer.Start();
    gpuErrchk(cudaMalloc(&d_visitedSets,
                         FreshVamana::Globals::d_graph_size *
                             FreshVamana::Consts::max_num_parents_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, FreshVamana::Globals::d_graph_size * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex,
                         FreshVamana::Globals::d_graph_size *
                             FreshVamana::Consts::reverse_index_entry_size_g * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, FreshVamana::Globals::d_graph_size * sizeof(uint)));
    cputimer.Stop();
    printf("vamanaInner mallocs: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    uint* d_worklist = greedySearch<T__>(graph_->d_graph,
                                         delete_list_.data(),
                                         delete_list_.size(),
                                         d_queryVecs,
                                         d_visitedSets,
                                         d_visitedSetCount,
                                         num);
    cputimer.Stop();
    printf("greedySearch: %f sec\n", cputimer.Elapsed());

    cputimermain.Stop();
    printf("Vamana<T__>::search: %f sec\n", cputimermain.Elapsed());
    return d_worklist;
}

// PRIVATES
template <typename T__>
void Vamana<T__>::findPointsInGraph(int*           h_results,
                                    const uint8_t* d_graph,
                                    uint           n_nodes,
                                    const dtype_g* d_query_vecs,
                                    uint           n_queries) {
    int* d_results = nullptr;
    cudaMalloc(&d_results, n_queries * sizeof(int));

    // Initialize results to -1
    cudaMemset(d_results, 0xFF, n_queries * sizeof(int));

    dim3 threads(256);
    dim3 blocks((n_nodes + threads.x - 1) / threads.x, n_queries);

    findPointsKernel<<<blocks, threads>>>(d_graph, d_query_vecs, n_nodes, n_queries, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, n_queries * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);
}

template <typename T__>
void Vamana<T__>::runVamana() {
    uint*    d_visitedSets;
    uint*    d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;

    float alpha = 1.5;
    T__*  d_queryVecs;
    gpuErrchk(cudaMalloc(
        &d_queryVecs, FreshVamana::Globals::d_graph_size * FreshVamana::Consts::D_g * sizeof(T__)));

    for (uint i = 0; i < FreshVamana::Globals::d_graph_size; i++) {
        T__* src = (T__*)(graph_->d_graph + (i)*FreshVamana::Consts::graph_entry_bytes_g);
        T__* dst = (T__*)(d_queryVecs + i * FreshVamana::Consts::D_g);
        cudaMemcpy(dst, src, FreshVamana::Consts::D_g * sizeof(T__), cudaMemcpyDeviceToDevice);
    }

    CPUTimer cputimer;

    cputimer.Start();
    gpuErrchk(cudaMalloc(&d_visitedSets,
                         FreshVamana::Globals::d_graph_size *
                             FreshVamana::Consts::max_num_parents_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, FreshVamana::Globals::d_graph_size * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex,
                         FreshVamana::Globals::d_graph_size *
                             FreshVamana::Consts::reverse_index_entry_size_g * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, FreshVamana::Globals::d_graph_size * sizeof(uint)));
    cputimer.Stop();
    printf("vamanaInner mallocs: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    uint* d_worklist = greedySearch<T__>(graph_->d_graph,
                                         delete_list_.data(),
                                         delete_list_.size(),
                                         d_queryVecs,
                                         d_visitedSets,
                                         d_visitedSetCount,
                                         FreshVamana::Globals::d_graph_size);
    gpuErrchk(cudaFree(d_worklist));

    cputimer.Stop();
    printf("greedySearch: %f sec\n", cputimer.Elapsed());

    cputimer.Start();

    computeOutNeighbors<T__>(graph_->d_graph,
                             d_queryVecs,
                             d_visitedSets,
                             d_visitedSetCount,
                             alpha,
                             d_reverseEdgeIndex,
                             0,
                             FreshVamana::Globals::d_graph_size);
    cputimer.Stop();
    printf("computeOutNeighbors: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    computeReverseEdges<T__>(graph_->d_graph, d_reverseEdgeIndex, alpha);
    cputimer.Stop();
    printf("computeReverseEdges: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    gpuErrchk(cudaFree(d_visitedSets));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
    cputimer.Stop();
}