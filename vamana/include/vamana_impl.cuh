#pragma once
#include "constants.cuh"
#include "graph.cuh"
#include "kernels.cuh"
#include "search.cuh"
#include "timer.h"
#include "utils.cuh"
#include "vamana.cuh"

template <typename T__>
Vamana<T__>::Vamana(std::unique_ptr<GraphT<T__>> graph_arg) {
    graph_ = std::move(graph_arg);

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
};

template <typename T__>
void Vamana<T__>::search(T__* point) {
    uint*    d_visitedSets;
    uint*    d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;
    std::cout << "[ Searching ]\n";

    // float alpha = 1.5;
    T__* d_queryVecs;
    gpuErrchk(cudaMalloc(&d_queryVecs, 1 * FreshVamana::Consts::D_g * sizeof(T__)));

    // for (uint i = 0; i < 1; i++) {
    //     T__* src = (T__*)(point);
    //     T__* dst = (T__*)(d_queryVecs + i * FreshVamana::Consts::D_g);
    //     cudaMemcpy(dst, src, FreshVamana::Consts::D_g * sizeof(T__), cudaMemcpyDeviceToDevice);
    // }

    gpuErrchk(cudaMemcpy(d_queryVecs,
                         graph_->d_graph,
                         FreshVamana::Consts::D_g * sizeof(T__),
                         cudaMemcpyDeviceToDevice));

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
    uint* d_worklist =
        greedySearch<T__>(graph_->d_graph, d_queryVecs, d_visitedSets, d_visitedSetCount, 1);
    cputimer.Stop();
    printf("greedySearch: %f sec\n", cputimer.Elapsed());

    std::vector<uint> h_worklist(1 * FreshVamana::Consts::L_g);
    cudaMemcpy(
        h_worklist.data(), d_worklist, h_worklist.size() * sizeof(uint), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1; ++i) {
        printf("Worklist[%d] = %u\n", i, h_worklist[i]);

        const size_t vecDim    = 128;
        const size_t entrySize = FreshVamana::Consts::graph_entry_bytes_g;

        std::vector<T__> h_vec(vecDim);

        size_t offset = entrySize * h_worklist[i];
        cudaMemcpy(
            h_vec.data(), graph_->d_graph + offset, vecDim * sizeof(T__), cudaMemcpyDeviceToHost);

        for (size_t j = 0; j < vecDim; ++j)
            printf("[%3zu] %f\n", j, h_vec[j]);
    }

    cudaFree(d_worklist);
}