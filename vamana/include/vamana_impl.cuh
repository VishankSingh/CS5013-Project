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
};

template <typename T__>
[[nodiscard]] uint* Vamana<T__>::search(T__* d_queryVecs, size_t num) {
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

// =======================================================================================================
