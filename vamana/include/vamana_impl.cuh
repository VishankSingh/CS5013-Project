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
    // size_t num_to_print = 128;
    // std::vector<float> h_data(num_to_print);
    // gpuErrchk(cudaMemcpy(h_data.data(), graph_->d_graph, num_to_print * sizeof(float),
    //                      cudaMemcpyDeviceToHost));
    // for (size_t i = 0; i < num_to_print; ++i) {
    //     printf("[%3zu] %f\n", i, h_data[i]);
    // }

    uint*    d_visitedSets;
    uint*    d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;

    float alpha = 1.5;
    T__*  d_queryVecs;
    gpuErrchk(cudaMalloc(&d_queryVecs,
                         FreshVamana::Consts::N_g * FreshVamana::Consts::D_g * sizeof(T__)));

    for (uint i = 0; i < FreshVamana::Consts::N_g; i++) {
        T__* src = (T__*)(graph_->d_graph + (i)*FreshVamana::Consts::graph_entry_size_g);
        T__* dst = (T__*)(d_queryVecs + i * FreshVamana::Consts::D_g);
        cudaMemcpy(dst, src, FreshVamana::Consts::D_g * sizeof(T__), cudaMemcpyDeviceToDevice);
    }

    CPUTimer cputimer;

    cputimer.Start();
    gpuErrchk(cudaMalloc(
        &d_visitedSets,
        FreshVamana::Consts::N_g * FreshVamana::Consts::max_paren_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, FreshVamana::Consts::N_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex,
                         FreshVamana::Consts::N_g *
                             FreshVamana::Consts::reverse_index_entry_size_g * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, FreshVamana::Consts::N_g * sizeof(uint)));
    cputimer.Stop();
    printf("vamanaInner mallocs: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    greedySearch<T__>(
        graph_->d_graph, d_queryVecs, d_visitedSets, d_visitedSetCount, FreshVamana::Consts::N_g);
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
                             FreshVamana::Consts::N_g);
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
void Vamana<T__>::search(T__* point) {}