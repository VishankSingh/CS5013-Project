#pragma once
#include "constants.cuh"
#include "graphT.cuh"
#include "kernels.cuh"
#include "search.cuh"
#include "timer.h"
#include "utils.cuh"

#include "delete_list.cuh"

#include <memory>
#include <vector>

// DONE: design this
// checkout

/*
Binary file data layout

graph is a random graph of size 10000
graph has format

struct node {
    float vec[128];
    uint degree;
    uint neighbors[degree];
} node_t;

basepoints is all the points in the graph
float basepoinst[N][128];
*/

using namespace FreshVamana::Consts;

template <typename T__>
class Vamana {
   public:
    Vamana(std::unique_ptr<GraphT<T__>> graph_arg);

    void insertPoint(/*something*/);
    void deletePoint(T__* d_queryVecs, size_t num) {
        std::vector<int> h_results(num);
        CPUTimer         cputimer;

        cputimer.Start();

        findPointsInGraph(h_results.data(),
                          graph_->d_graph,
                          FreshVamana::Globals::d_graph_size,
                          d_queryVecs,
                          num);

        cputimer.Stop();
        printf("findPointsInGraph(%lu points): %f sec\n", num, cputimer.Elapsed());

        // for (uint i = 0; i < num; ++i) {
        //     printf("Query %u found at node index: %d\n", i, h_results[i]);
        //     delete_list_.addNode(h_results[i]);
        // }
    }

    // IMPORTANT: returns a pointer to gpu buffer of size num * FreshVamana::Consts::L_g *
    // sizeof(uint)
    // FREE IT LATER
    [[nodiscard]] uint* search(T__* d_queryVecs, size_t num);

   private:
    void findPointsInGraph(int*           h_results,
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

   public:
    std::unique_ptr<GraphT<T__>> graph_;

    DeleteList delete_list_;

   private:
    // Data structures
    //
};
// DONT REMOVE THIS
#include "vamana_impl.cuh"