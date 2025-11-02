#pragma once
#include "constants.cuh"
#include "graph.cuh"
#include "kernels.cuh"
#include "search.cuh"
#include "timer.h"
#include "utils.cuh"

#include <memory>
#include <vector>

// TODO: design this

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
class DeleteList {
   public:
    static_assert(!std::is_void<T__>::value, "DeleteList requires a concrete value type.");
    using ValueType = T__;

    DeleteList(unsigned initial_capacity = 1000u)
        : growth_factor_(1.3f), capacity_(initial_capacity), size_(0), d_delete_list_(nullptr) {
        if (capacity_ == 0)
            capacity_ = 1;
        gpuErrchk(cudaMalloc(&d_delete_list_, capacity_ * getPointSize()));
    }

    ~DeleteList() {
        cudaFree(d_delete_list_);
        d_delete_list_ = nullptr;
    }

    unsigned size() const noexcept { return size_; }
    unsigned capacity() const noexcept { return capacity_; }

    void addPoint(const ValueType* d_point) {
        if (size_ >= capacity_)
            expandDeleteList();

        const size_t offset_bytes = static_cast<size_t>(size_) * getPointSize();
        uint8_t*     dst          = d_delete_list_ + offset_bytes;
        gpuErrchk(cudaMemcpy(dst,
                             reinterpret_cast<const uint8_t*>(d_point),
                             getPointSize(),
                             cudaMemcpyDeviceToDevice));
        ++size_;
    }

   private:
    [[nodiscard]] static constexpr size_t getPointSize() noexcept {
        return static_cast<size_t>(FreshVamana::Consts::D_g) * sizeof(ValueType);
    }

    void expandDeleteList() {
        const unsigned new_capacity = static_cast<unsigned>(
            std::max<unsigned>(1u, static_cast<unsigned>(capacity_ * growth_factor_)));
        uint8_t* new_buffer = nullptr;
        gpuErrchk(cudaMalloc(&new_buffer, static_cast<size_t>(new_capacity) * getPointSize()));

        if (d_delete_list_ && capacity_ > 0) {
            gpuErrchk(cudaMemcpy(new_buffer,
                                 d_delete_list_,
                                 static_cast<size_t>(capacity_) * getPointSize(),
                                 cudaMemcpyDeviceToDevice));
            cudaFree(d_delete_list_);
        }

        d_delete_list_ = new_buffer;
        capacity_      = new_capacity;
    }

    void clear() noexcept {
        cudaFree(d_delete_list_);
        d_delete_list_ = nullptr;
        capacity_      = 0;
        size_          = 0;
    }

   private:
    uint8_t* d_delete_list_ = nullptr;
    float    growth_factor_ = 1.3f;
    uint     capacity_      = 0;
    uint     size_          = 0;
};

template <typename T__>
class Vamana {
   public:
    using GraphType = GraphT<T__>;
    Vamana(std::unique_ptr<GraphType> graph_arg) : graph_(std::move(graph_arg)) {
        // size_t num_to_print = 128;
        // std::vector<float> h_data(num_to_print);
        // gpuErrchk(cudaMemcpy(h_data.data(), graph_->d_graph, num_to_print * sizeof(float),
        //                      cudaMemcpyDeviceToHost));
        // for (size_t i = 0; i < num_to_print; ++i) {
        //     printf("[%3zu] %f\n", i, h_data[i]);
        // }

        unsigned* d_visitedSets;
        unsigned* d_visitedSetCount;
        uint8_t*  d_reverseEdgeIndex;

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
        gpuErrchk(cudaMalloc(&d_visitedSets,
                             FreshVamana::Consts::N_g * FreshVamana::Consts::max_paren_per_query *
                                 sizeof(unsigned)));
        gpuErrchk(cudaMalloc(&d_visitedSetCount, FreshVamana::Consts::N_g * sizeof(unsigned)));
        gpuErrchk(cudaMalloc(&d_reverseEdgeIndex,
                             FreshVamana::Consts::N_g *
                                 FreshVamana::Consts::reverse_index_entry_size * sizeof(uint8_t)));
        gpuErrchk(cudaMemset(d_visitedSetCount, 0, FreshVamana::Consts::N_g * sizeof(unsigned)));
        cputimer.Stop();
        printf("vamanaInner mallocs: %f sec\n", cputimer.Elapsed());

        cputimer.Start();
        greedySearch<T__>(graph_->d_graph,
                          d_queryVecs,
                          d_visitedSets,
                          d_visitedSetCount,
                          0,
                          FreshVamana::Consts::N_g);
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

    void insertPoint(/*something*/);
    void deletePoint(/*something*/);
    void search(/*something*/);

   private:
    std::unique_ptr<GraphType> graph_;
    // Data structures
    //
};