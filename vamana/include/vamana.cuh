#pragma once
#include "constants.cuh"
#include "graph.cuh"
#include "kernels.cuh"
#include "search.cuh"
#include "timer.h"
#include "utils.cuh"

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
class DeleteList {
   public:
    static_assert(!std::is_void<T__>::value, "DeleteList requires a concrete value type.");
    using ValueType = T__;

    DeleteList(uint initial_capacity = 1000u)
        : growth_factor_(1.3f), capacity_(initial_capacity), size_(0), d_delete_list_(nullptr) {
        if (capacity_ == 0)
            capacity_ = 1;
        gpuErrchk(cudaMalloc(&d_delete_list_, capacity_ * getPointSize()));
    }

    ~DeleteList() {
        cudaFree(d_delete_list_);
        d_delete_list_ = nullptr;
    }

    uint size() const noexcept { return size_; }
    uint capacity() const noexcept { return capacity_; }

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
        const uint new_capacity =
            static_cast<uint>(std::max<uint>(1u, static_cast<uint>(capacity_ * growth_factor_)));
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
    Vamana(std::unique_ptr<GraphT<T__>> graph_arg);

    void insertPoint(/*something*/);
    void deletePoint(/*something*/);
    void search(T__*);

   private:
    std::unique_ptr<GraphT<T__>> graph_;
    // Data structures
    //
};
// DONT REMOVE THIS
#include "vamana_impl.cuh"