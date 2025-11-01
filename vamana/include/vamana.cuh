#pragma once
#include "graph.cuh"
#include "utils.h"

#include <memory>

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

template <typename T__, uint D__, uint R__>
class DeleteList {
   public:
    static_assert(!std::is_void<T__>::value, "DeleteList requires a concrete value type.");
    using ValueType = T__;

    DeleteList(unsigned initial_capacity = 1000u)
        : growth_factor_(1.3f), capacity_(initial_capacity), size_(0), d_delete_list_(nullptr) {
        if (capacity_ == 0)
            capacity_ = 1;
        gpuErrchk(cudaMalloc(&d_delete_list_, capacity_ * get_point_size()));
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

        const size_t offset_bytes = static_cast<size_t>(size_) * get_point_size();
        uint8_t*     dst          = d_delete_list_ + offset_bytes;
        gpuErrchk(cudaMemcpy(dst, reinterpret_cast<const uint8_t*>(d_point), get_point_size(),
                             cudaMemcpyDeviceToDevice));
        ++size_;
    }

   private:
    [[nodiscard]] static constexpr size_t get_point_size() noexcept {
        return static_cast<size_t>(D__) * sizeof(ValueType);
    }

    void expandDeleteList() {
        const unsigned new_capacity = static_cast<unsigned>(
            std::max<unsigned>(1u, static_cast<unsigned>(capacity_ * growth_factor_)));
        uint8_t* new_buffer = nullptr;
        gpuErrchk(cudaMalloc(&new_buffer, static_cast<size_t>(new_capacity) * get_point_size()));

        if (d_delete_list_ && capacity_ > 0) {
            gpuErrchk(cudaMemcpy(new_buffer, d_delete_list_,
                                 static_cast<size_t>(capacity_) * get_point_size(),
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

template <typename T__, uint D__, uint R__>
class Vamana {
   public:
    using GraphType = Graph_t<T__, D__, R__>;
    Vamana(std::unique_ptr<GraphType> graph) : graph_(std::move(graph)) {};

    void insertPoint(/*something*/);
    void deletePoint(/*something*/);
    void search(/*something*/);

   private:
    std::unique_ptr<GraphType> graph_;
    // Data structures
    //
};