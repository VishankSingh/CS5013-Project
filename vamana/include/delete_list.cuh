#pragma once
#include "constants.cuh"
#include "delete_list_kernels.cuh"
#include "utils.cuh"

class DeleteList {
   public:
    DeleteList(uint initial_capacity = 1000u)
        : d_delete_list_(nullptr),
          growth_factor_(1.3f),
          capacity_(std::max(1u, initial_capacity)),
          size_(0) {
        gpuErrchk(cudaMalloc(&d_delete_list_, capacity_ * sizeof(uint)));
    }

    ~DeleteList() {
        cudaFree(d_delete_list_);
        d_delete_list_ = nullptr;
    }

    uint size() const noexcept { return size_; }
    uint capacity() const noexcept { return capacity_; }

    void addNode(uint node_id) {
        if (size_ >= capacity_)
            expandDeleteList();

        gpuErrchk(
            cudaMemcpy(d_delete_list_ + size_, &node_id, sizeof(uint), cudaMemcpyHostToDevice));
        ++size_;
    }

    uint*       data() noexcept { return d_delete_list_; }
    const uint* data() const noexcept { return d_delete_list_; }

    void clear() noexcept {
        cudaFree(d_delete_list_);
        d_delete_list_ = nullptr;
        capacity_      = 0;
        size_          = 0;
    }

   private:
    void expandDeleteList() {
        const uint new_capacity =
            static_cast<uint>(std::max<uint>(1u, static_cast<uint>(capacity_ * growth_factor_)));
        uint* new_buffer = nullptr;
        gpuErrchk(cudaMalloc(&new_buffer, new_capacity * sizeof(uint)));

        if (d_delete_list_ && capacity_ > 0) {
            gpuErrchk(cudaMemcpy(
                new_buffer, d_delete_list_, capacity_ * sizeof(uint), cudaMemcpyDeviceToDevice));
            cudaFree(d_delete_list_);
        }

        d_delete_list_ = new_buffer;
        capacity_      = new_capacity;
    }

   private:
    uint* d_delete_list_ = nullptr;
    float growth_factor_ = 1.3f;
    uint  capacity_      = 0;
    uint  size_          = 0;
};

__host__ inline bool isNodeInDeleteList(uint* d_delete_list, size_t size, uint node_id) {
    if (size == 0)
        return false;

    bool  h_found = false;
    bool* d_found = nullptr;

    cudaMalloc(&d_found, sizeof(bool));
    cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice);

    const uint threads = 256;
    const uint blocks  = (size + threads - 1) / threads;

    checkIfNodeDeletedKernel<<<blocks, threads>>>(d_delete_list, size, node_id, d_found);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_found);

    return h_found;
}

__device__ inline bool isNodeInDeleteList(const uint* d_delete_list, size_t size, uint node_id) {
    for (size_t i = 0; i < size; ++i) {
        if (d_delete_list[i] == node_id)
            return true;
    }
    return false;
}
