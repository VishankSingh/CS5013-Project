#pragma once
#include "constants.cuh"
#include "insert_list_kernels.cuh"
#include "utils.cuh"

#include "globals.cuh"

// FreshVamana::Consts::D_g * sizeof(FreshVamana::Consts::dtype_g) = insert list element size
constexpr uint insert_entry_bytes_g =
    FreshVamana::Consts::D_g * sizeof(FreshVamana::Consts::dtype_g);

// TODO: correct & complete this.
class InsertList {
   public:
    InsertList(uint initial_capacity = 1000u) {
        growth_factor_ = 1.3f;
        capacity_      = std::max(1u, initial_capacity);
        size_          = 0;
        using namespace FreshVamana;
        gpuErrchk(cudaMalloc(&Globals::d_insert_list_g, capacity_ * insert_entry_bytes_g));
    }

    ~InsertList() {
        using namespace FreshVamana;
        cudaFree(Globals::d_insert_list_g);
        Globals::d_insert_list_g = nullptr;
    }

    uint size() const noexcept { return size_; }
    uint capacity() const noexcept { return capacity_; }

    void addVectors(FreshVamana::Consts::dtype_g* vectors, size_t num) {
        if (num == 0)
            return;

        if (size_ + num >= capacity_)
            expandInsertList();

        using namespace FreshVamana;

        FreshVamana::Consts::dtype_g* dst =
            Globals::d_insert_list_g + static_cast<size_t>(size_) * FreshVamana::Consts::D_g;
        const size_t bytes = num * insert_entry_bytes_g;
        gpuErrchk(cudaMemcpy(dst, vectors, bytes, cudaMemcpyHostToDevice));
        size_ = static_cast<uint>(size_ + num);
    }

    FreshVamana::Consts::dtype_g* data() noexcept { return FreshVamana::Globals::d_insert_list_g; }
    const FreshVamana::Consts::dtype_g* data() const noexcept {
        return FreshVamana::Globals::d_insert_list_g;
    }

    void clear() noexcept {
        using namespace FreshVamana;
        cudaFree(FreshVamana::Globals::d_insert_list_g);
        FreshVamana::Globals::d_insert_list_g = nullptr;
        capacity_                             = 0;
        size_                                 = 0;
    }

   private:
    void expandInsertList() {
        using namespace FreshVamana;

        const uint new_capacity =
            static_cast<uint>(std::max<uint>(1u, static_cast<uint>(capacity_ * growth_factor_)));
        Consts::dtype_g* new_buffer = nullptr;
        gpuErrchk(
            cudaMalloc(&new_buffer, static_cast<size_t>(new_capacity) * insert_entry_bytes_g));

        if (Globals::d_insert_list_g && capacity_ > 0) {
            gpuErrchk(cudaMemcpy(new_buffer,
                                 Globals::d_insert_list_g,
                                 size_ * insert_entry_bytes_g,
                                 cudaMemcpyDeviceToDevice));
            cudaFree(Globals::d_insert_list_g);
        }

        Globals::d_insert_list_g = new_buffer;
        capacity_                = new_capacity;
    }

   private:
    float growth_factor_ = 1.3f;
    uint  capacity_      = 0;
    uint  size_          = 0;
};

__host__ inline bool isNodeInInsertList(FreshVamana::Consts::dtype_g* d_insert_list,
                                        size_t                        d_insert_list_size,
                                        FreshVamana::Consts::dtype_g* d_query_vecs,
                                        size_t                        d_query_vecs_size) {
    using namespace FreshVamana;
    if (d_insert_list_size == 0 || d_query_vecs_size == 0)
        return false;

    unsigned int  h_found = 0u;
    unsigned int* d_found = nullptr;
    cudaMalloc(&d_found, sizeof(unsigned int));
    cudaMemcpy(d_found, &h_found, sizeof(unsigned int), cudaMemcpyHostToDevice);

    const uint threads = 256u;
    const uint blocks  = static_cast<uint>((d_insert_list_size + threads - 1) / threads);

    checkIfNodeInsertedKernel<<<blocks, threads>>>(d_insert_list,
                                                   static_cast<uint>(d_insert_list_size),
                                                   d_query_vecs,
                                                   static_cast<uint>(d_query_vecs_size),
                                                   d_found);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_found, d_found, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_found);
    return (h_found != 0u);
}

__device__ inline bool isNodeInInsertList(FreshVamana::Consts::dtype_g* d_insert_list,
                                          size_t                        d_insert_list_size,
                                          FreshVamana::Consts::dtype_g* d_query_vecs,
                                          size_t                        d_query_vecs_size) {
    using namespace FreshVamana::Consts;
    for (size_t i = 0; i < d_insert_list_size; ++i) {
        bool same = true;
        for (uint j = 0; j < D_g; ++j) {
            dtype_g a = d_insert_list[i * D_g + j];
            dtype_g b = d_query_vecs[j];
            if constexpr (std::is_floating_point_v<dtype_g>) {
                if (fabsf(a - b) > 1e-6f) {
                    same = false;
                    break;
                }
            } else {
                if (a != b) {
                    same = false;
                    break;
                }
            }
        }
        if (same)
            return true;
    }
    return false;

    return false;
}
