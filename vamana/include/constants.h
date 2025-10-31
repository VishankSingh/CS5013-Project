#pragma once
#include <cstdint>

enum class GraphDataType : uint8_t { float32, int32, unsupported };

namespace FreshVamana::Consts {
constexpr uint D_g           = 128;
constexpr uint R_g           = 64;
constexpr uint rg_bin_size_g = 10000;  // size of binary file randomgraph.bin
constexpr uint N_g           = 10000;  // total size of graph

constexpr GraphDataType dtype_g = GraphDataType::float32;

}  // namespace FreshVamana::Consts
