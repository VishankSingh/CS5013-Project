#pragma once
#include <cstdint>

namespace FreshVamana::Consts {
constexpr uint D_g           = 128;
constexpr uint R_g           = 64;
constexpr uint rg_bin_size_g = 10000;  // size of binary file randomgraph.bin
constexpr uint N_g           = 10000;  // total size of graph
constexpr uint L_g           = 150;

constexpr uint medoid_g            = 5000;
constexpr uint max_paren_per_query = 600;

using dtype_g = float;

constexpr uint graph_entry_size_g =
    D_g * sizeof(dtype_g) + sizeof(unsigned) + R_g * sizeof(unsigned);

}  // namespace FreshVamana::Consts
