#pragma once
#include <cstdint>

using uint = unsigned int;

namespace FreshVamana::Consts {
constexpr uint D_g           = 128;
constexpr uint R_g           = 64;
constexpr uint rg_bin_size_g = 10000;  // size of binary file randomgraph.bin
constexpr uint N_g           = 10000;  // total size of graph
constexpr uint L_g           = 150;

constexpr uint medoid_g            = 5000;
constexpr uint max_paren_per_query = 600;

using dtype_g = float;

constexpr uint graph_entry_size_g = D_g * sizeof(dtype_g) + sizeof(uint) + R_g * sizeof(uint);

constexpr uint max_reverse_index_entries_g = 500;
constexpr uint reverse_index_entry_size_g  = (max_reverse_index_entries_g + 1) * sizeof(uint);

// BINARY_LAYOUT_OF_AN_ENTRY_IN_THE_GRAPH_INDEX:
// <vector_of_D_g_dimension><out_degree><indices_of_the_out_neighbors_in_the_graph_index>
constexpr size_t graph_entry_size_in_bytes =
    D_g * sizeof(float) + 1 * sizeof(uint) + R_g * sizeof(uint);

}  // namespace FreshVamana::Consts

/*
FreshVamana::Consts::D_g
FreshVamana::Consts::R_g
FreshVamana::Consts::rg_bin_size_g
FreshVamana::Consts::N_g
FreshVamana::Consts::L_g
FreshVamana::Consts::medoid_g
FreshVamana::Consts::max_paren_per_query
FreshVamana::Consts::dtype_g
FreshVamana::Consts::graph_entry_size_g
FreshVamana::Consts::max_reverse_index_entries_g
FreshVamana::Consts::reverse_index_entry_size_g



*/
