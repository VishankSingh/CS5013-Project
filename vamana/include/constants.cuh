#pragma once
#include <cstdint>

using uint = unsigned int;

namespace FreshVamana::Consts {
constexpr uint D_g           = 128; // dimension of a vector in the actual input dataset
constexpr uint R_g           = 64; // max out-degree of a node in the graph
constexpr uint rg_bin_size_g = 10000;  // size of the graph index binary file randomgraph.bin
constexpr uint N_g           = 10000;  // total size of graph
constexpr uint L_g           = 150;
constexpr uint medoid_g            = 5000;
constexpr uint max_num_parents_per_query = 600;

using dtype_g = float; // need to change this if we want to support other data types like double, int8_t, etc.

// BINARY_LAYOUT_OF_AN_ENTRY_IN_THE_GRAPH_INDEX:
// <vector_of_D_g_dimension><out_degree><indices_of_the_out_neighbors_in_the_graph_index>
constexpr size_t graph_entry_size_in_bytes =
    D_g * sizeof(dtype_g) + 1 * sizeof(uint) + R_g * sizeof(uint);

// Reverse Edges (Runtime Generated)
constexpr uint max_num_entries_in_reverse_index_g = 500;
constexpr uint reverse_index_entry_size_in_bytes_g  = (max_num_entries_in_reverse_index_g + 1) * sizeof(uint); //<n><entry1><entry2>...<entry_n>



enum class queryType {  undefined_q = -1,
                        insert_q = 0,
                        delete_q = 1, //WARNING: as delete is conflicting with the C++ `delete` keyword
                        search_q = 2}; 

}  // END namespace FreshVamana::Consts

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
