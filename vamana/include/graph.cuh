// graph.cuh
#pragma once
#include "constants.cuh"

#include <cstdint>

using uint = unsigned int;



template <typename T__> 
// T in the following variable names indicates the template type parameter T
struct GraphT {
    // This is the actual entry raw binary layout in the graph index(random_graph.bin)
    static constexpr FreshVamana::Consts::dtype_g *vec_elements = nullptr; // Vector of 128 entries each of data type float(default)
    static constexpr uint outDegree = FreshVamana::Consts::R_g;
    static constexpr uint *adjList = nullptr; // out_neighbors indices in the graph index
};

template <typename T__> 
struct QueryT{
    static constexpr FreshVamana::Consts::queryType type = FreshVamana::Consts::queryType::search_q; // have to overwrite(INITIALIZE) this at the time of instatntiation of the query struct
    static constexpr FreshVamana::Consts::dtype_g *vec_elements = nullptr; // Vector of 128 entries each of data type float(default)
};
