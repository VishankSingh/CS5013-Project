// graph.cuh
#pragma once
#include "constants.cuh"

#include <cstdint>

using uint = unsigned int;

// template <typename T__>
// // T in the following variable names indicates the template type parameter T
// struct GraphT {
//     // This is the actual entry raw binary layout in the graph index(random_graph.bin)
//     using dtype_g = FreshVamana::Consts::dtype_g;
//     using FreshVamana::Consts::R_g;

//     dtype_g *vec_elements = nullptr; // (default) Vector of 128 entries each of data_type =float
//     uint outDegree = R_g; // (default) can be overridden using the custom constructor
//     uint *adjList = nullptr; // out_neighbors indices in the graph index

//     // Default constructor; Keep R_g as fallback out-degree
//     GraphT() = default;

//     // Custom constructor to set a different out-degree than R_g
//     GraphT(uint outDegree_arg) : outDegree(outDegree_arg) {}

//     getNode(uint idx) {
//         return adjList[idx];
//     }

//     // Factory to create a view from raw binary memory
//     __host__ __device__
//     static GraphT from_blob(uint8_t* base_ptr) {
//         GraphT g;
//         g.vec_elements = reinterpret_cast<dtype_g*>(base_ptr);

//         uint offset_vec = FreshVamana::Consts::D_g * sizeof(dtype_g);
//         g.outDegree = *reinterpret_cast<uint*>(base_ptr + offset_vec);

//         uint offset_adj = offset_vec + sizeof(uint);
//         g.adjList = reinterpret_cast<uint*>(base_ptr + offset_adj);
//         return g;
//     }

// };

// template <typename T__>
// struct QueryT{
//     FreshVamana::Consts::queryType type = FreshVamana::Consts::queryType::search_q; // have to
//     overwrite(INITIALIZE) this at the time of instantiation of the query struct
//     FreshVamana::Consts::dtype_g *vec_elements = nullptr; // (default) Vector of 128 entries each
//     of data_type= float
// };

template <typename T__>
struct GraphT {
    uint8_t* d_graph = nullptr;
    // uint     d_graph_size     = 0;
    // uint     d_graph_capacity = 0;

    // static constexpr uint dim    = FreshVamana::Consts::D_g;
    // static constexpr uint degree = FreshVamana::Consts::R_g;
};
