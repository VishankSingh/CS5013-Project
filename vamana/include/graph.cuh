// graph.cuh
#pragma once
#include "constants.cuh"
// #include "vamana.h"

#include <cstdint>

template <typename T__>
struct GraphT {
    uint8_t* d_graph          = nullptr;
    uint     d_graph_size     = 0;
    uint     d_graph_capacity = 0;

    static constexpr uint dim    = FreshVamana::Consts::D_g;
    static constexpr uint degree = FreshVamana::Consts::R_g;
};
