// graph.cuh
#pragma once
#include "constants.h"
// #include "vamana.h"

#include <cstdint>

template <typename T__, uint D__, uint R__>
struct GraphT {
    uint8_t* d_graph          = nullptr;
    uint     d_graph_size     = 0;
    uint     d_graph_capacity = 0;

    static constexpr uint dim    = D__;
    static constexpr uint degree = R__;

    using value_type = T__;

    [[nodiscard]] static constexpr size_t getGraphEntrySize() noexcept {
        return D__ * sizeof(T__) + sizeof(unsigned) + R__ * sizeof(unsigned);
    }
};
