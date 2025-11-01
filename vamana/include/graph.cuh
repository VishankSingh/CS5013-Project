// graph.cuh
#pragma once
#include "constants.h"
// #include "vamana.h"

#include <cstdint>

template <typename T__, uint D__, uint R__>
struct Graph_t {
    uint8_t* h_graph          = nullptr;
    uint     h_graph_size     = 0;
    uint     h_graph_capacity = 0;

    uint8_t* d_graph          = nullptr;
    uint     d_graph_size     = 0;
    uint     d_graph_capacity = 0;

    static constexpr uint dim    = D__;
    static constexpr uint degree = R__;

    using value_type = T__;

    [[nodiscard]] static constexpr size_t get_graph_entry_size() noexcept {
        return D__ * sizeof(T__) + sizeof(unsigned) + R__ * sizeof(unsigned);
    }
};

// template <GraphDataType DataType__>
// struct Graph_t {
//     uint8_t* h_graph          = nullptr;
//     uint     h_graph_size     = 0;
//     uint     h_graph_capacity = 0;

//     uint8_t* d_graph          = nullptr;
//     uint     d_graph_size     = 0;
//     uint     d_graph_capacity = 0;

//     // IMP: use if constexpr statements whenever possible if graph_type is used
//     // or idk
//     static constexpr GraphDataType graph_type = GraphTypeInfo<DataType__>::dtype;

//     [[nodiscard]] static constexpr size_t get_graph_entry_size() noexcept {
//         if constexpr (graph_type == GraphDataType::unsupported) {
//             return 0;
//         } else {
//             return D * sizeof(DataType__) + sizeof(unsigned) + R * sizeof(unsigned);
//         }
//     }
// };