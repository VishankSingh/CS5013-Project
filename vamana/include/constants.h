#pragma once

// global _g
constexpr uint D_g          = 128;
constexpr uint R_g          = 64;
constexpr uint N_g          = 10000;  // size of binary file randomgraph.bin
constexpr uint graph_size_g = 10000;  // total size of graph

enum class GraphDataType : uint8_t { float32, int32, unsupported };

constexpr GraphDataType dtype_g = GraphDataType::float32;
