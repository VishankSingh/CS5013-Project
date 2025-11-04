#pragma once
#include "constants.cuh"

namespace FreshVamana::Globals {

extern __device__ __managed__ uint d_graph_capacity;
extern __device__ __managed__ uint d_graph_size;

}  // namespace FreshVamana::Globals
