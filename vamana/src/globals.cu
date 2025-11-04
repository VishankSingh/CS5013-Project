#include "globals.cuh"

#include "delete_list.cuh"

namespace FreshVamana::Globals {

__managed__ uint d_graph_capacity = 0;
__managed__ uint d_graph_size     = 0;

}  // namespace FreshVamana::Globals
