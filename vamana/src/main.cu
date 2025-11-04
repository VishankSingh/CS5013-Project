#include "graph.cuh"
#include "utils.cuh"
#include "vamana.cuh"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

__global__ void pp() {
    printf("\n\n%d\n\n", FreshVamana::Globals::d_graph_size);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <graph> <basepoints> <output>\n", argv[0]);
        return 1;
    }

    // we have the graph_struct with all data. we just need to implement insert/delete methods.
    // add workfloat handler here;

    std::string random_graph_bin_path = argv[1];

    std::unique_ptr<GraphT<dtype_g>> graph = initGraph<dtype_g>(random_graph_bin_path);
    // std::cout << graph->h_graph_capacity << " " << graph->h_graph_size << '\n';
    std::cout << FreshVamana::Globals::d_graph_capacity << " " << FreshVamana::Globals::d_graph_size
              << '\n';

    Vamana<dtype_g> index(std::move(graph));

    index.search(0);

    // pp<<<1, 1>>>();

    return 0;
}
