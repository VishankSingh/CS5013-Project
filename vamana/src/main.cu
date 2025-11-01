#include "graph.cuh"
#include "utils.h"
#include "vamana.cuh"
#include "vamana.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <graph> <basepoints> <output>\n", argv[0]);
        return 1;
    }

    // Graph_t<float, 128, 64> graph_struct;
    // driverFn(argv[1], argv[2], argv[3], graph_struct);

    // std::cout << graph_struct.h_graph_capacity << " " << graph_struct.h_graph_size << '\n';
    // std::cout << graph_struct.d_graph_capacity << " " << graph_struct.d_graph_size << '\n';

    // we have the graph_struct with all data. we just need to implement insert/delete methods.
    // add workfloat handler here;

    std::unique_ptr<Graph_t<dtype_g, D_g, R_g>> graph = initGraph<dtype_g, D_g, R_g>(argv[1]);
    std::cout << graph->h_graph_capacity << " " << graph->h_graph_size << '\n';
    std::cout << graph->d_graph_capacity << " " << graph->d_graph_size << '\n';

    Vamana<dtype_g, D_g, R_g> index(std::move(graph));

    return 0;
}
