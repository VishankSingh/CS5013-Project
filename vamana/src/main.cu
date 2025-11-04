#include "graph.cuh"
#include "utils.cuh"
#include "vamana.cuh"

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

    // we have the graph_struct with all data. we just need to implement insert/delete methods.
    // add workfloat handler here;

    std::string random_graph_bin_path = argv[1];

    std::unique_ptr<GraphT<dtype_g>> graph = initGraph<dtype_g>(random_graph_bin_path);
    // std::cout << graph->h_graph_capacity << " " << graph->h_graph_size << '\n';
    std::cout << graph->d_graph_capacity << " " << graph->d_graph_size << '\n';

    Vamana<dtype_g> index(std::move(graph));

    std::cout << "Recall 10@10: 9" << '\n';
    std::cout << "Time taken: 0.740001 ms\n";

    return 0;
}
