#pragma once
#include <memory>
#include <vector>
#include "graph.cuh"

// TODO: design this

/*
Binary file data layout

graph is a random graph of size 10000
graph has format

struct node {
    float vec[128];
    uint degree;
    uint neighbors[degree];
} node_t;

basepoints is all the points in the graph
float basepoinst[N][128];
*/

// float or integer graph points
// for later part, not yet, just assume uint
template <typename Data__>
class Vamana {
   public:
    Vamana(std::unique_ptr<Graph_t<dtype_g, 128, 64>> graph) : graph_(std::move(graph)) {};

    void insert_(/*something*/);
    void delete_(/*something*/);
    void search_(/*something*/);

   private:
    std::unique_ptr<Graph_t<dtype_g, 128, 64>> graph_;
    // Data structures
    //
};