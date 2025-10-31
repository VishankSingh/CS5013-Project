#pragma once
#include "graph.cuh"

#include <memory>

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

using namespace FreshVamana::Consts;

template <GraphDataType DataType__>
class VamanaIndex {
   public:
    VamanaIndex(std::unique_ptr<Graph_t<dtype_g, D_g, R_g>> graph) : graph_(std::move(graph)) {};

    void insertPoint(/*something*/);
    void deletePoint(/*something*/);
    void search(/*something*/);

   private:
    std::unique_ptr<Graph_t<dtype_g, D_g, R_g>> graph_;
    // Data structures
    //
};