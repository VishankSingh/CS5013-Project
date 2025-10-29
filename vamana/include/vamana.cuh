#pragma once
#include <vector>

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
    void insert_(/*something*/);
    void delete_(/*something*/);
    void search_(/*something*/);

   private:
    // Data structures
    //
};