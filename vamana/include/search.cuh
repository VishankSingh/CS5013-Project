#pragma once

#include "bloom_filter.cuh"
#include "constants.cuh"
#include "graph.cuh"
#include "utils.cuh"

#include "kernels.cuh"

__device__ bool contains(unsigned* set, unsigned count, unsigned el) {
    for (uint i = 0; i < count; i++) {
        if (set[i] == el) {
            return true;
        }
    }

    return false;
}

__global__ void initializeParents(bool* d_hasParent, unsigned* d_parents) {
    unsigned query_id = blockIdx.x;
    unsigned tid      = threadIdx.x;

    if (tid == 0) {
        d_hasParent[query_id] = true;
        d_parents[query_id]   = FreshVamana::Consts::medoid_g;
    }
}

template <typename T__>
__global__ void initializeWorklist(uint8_t*  d_graph,
                                   T__*      d_queryVecs,
                                   unsigned* d_worklist,
                                   unsigned* d_worklistCount,
                                   float*    d_worklistDist,
                                   bool*     d_worklistVisited) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    unsigned worklistOffset = FreshVamana::Consts::L_g * queryID;

    T__* queryVec = d_queryVecs + FreshVamana::Consts::D_g * queryID;
    T__* medoidVec =
        (T__*)(d_graph + FreshVamana::Consts::graph_entry_size_g * FreshVamana::Consts::medoid_g);

    if (tid == 0) {
        d_worklist[worklistOffset]        = FreshVamana::Consts::medoid_g;
        d_worklistCount[queryID]          = 1;
        d_worklistVisited[worklistOffset] = true;

        float dist = 0;
        for (uint i = 0; i < FreshVamana::Consts::D_g; i++) {
            T__ diff = queryVec[i] - medoidVec[i];
            dist += diff * diff;
        }
        d_worklistDist[worklistOffset] = dist;
    }
}

/* Adds unvisited neighbours of nodes in d_parents to d_neighbors
 *
 * d_graph           - The graph
 * d_hasParent       - Whether there is a node to be visited (a parent) for a query
 * d_parents         - If the query has a parent, then the index of the parent
 * d_bloomFilters    - The bloom filters for checking if a node has been visited
 * d_neighbors       - Array for putting unvisited neighbors into
 * d_neighborsCount  - No. of unvisited neighbors for each query
 * d_visitedSet      - Array of visited points for each query
 * d_visitedSetCount - No. of visited points for each query
 */
__global__ void filterNeighbors(uint8_t*  d_graph,
                                bool*     d_hasParent,
                                unsigned* d_parents,
                                bool*     d_bloomFilters,
                                unsigned* d_neighbors,
                                unsigned* d_neighborsCount,
                                unsigned* d_visitedSet,
                                unsigned* d_visitedSetCount) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    if (!d_hasParent[queryID])
        return;

    bool* bloomFilter =
        d_bloomFilters + (queryID * BF_MEMORY);  // Get the bloom filter for this query
    unsigned parent = d_parents[queryID];        // Get the parent for this query

    // Get the pointers to the degree and neighbors of the parent
    unsigned* degreePtr   = (unsigned*)(d_graph + parent * FreshVamana::Consts::graph_entry_size_g +
                                      FreshVamana::Consts::D_g * sizeof(float));
    unsigned* neighborPtr = degreePtr + 1;

    __syncthreads();

    // Initialize neighbor count to zero, reset hasParent
    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID]      = false;

        unsigned visitedSetIdx = atomicAdd(&d_visitedSetCount[queryID], 1);
        if (visitedSetIdx < FreshVamana::Consts::max_paren_per_query) {
            d_visitedSet[FreshVamana::Consts::max_paren_per_query * queryID + visitedSetIdx] =
                parent;
        } else {
            printf("Limit hit for visited set: %d\n", queryID);
            atomicSub(&d_visitedSetCount[queryID], 1);
        }
    }

    __syncthreads();

    // Loop over each neighbor
    unsigned degree = *degreePtr;

    for (unsigned ii = tid; ii < degree; ii += blockDim.x) {
        unsigned neighbor = neighborPtr[ii];

        if (neighbor == queryID)
            continue;

        // Ensure the neighbor has not been visited yet
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);

            // Add the neighbor to d_neighbors
            unsigned neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1);
            d_neighbors[(FreshVamana::Consts::R_g + 1) * queryID + neighborIdx] = neighbor;
        }
    }
}

__global__ void mergeIntoWorklist(unsigned* d_worklistCount,
                                  unsigned* d_worklist,
                                  float*    d_worklistDist,
                                  bool*     d_worklistVisited,
                                  unsigned* d_neighborsCount,
                                  unsigned* d_neighbors,
                                  float*    d_neighborsDist,
                                  bool*     d_hasParent,
                                  unsigned* d_parents,
                                  bool*     d_nextIter) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    unsigned neighborsOffset = queryID * (FreshVamana::Consts::R_g + 1);
    unsigned worklistOffset  = queryID * FreshVamana::Consts::L_g;

    unsigned numNeighbors    = d_neighborsCount[queryID];
    unsigned worklistSize    = d_worklistCount[queryID];
    unsigned newWorklistSize = min(numNeighbors + worklistSize, FreshVamana::Consts::L_g);

    __shared__ unsigned sortedPositions[FreshVamana::Consts::R_g + FreshVamana::Consts::L_g + 1];

    unsigned id;
    float    dist;
    bool     visited;
    unsigned newPos = FreshVamana::Consts::L_g;

    if (tid < worklistSize) {
        // Fist L threads find new position for worklist elements
        unsigned before      = lowerBound(&d_neighborsDist[neighborsOffset],
                                     0,
                                     numNeighbors,
                                     d_worklistDist[worklistOffset + tid]);
        id                   = d_worklist[worklistOffset + tid];
        dist                 = d_worklistDist[worklistOffset + tid];
        visited              = d_worklistVisited[worklistOffset + tid];
        newPos               = before + tid;
        sortedPositions[tid] = newPos;
    } else if (tid >= FreshVamana::Consts::L_g && tid < FreshVamana::Consts::L_g + numNeighbors) {
        // Next R + 1 threads find new position for neighbors
        unsigned idx         = tid - FreshVamana::Consts::L_g;  // Index into the neighbors array
        unsigned before      = upperBound(&d_worklistDist[worklistOffset],
                                     0,
                                     worklistSize,
                                     d_neighborsDist[neighborsOffset + idx]);
        id                   = d_neighbors[neighborsOffset + idx];
        dist                 = d_neighborsDist[neighborsOffset + idx];
        visited              = false;
        newPos               = before + idx;
        sortedPositions[tid] = newPos;
    }

    __syncthreads();
    __threadfence_block();

    if (newPos < newWorklistSize) {
        d_worklist[worklistOffset + newPos]        = id;
        d_worklistDist[worklistOffset + newPos]    = dist;
        d_worklistVisited[worklistOffset + newPos] = visited;
    }

    __syncthreads();
    __threadfence_block();

    if (tid == 0) {
        d_worklistCount[queryID] = newWorklistSize;

        for (unsigned ii = 0; ii < newWorklistSize; ii++) {
            // Find the closest unvisited node, set it as the parent for the next iteration, and
            // mark it as visited.
            if (!d_worklistVisited[worklistOffset + ii]) {
                *d_nextIter                            = true;
                d_hasParent[queryID]                   = true;
                d_parents[queryID]                     = d_worklist[worklistOffset + ii];
                d_worklistVisited[worklistOffset + ii] = true;

                break;
            }
        }
    }
}

// Performs greedy search and returns the visited sets
template <typename T__>
void greedySearch(uint8_t*  d_graph,
                  T__*      d_queryVecs,
                  unsigned* d_visitedSet /*empty*/,
                  unsigned* d_visitedSetCount /*0*/,
                  unsigned  batchSize) {
    bool*     d_hasParent;  // 10k
    unsigned* d_parents;    // 10k unsigned
    bool*     d_bloomFilters;

    gpuErrchk(cudaMalloc(&d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_parents, batchSize * sizeof(unsigned)));
    size_t allocSize = batchSize * BF_MEMORY * sizeof(bool);

    printf("Allocating %.2f MB (%zu bytes) for d_bloomFilters\n",
           allocSize / (1024.0 * 1024.0),
           allocSize);

    gpuErrchk(cudaMalloc(&d_bloomFilters, batchSize * BF_MEMORY * sizeof(bool)));
    gpuErrchk(cudaMemset(d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool)));

    unsigned* d_neighbors;         // 10k * (64+1) unsigned
    unsigned* d_neighborsCount;    // 10k * 1
    float*    d_neighborDists;     // 10k * (64+1) float
    unsigned* d_neighborsAux;      // 10k * (64+1) unsigned
    float*    d_neighborDistsAux;  // 10k * (64+1) unsigned

    gpuErrchk(
        cudaMalloc(&d_neighbors, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)));
    gpuErrchk(
        cudaMemset(d_neighbors, 0, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)));

    gpuErrchk(cudaMalloc(&d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_neighborsCount, 0, batchSize * sizeof(unsigned)));

    gpuErrchk(
        cudaMalloc(&d_neighborDists, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(float)));
    gpuErrchk(
        cudaMalloc(&d_neighborsAux, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux,
                         batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(float)));

    // 150 is worklist size
    unsigned* d_worklist;         // 10k * 150 unsigned
    unsigned* d_worklistCount;    // 10k unsigned
    float*    d_worklistDist;     // 10k * 150 float
    bool*     d_worklistVisited;  // 10k * 150 bool

    gpuErrchk(cudaMalloc(&d_worklist, batchSize * FreshVamana::Consts::L_g * sizeof(unsigned)));

    gpuErrchk(cudaMalloc(&d_worklistCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_worklistCount, 0, batchSize * sizeof(unsigned)));

    gpuErrchk(cudaMalloc(&d_worklistDist, batchSize * FreshVamana::Consts::L_g * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_worklistVisited, batchSize * FreshVamana::Consts::L_g * sizeof(bool)));

    bool  nextIter;
    bool* d_nextIter;

    gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    initializeParents<<<batchSize, 1>>>(d_hasParent, d_parents);
    initializeWorklist<T__><<<batchSize, 1>>>(
        d_graph, d_queryVecs, d_worklist, d_worklistCount, d_worklistDist, d_worklistVisited);
    // cudaDeviceSynchronize();

    int iter = 0;
    do {
        iter++;
        gpuErrchk(cudaMemset(d_nextIter, false, sizeof(bool)));

        filterNeighbors<<<batchSize, FreshVamana::Consts::R_g>>>(d_graph,
                                                                 d_hasParent,
                                                                 d_parents,
                                                                 d_bloomFilters,
                                                                 d_neighbors,
                                                                 d_neighborsCount,
                                                                 d_visitedSet,
                                                                 d_visitedSetCount);

        // gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        computeDists<<<batchSize, FreshVamana::Consts::R_g * 8>>>(d_graph,
                                                                  d_neighbors,
                                                                  d_neighborsCount,
                                                                  d_queryVecs,
                                                                  d_neighborDists,
                                                                  (FreshVamana::Consts::R_g + 1));
        // gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        sortByDistance<<<batchSize,
                         FreshVamana::Consts::R_g,
                         FreshVamana::Consts::R_g * sizeof(unsigned)>>>(
            d_neighbors,
            d_neighborsCount,
            d_neighborDists,
            d_neighborsAux,
            d_neighborDistsAux,
            FreshVamana::Consts::R_g + 1);
        // gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        mergeIntoWorklist<<<batchSize, FreshVamana::Consts::R_g + FreshVamana::Consts::L_g>>>(
            d_worklistCount,
            d_worklist,
            d_worklistDist,
            d_worklistVisited,

            d_neighborsCount,
            d_neighbors,
            d_neighborDists,

            d_hasParent,
            d_parents,
            d_nextIter);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));
    } while (nextIter);

    gpuErrchk(cudaFree(d_hasParent));
    gpuErrchk(cudaFree(d_parents));
    gpuErrchk(cudaFree(d_bloomFilters));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    gpuErrchk(cudaFree(d_worklist));
    gpuErrchk(cudaFree(d_worklistCount));
    gpuErrchk(cudaFree(d_worklistDist));
    gpuErrchk(cudaFree(d_worklistVisited));

    gpuErrchk(cudaFree(d_nextIter));

    printf("Greedy search finished in %d iterations.\n", iter);
}
