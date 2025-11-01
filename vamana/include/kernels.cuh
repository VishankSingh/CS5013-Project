#pragma once
#include "constants.cuh"
#include "graph.cuh"
#include "timer.h"
#include "utils.cuh"

#include <cuda/std/limits>

typedef enum : uint8_t { INIT, PRUNED, NEIGHBOR } NodeState;

__global__ void computeDists(uint8_t*  d_graph,
                             unsigned* d_nodes,
                             unsigned* d_nodeCount,
                             float*    d_queryVecs,
                             float*    d_dists,
                             unsigned  rowSize) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    float* queryVec = d_queryVecs + FreshVamana::Consts::D_g * queryID;  // Pointer to query vector
    unsigned offset = rowSize * queryID;
    unsigned numNodes = d_nodeCount[queryID];

    // Initialize distances to zero
    for (unsigned i = tid; i < numNodes; i += blockDim.x) {
        d_dists[offset + i] = 0;
    }

    __syncthreads();

    // if (queryID == 0 & tid == 0) printf("NumNodes: %d\n", numNodes);

    // Assign 8 threads to each node
    for (unsigned j = tid / 8; j < numNodes; j += (blockDim.x + 7) / 8) {
        unsigned node    = d_nodes[offset + j];
        float*   nodeVec = (float*)(d_graph + FreshVamana::Consts::graph_entry_size_g *
                                                node);  // Pointer to node vector
        float    sum     = 0;

        // Sum up 8 dimensions in parallel
        for (unsigned i = tid % 8; i < FreshVamana::Consts::D_g; i += 8) {
            float diff = nodeVec[i] - queryVec[i];
            sum += diff * diff;
        }
        atomicAdd(&d_dists[offset + j], sum);
    }

    /*
    for (unsigned j = tid; j < numNodes; j += blockDim.x) {
        unsigned node = d_nodes[offset + j];
        float *nodeVec = (float*)(d_graph + graphEntrySize*node); // Pointer to node vector
        float sum = 0;
        for (int i = 0; i < D; i++) {
            float diff = nodeVec[i] - queryVec[i];
            sum += diff * diff;
        }
        atomicAdd(&d_dists[offset + j], sum);
    }
    */
}

__device__ unsigned lowerBound(float arr[], unsigned lo, unsigned hi, float target) {
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (target > arr[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ unsigned upperBound(float arr[], unsigned lo, unsigned hi, float target) {
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (target >= arr[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void sortByDistance(unsigned* d_items,
                               unsigned* d_itemCount,
                               float*    d_dists,
                               unsigned* d_itemsAux,
                               float*    d_distsAux,
                               unsigned  rowSize) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    unsigned numItems = d_itemCount[queryID];
    unsigned offset   = queryID * rowSize;

    extern __shared__ unsigned sortedPositions[];

    for (unsigned subarraySize = 2; subarraySize < 2 * numItems; subarraySize *= 2) {
        unsigned subarrayID = tid / subarraySize;
        unsigned start      = subarrayID * subarraySize;
        unsigned mid        = min(start + subarraySize / 2, numItems);
        unsigned end        = min(start + subarraySize, numItems);

        unsigned before;

        if (tid >= start && tid < mid) {
            // If current thread corresponds to lower half, find the no. of elements before this
            // element from the upper half
            before = lowerBound(&d_dists[offset + mid], 0, end - mid, d_dists[offset + tid]);
            sortedPositions[tid] = tid + before;
        } else if (tid >= mid && tid < end) {
            // If current thread corresponds to upper half, find the no. of elements before this
            // element from the lower half
            before = upperBound(&d_dists[offset + start], 0, mid - start, d_dists[offset + tid]);
            sortedPositions[tid] = before + (tid - mid + start);
        }

        __syncthreads();
        __threadfence_block();

        // Copy the neigbors to correct positions in auxiliary array
        for (uint i = tid; i < numItems; i += blockDim.x) {
            d_itemsAux[offset + sortedPositions[i]] = d_items[offset + i];
            d_distsAux[offset + sortedPositions[i]] = d_dists[offset + i];
        }

        __syncthreads();
        __threadfence_block();

        // Copy from auxiliary array back into original array
        for (uint i = tid; i < numItems; i += blockDim.x) {
            d_items[offset + i] = d_itemsAux[offset + i];
            d_dists[offset + i] = d_distsAux[offset + i];
        }

        __syncthreads();
        __threadfence_block();
    }
}

__global__ void getNeighbors(uint8_t*  d_graph,
                             unsigned  batchStart,
                             unsigned* d_neighbors,
                             unsigned* d_neighborsCount) {
    unsigned queryID         = blockIdx.x;
    unsigned extendedQueryID = batchStart + queryID;
    unsigned tid             = threadIdx.x;

    unsigned* degreePtr =
        (unsigned*)(d_graph + extendedQueryID * FreshVamana::Consts::graph_entry_size_g +
                    FreshVamana::Consts::D_g * sizeof(float));
    unsigned* neighborPtr = degreePtr + 1;

    unsigned degree = *degreePtr;

    if (tid == 0) {
        d_neighborsCount[queryID] = degree;
    }

    // TODO: Test necessity
    __syncthreads();

    // Loop over each neighbor
    for (uint ii = tid; ii < degree; ii += blockDim.x) {
        d_neighbors[(FreshVamana::Consts::R_g + 1) * queryID + ii] = neighborPtr[ii];
    }
}

// Could be merged with mergeIntoWorklist (with a dummy array for d_visited))
__global__ void mergeIntoVisitedSet(unsigned* d_visitedSetCount,
                                    unsigned* d_visitedSet,
                                    float*    d_visitedSetDists,
                                    unsigned* d_neighborsCount,
                                    unsigned* d_neighbors,
                                    float*    d_neighborsDist) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    unsigned visitedSetOffset = queryID * FreshVamana::Consts::max_paren_per_query;
    unsigned neighborsOffset  = queryID * (FreshVamana::Consts::R_g + 1);

    unsigned numNeighbors   = d_neighborsCount[queryID];
    unsigned visitedSetSize = d_visitedSetCount[queryID];

    unsigned newVisitedSetSize =
        min(numNeighbors + visitedSetSize, FreshVamana::Consts::max_paren_per_query);

    unsigned id;
    float    dist;
    unsigned newPos = FreshVamana::Consts::max_paren_per_query;

    if (tid < visitedSetSize) {
        unsigned before = lowerBound(&d_neighborsDist[neighborsOffset],
                                     0,
                                     numNeighbors,
                                     d_visitedSetDists[visitedSetOffset + tid]);
        id              = d_visitedSet[visitedSetOffset + tid];
        dist            = d_visitedSetDists[visitedSetOffset + tid];
        newPos          = before + tid;
    } else if (tid >= FreshVamana::Consts::max_paren_per_query &&
               tid < FreshVamana::Consts::max_paren_per_query + numNeighbors) {
        unsigned idx    = tid - FreshVamana::Consts::max_paren_per_query;
        unsigned before = upperBound(&d_visitedSetDists[visitedSetOffset],
                                     0,
                                     visitedSetSize,
                                     d_neighborsDist[neighborsOffset + idx]);
        id              = d_neighbors[neighborsOffset + idx];
        dist            = d_neighborsDist[neighborsOffset + idx];
        newPos          = before + idx;
    }

    __syncthreads();

    if (newPos < newVisitedSetSize) {
        d_visitedSet[visitedSetOffset + newPos]      = id;
        d_visitedSetDists[visitedSetOffset + newPos] = dist;
    }

    __syncthreads();

    if (tid == 0) {
        d_visitedSetCount[queryID] = newVisitedSetSize;
    }
}

// Robust Prune
__global__ void pruneOutNeighbors(uint8_t*   d_graph,
                                  unsigned   batchStart,
                                  unsigned*  d_visitedSet,
                                  unsigned*  d_visitedSetCount,
                                  float*     d_visitedSetDists,
                                  NodeState* d_visitedSetStatus,

                                  float*   d_queryVecs,
                                  uint8_t* d_reverseEdgeIndex,
                                  float    alpha) {
    for (unsigned iter = 1;; iter++) {
        unsigned queryID         = blockIdx.x;
        unsigned extendedQueryID = batchStart + queryID;
        unsigned tid             = threadIdx.x;

        unsigned numNodes         = d_visitedSetCount[queryID];
        unsigned visitedSetOffset = queryID * FreshVamana::Consts::max_paren_per_query;

        unsigned* degreePtr =
            (unsigned*)(d_graph + extendedQueryID * FreshVamana::Consts::graph_entry_size_g +
                        FreshVamana::Consts::D_g * sizeof(float));
        unsigned* neighborPtr = degreePtr + 1;

        // Initialization
        if (tid == 0 && iter == 1) {
            // Mark all candidate nodes as INIT
            for (unsigned i = 0; i < numNodes; i++) {
                d_visitedSetStatus[visitedSetOffset + i] = INIT;
            }

            // Set degree to zero
            *degreePtr = 0;
        }

        __syncthreads();

        // Don't add more neighbors if we already have R neighbors
        if (*degreePtr >= FreshVamana::Consts::R_g)
            return;

        __shared__ unsigned pStarShared[1];
        *pStarShared = cuda::std::numeric_limits<unsigned>::max();

        // Find p_star
        if (tid == 0) {
            for (unsigned i = 0; i < numNodes; i++) {
                // Find the closest 'INIT' node (p_star)
                if (d_visitedSetStatus[visitedSetOffset + i] != INIT)
                    continue;

                *pStarShared = d_visitedSet[visitedSetOffset + i];

                // Add an edge from the query to p_star
                unsigned oldDegree     = atomicAdd(degreePtr, 1);
                neighborPtr[oldDegree] = *pStarShared;

                // Set it to neighbor
                d_visitedSetStatus[visitedSetOffset + i] = NEIGHBOR;

                // We need to add a reverse edge from p_star to query
                unsigned* entryPtr =
                    (unsigned*)&d_reverseEdgeIndex[*pStarShared *
                                                   FreshVamana::Consts::reverse_index_entry_size];
                unsigned oldLen = atomicAdd(entryPtr, 1);
                if (oldLen < FreshVamana::Consts::max_reverse_index_entries) {
                    entryPtr[1 + oldLen] = extendedQueryID;
                } else {
                    // printf("Reverse index limit hit: %d\n", queryID);
                    atomicSub(entryPtr, 1);
                }

                // *d_nextIter = true;
                break;
            }
        }

        __syncthreads();

        uint pStar = *pStarShared;
        if (pStar == cuda::std::numeric_limits<unsigned>::max()) {
            return;
        }

        // Copy p_star into shared memory
        __shared__ float pStarVec[FreshVamana::Consts::D_g];
        float*           vecPtr =
            (float*)(d_graph +
                     pStar * FreshVamana::Consts::graph_entry_size_g);  // Pointer to query vector
        for (uint ii = tid; ii < FreshVamana::Consts::D_g; ii += blockDim.x) {
            pStarVec[ii] = vecPtr[ii];
        }

        __syncthreads();

        unsigned laneId        = threadIdx.x & 31;
        unsigned warpId        = threadIdx.x >> 5;  // warp index within block
        unsigned warpsPerBlock = blockDim.x >> 5;

        for (unsigned ii = warpId; ii < numNodes; ii += warpsPerBlock) {
            if (d_visitedSetStatus[visitedSetOffset + ii] != INIT)
                continue;

            unsigned     p    = d_visitedSet[visitedSetOffset + ii];
            const float* pVec = reinterpret_cast<const float*>(
                d_graph + p * FreshVamana::Consts::graph_entry_size_g);

            // cooperative distance computation
            float partial = 0.0f;
            for (uint j = laneId; j < FreshVamana::Consts::D_g; j += 32) {
                float diff = pVec[j] - pStarVec[j];
                partial    = fmaf(diff, diff, partial);
            }

            // warp reduce sum
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (laneId == 0) {
                float queryDist = d_visitedSetDists[visitedSetOffset + ii];
                if (partial * alpha <= queryDist) {
                    d_visitedSetStatus[visitedSetOffset + ii] = PRUNED;
                }
            }
        }
    }
}

void computeOutNeighbors(uint8_t*  d_graph,
                         float*    d_queryVecs,
                         unsigned* d_visitedSets,
                         unsigned* d_visitedSetCount,
                         float     alpha,
                         uint8_t*  d_reverseEdgeIndex,
                         unsigned  batchStart,
                         unsigned  batchSize) {
    bool log = false;

    cudaStream_t stream = 0;  // default stream
    GPUTimer     gputimer(stream, !log);
    // printf("%d\n", batchSize);

    float*     d_visitedSetDists;
    unsigned*  d_visitedSetAux;
    float*     d_visitedSetDistsAux;
    NodeState* d_visitedSetStatus;

    gpuErrchk(cudaMalloc(&d_visitedSetDists,
                         batchSize * FreshVamana::Consts::max_paren_per_query * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_visitedSetAux,
                         batchSize * FreshVamana::Consts::max_paren_per_query * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetDistsAux,
                         batchSize * FreshVamana::Consts::max_paren_per_query * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_visitedSetStatus,
                         batchSize * FreshVamana::Consts::max_paren_per_query * sizeof(NodeState)));

    unsigned* d_neighbors;
    unsigned* d_neighborsCount;
    float*    d_neighborsDists;
    unsigned* d_neighborsAux;
    float*    d_neighborsDistsAux;

    gpuErrchk(
        cudaMalloc(&d_neighbors, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(
        cudaMalloc(&d_neighborsDists, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(float)));
    gpuErrchk(
        cudaMalloc(&d_neighborsAux, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsDistsAux,
                         batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(float)));

    // bool nextIter;
    // bool *d_nextIter;
    // gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    gputimer.Start();
    getNeighbors<<<batchSize, FreshVamana::Consts::R_g>>>(
        d_graph, batchStart, d_neighbors, d_neighborsCount);
    gputimer.Stop();
    gpuErrchk(cudaDeviceSynchronize());
    // printf("getNeighbors GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    computeDists<<<batchSize, FreshVamana::Consts::R_g * 8>>>(d_graph,
                                                              d_neighbors,
                                                              d_neighborsCount,
                                                              d_queryVecs,
                                                              d_neighborsDists,
                                                              (FreshVamana::Consts::R_g + 1));
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("computeDists GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    sortByDistance<<<batchSize,
                     FreshVamana::Consts::R_g + 1,
                     (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)>>>(
        d_neighbors,
        d_neighborsCount,
        d_neighborsDists,
        d_neighborsAux,
        d_neighborsDistsAux,
        FreshVamana::Consts::R_g + 1);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("sortByDistance GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    computeDists<<<batchSize, 1024>>>(d_graph,
                                      d_visitedSets,
                                      d_visitedSetCount,
                                      d_queryVecs,
                                      d_visitedSetDists,
                                      FreshVamana::Consts::max_paren_per_query);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("computeDists GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    sortByDistance<<<batchSize,
                     FreshVamana::Consts::max_paren_per_query,
                     FreshVamana::Consts::max_paren_per_query * sizeof(unsigned)>>>(
        d_visitedSets,
        d_visitedSetCount,
        d_visitedSetDists,
        d_visitedSetAux,
        d_visitedSetDistsAux,
        FreshVamana::Consts::max_paren_per_query);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("sortByDistance GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    mergeIntoVisitedSet<<<batchSize,
                          FreshVamana::Consts::max_paren_per_query + FreshVamana::Consts::R_g>>>(
        d_visitedSetCount,
        d_visitedSets,
        d_visitedSetDists,
        d_neighborsCount,
        d_neighbors,
        d_neighborsDists);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("mergeIntoVisitedSet GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    pruneOutNeighbors<<<batchSize, 32>>>(d_graph,
                                         batchStart,
                                         d_visitedSets,
                                         d_visitedSetCount,
                                         d_visitedSetDists,
                                         d_visitedSetStatus,

                                         d_queryVecs,
                                         d_reverseEdgeIndex,
                                         alpha);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("pruneOutNeighbors GPU time: %f ms\n", gputimer.Elapsed());

    gpuErrchk(cudaFree(d_visitedSetDists));
    gpuErrchk(cudaFree(d_visitedSetAux));
    gpuErrchk(cudaFree(d_visitedSetDistsAux));
    gpuErrchk(cudaFree(d_visitedSetStatus));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborsDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborsDistsAux));
    // gpuErrchk(cudaFree(d_nextIter));

    // printf("Out neighbor pruning finished in %d iterations.\n", iter);
}

__global__ void loadQueryVecs(uint8_t* d_graph, float* d_queryVecs) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    float* queryVec = (float*)(d_graph + queryID * FreshVamana::Consts::graph_entry_size_g);

    for (uint i = tid; i < FreshVamana::Consts::D_g; i += blockDim.x) {
        d_queryVecs[queryID * FreshVamana::Consts::D_g + i] = queryVec[i];
    }
}

// Convert from the byte-array representation into the usual array-and-length representation
__global__ void parseReverseIndex(uint8_t*  d_reverseEdgeIndex,
                                  unsigned* d_reverseEdges,
                                  unsigned* d_reverseEdgeCount,
                                  unsigned* degreeCounts) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    unsigned* entryPtr =
        (unsigned*)(d_reverseEdgeIndex + queryID * FreshVamana::Consts::reverse_index_entry_size);
    unsigned numReverseEdges = *entryPtr;

    if (tid == 0) {
        d_reverseEdgeCount[queryID] = numReverseEdges;
        atomicAdd(&degreeCounts[numReverseEdges], 1);
    }

    for (unsigned ii = tid; ii < numReverseEdges; ii += blockDim.x) {
        d_reverseEdges[queryID * FreshVamana::Consts::max_reverse_index_entries + ii] =
            entryPtr[1 + ii];
    }
}

__global__ void getPrunableQueryIDs(unsigned* d_reverseEdgeCount,
                                    unsigned* d_queryIDs,
                                    unsigned* d_queryCount) {
    int      tid  = threadIdx.x;
    int      lane = tid % 32;
    uint     i    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned flag = (i < FreshVamana::Consts::N_g) && (d_reverseEdgeCount[i] != 0);
    if (i >= FreshVamana::Consts::N_g)
        return;

    // unsigned flag = (d_reverseEdgeCount[i] != 0);
    unsigned mask       = __ballot_sync(0xffffffff, flag);
    int      warpActive = __popc(mask);

    unsigned warpBase = 0;
    if (lane == 0) {
        warpBase = atomicAdd(d_queryCount, warpActive);
    }
    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    int posInWarp = __popc(mask & ((1u << lane) - 1));
    if (flag)
        d_queryIDs[warpBase + posInWarp] = i;
}

// Could be merged with mergeIntoVisitedSets
__global__ void mergeIntoReverseEdges(unsigned* d_reverseEdgeCount,
                                      unsigned* d_reverseEdges,
                                      float*    d_reverseEdgeDists,
                                      unsigned* d_neighborsCount,
                                      unsigned* d_neighbors,
                                      float*    d_neighborsDist) {
    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;

    unsigned reverseEdgeOffset = queryID * FreshVamana::Consts::max_reverse_index_entries;
    unsigned neighborsOffset   = queryID * (FreshVamana::Consts::R_g + 1);

    unsigned numNeighbors     = d_neighborsCount[queryID];
    unsigned reverseEdgeCount = d_reverseEdgeCount[queryID];

    unsigned newReverseEdgeCount =
        min(numNeighbors + reverseEdgeCount, FreshVamana::Consts::max_reverse_index_entries);

    unsigned id;
    float    dist;
    unsigned newPos = FreshVamana::Consts::max_reverse_index_entries;

    if (tid < reverseEdgeCount) {
        unsigned before = lowerBound(&d_neighborsDist[neighborsOffset],
                                     0,
                                     numNeighbors,
                                     d_reverseEdgeDists[reverseEdgeOffset + tid]);
        id              = d_reverseEdges[reverseEdgeOffset + tid];
        dist            = d_reverseEdgeDists[reverseEdgeOffset + tid];
        newPos          = before + tid;
    } else if (tid >= FreshVamana::Consts::max_reverse_index_entries &&
               tid < FreshVamana::Consts::max_reverse_index_entries + numNeighbors) {
        unsigned idx    = tid - FreshVamana::Consts::max_reverse_index_entries;
        unsigned before = upperBound(&d_reverseEdgeDists[reverseEdgeOffset],
                                     0,
                                     reverseEdgeCount,
                                     d_neighborsDist[neighborsOffset + idx]);
        id              = d_neighbors[neighborsOffset + idx];
        dist            = d_neighborsDist[neighborsOffset + idx];
        newPos          = before + idx;
    }

    __syncthreads();

    if (newPos < newReverseEdgeCount) {
        d_reverseEdges[reverseEdgeOffset + newPos]     = id;
        d_reverseEdgeDists[reverseEdgeOffset + newPos] = dist;
    }

    __syncthreads();

    if (tid == 0) {
        d_reverseEdgeCount[queryID] = newReverseEdgeCount;
    }
}

__global__ void pruneReverseEdges(uint8_t*   d_graph,
                                  unsigned*  d_queryIDs,
                                  unsigned*  d_reverseEdges,
                                  unsigned*  d_reverseEdgeCount,
                                  float*     d_reverseEdgeDists,
                                  NodeState* d_reverseEdgeStatus,

                                  float* d_queryVecs,
                                  float  alpha) {
    for (unsigned iter = 1;; iter++) {
        // unsigned bid = blockIdx.x;
        // unsigned queryID = d_queryIDs[bid];
        unsigned queryID = blockIdx.x;
        unsigned tid     = threadIdx.x;

        unsigned numNodes          = d_reverseEdgeCount[queryID];
        unsigned reverseEdgeOffset = queryID * FreshVamana::Consts::max_reverse_index_entries;

        unsigned* degreePtr =
            (unsigned*)(d_graph + queryID * FreshVamana::Consts::graph_entry_size_g +
                        FreshVamana::Consts::D_g * sizeof(float));
        unsigned* neighborPtr = degreePtr + 1;

        // printf("%d\n", queryID);

        if (tid == 0 && iter == 1) {
            // Mark all candidate nodes as INIT
            for (unsigned i = 0; i < numNodes; i++) {
                d_reverseEdgeStatus[reverseEdgeOffset + i] = INIT;
            }

            // Set degree to zero
            *degreePtr = 0;
        }

        __syncthreads();

        // Don't add more neighbors if we already have R neighbors
        if (*degreePtr >= FreshVamana::Consts::R_g)
            return;

        __shared__ unsigned pStarShared[1];
        *pStarShared = cuda::std::numeric_limits<unsigned>::max();

        // Find p_star
        if (tid == 0) {
            for (unsigned i = 0; i < numNodes; i++) {
                // Find the closest 'INIT' node (p_star)
                if (d_reverseEdgeStatus[reverseEdgeOffset + i] != INIT)
                    continue;

                *pStarShared = d_reverseEdges[reverseEdgeOffset + i];

                // Add an edge from the query to p_star
                // TODO: Are we sure that oldDegree is always less than R
                unsigned oldDegree     = atomicAdd(degreePtr, 1);
                neighborPtr[oldDegree] = *pStarShared;

                if (oldDegree > FreshVamana::Consts::R_g)
                    printf("Backward degree exceeded R: %d\n", queryID);

                // Set it to neighbor
                d_reverseEdgeStatus[reverseEdgeOffset + i] = NEIGHBOR;
                // *d_nextIter = true;
                break;
            }
        }

        __syncthreads();

        unsigned pStar = *pStarShared;
        if (pStar == cuda::std::numeric_limits<unsigned>::max())
            return;

        // Copy p_star into shared memory
        __shared__ float pStarVec[FreshVamana::Consts::D_g];
        float*           vecPtr =
            (float*)(d_graph +
                     pStar * FreshVamana::Consts::graph_entry_size_g);  // Pointer to query vector
        for (unsigned ii = tid; ii < FreshVamana::Consts::D_g; ii += blockDim.x) {
            pStarVec[ii] = vecPtr[ii];
        }

        __syncthreads();

        unsigned laneId        = threadIdx.x & 31;
        unsigned warpId        = threadIdx.x >> 5;  // warp index within block
        unsigned warpsPerBlock = blockDim.x >> 5;

        for (unsigned ii = warpId; ii < numNodes; ii += warpsPerBlock) {
            if (d_reverseEdgeStatus[reverseEdgeOffset + ii] != INIT)
                continue;

            unsigned     p    = d_reverseEdges[reverseEdgeOffset + ii];
            const float* pVec = reinterpret_cast<const float*>(
                d_graph + p * FreshVamana::Consts::graph_entry_size_g);

            // cooperative distance computation
            float partial = 0.0f;
            for (unsigned j = laneId; j < FreshVamana::Consts::D_g; j += 32) {
                float diff = pVec[j] - pStarVec[j];
                partial    = fmaf(diff, diff, partial);
            }

            // warp reduce sum
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (laneId == 0) {
                float queryDist = d_reverseEdgeDists[reverseEdgeOffset + ii];
                if (partial * alpha <= queryDist) {
                    d_reverseEdgeStatus[reverseEdgeOffset + ii] = PRUNED;
                }
            }
        }
    }
}

void computeReverseEdges(uint8_t* d_graph, uint8_t* d_reverseEdgeIndex, float alpha) {
    // TODO: Replace queryVecs with something better. It seems like a good idea to do this pruning
    // in batches
    float* d_queryVecs;

    gpuErrchk(cudaMalloc(&d_queryVecs,
                         FreshVamana::Consts::N_g * FreshVamana::Consts::D_g * sizeof(float)));

    unsigned*  d_reverseEdges;
    unsigned*  d_reverseEdgeCount;
    float*     d_reverseEdgeDists;
    unsigned*  d_reverseEdgesAux;
    float*     d_reverseEdgeDistsAux;
    NodeState* d_reverseEdgeStatus;

    gpuErrchk(cudaMalloc(&d_reverseEdges,
                         FreshVamana::Consts::N_g * FreshVamana::Consts::max_reverse_index_entries *
                             sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeCount, FreshVamana::Consts::N_g * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(
        &d_reverseEdgeDists,
        FreshVamana::Consts::N_g * FreshVamana::Consts::max_reverse_index_entries * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_reverseEdgesAux,
                         FreshVamana::Consts::N_g * FreshVamana::Consts::max_reverse_index_entries *
                             sizeof(unsigned)));
    gpuErrchk(cudaMalloc(
        &d_reverseEdgeDistsAux,
        FreshVamana::Consts::N_g * FreshVamana::Consts::max_reverse_index_entries * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeStatus,
                         FreshVamana::Consts::N_g * FreshVamana::Consts::max_reverse_index_entries *
                             sizeof(NodeState)));

    unsigned* d_neighbors;
    unsigned* d_neighborsCount;
    float*    d_neighborDists;
    unsigned* d_neighborsAux;
    float*    d_neighborDistsAux;

    gpuErrchk(
        cudaMalloc(&d_neighbors,
                   FreshVamana::Consts::N_g * (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, FreshVamana::Consts::N_g * sizeof(unsigned)));
    gpuErrchk(
        cudaMalloc(&d_neighborDists,
                   FreshVamana::Consts::N_g * (FreshVamana::Consts::R_g + 1) * sizeof(float)));
    gpuErrchk(
        cudaMalloc(&d_neighborsAux,
                   FreshVamana::Consts::N_g * (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)));
    gpuErrchk(
        cudaMalloc(&d_neighborDistsAux,
                   FreshVamana::Consts::N_g * (FreshVamana::Consts::R_g + 1) * sizeof(float)));

    // bool nextIter;
    // bool *d_nextIter;

    // gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    loadQueryVecs<<<FreshVamana::Consts::N_g, FreshVamana::Consts::D_g>>>(d_graph, d_queryVecs);

    unsigned* degreeSum;
    gpuErrchk(cudaMalloc(&degreeSum,
                         (FreshVamana::Consts::max_reverse_index_entries + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMemset(
        degreeSum, 0, (FreshVamana::Consts::max_reverse_index_entries + 1) * sizeof(unsigned)));

    parseReverseIndex<<<FreshVamana::Consts::N_g, 1024>>>(
        d_reverseEdgeIndex, d_reverseEdges, d_reverseEdgeCount, degreeSum);

    // unsigned h_degreeCounts[MAX_REVERSE_INDEX_ENTRIES + 1];
    // cudaMemcpy(&h_degreeCounts, degreeSum, (MAX_REVERSE_INDEX_ENTRIES + 1) * sizeof(unsigned),
    // cudaMemcpyDeviceToHost);

    // for (int i = 0; i <= MAX_REVERSE_INDEX_ENTRIES; i++) {
    //     if (h_degreeCounts[i] != 0)
    //         printf("%d:%d ", i, h_degreeCounts[i]);
    // }
    // printf("\n");

    unsigned  h_queryCount;
    unsigned *d_queryIDs, *d_queryCount;
    gpuErrchk(cudaMalloc(&d_queryIDs, FreshVamana::Consts::N_g * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_queryCount, sizeof(unsigned)));
    const int numThreads = 32;
    getPrunableQueryIDs<<<(FreshVamana::Consts::N_g + numThreads - 1) / numThreads, numThreads>>>(
        d_reverseEdgeCount, d_queryIDs, d_queryCount);
    cudaMemcpy(&h_queryCount, d_queryCount, sizeof(unsigned), cudaMemcpyDeviceToHost);

    // printf("%d\n", h_queryCount)

    // Can't use 8*MAX_REVERSE_INDEX_ENTRIES because it exceeds block size limit
    computeDists<<<FreshVamana::Consts::N_g, 1024>>>(
        d_graph,
        d_reverseEdges,
        d_reverseEdgeCount,
        d_queryVecs,
        d_reverseEdgeDists,
        FreshVamana::Consts::max_reverse_index_entries);

    // sortByDistance<<<N, MAX_REVERSE_INDEX_ENTRIES,
    sortByDistance<<<FreshVamana::Consts::N_g,
                     1024,
                     FreshVamana::Consts::max_reverse_index_entries * sizeof(unsigned)>>>(
        d_reverseEdges,
        d_reverseEdgeCount,
        d_reverseEdgeDists,
        d_reverseEdgesAux,
        d_reverseEdgeDistsAux,
        FreshVamana::Consts::max_reverse_index_entries);

    getNeighbors<<<FreshVamana::Consts::N_g, FreshVamana::Consts::R_g>>>(
        d_graph, 0, d_neighbors, d_neighborsCount);

    computeDists<<<FreshVamana::Consts::N_g, FreshVamana::Consts::R_g * 8>>>(
        d_graph,
        d_neighbors,
        d_neighborsCount,
        d_queryVecs,
        d_neighborDists,
        (FreshVamana::Consts::R_g + 1));

    sortByDistance<<<FreshVamana::Consts::N_g,
                     FreshVamana::Consts::R_g + 1,
                     (FreshVamana::Consts::R_g + 1) * sizeof(unsigned)>>>(
        d_neighbors,
        d_neighborsCount,
        d_neighborDists,
        d_neighborsAux,
        d_neighborDistsAux,
        FreshVamana::Consts::R_g + 1);

    mergeIntoReverseEdges<<<FreshVamana::Consts::N_g, 1024>>>(
        d_reverseEdgeCount,
        // mergeIntoReverseEdges<<<N, R+MAX_REVERSE_INDEX_ENTRIES>>>(d_reverseEdgeCount,
        d_reverseEdges,
        d_reverseEdgeDists,
        d_neighborsCount,
        d_neighbors,
        d_neighborDists);

    // unsigned iter = 0;

    h_queryCount = FreshVamana::Consts::N_g;
    pruneReverseEdges<<<h_queryCount, 32>>>(d_graph,
                                            d_queryIDs,
                                            d_reverseEdges,
                                            d_reverseEdgeCount,
                                            d_reverseEdgeDists,
                                            d_reverseEdgeStatus,
                                            d_queryVecs,
                                            alpha);

    gpuErrchk(cudaFree(d_queryVecs));

    gpuErrchk(cudaFree(d_reverseEdges));
    gpuErrchk(cudaFree(d_reverseEdgeCount));
    gpuErrchk(cudaFree(d_reverseEdgeDists));
    gpuErrchk(cudaFree(d_reverseEdgesAux));
    gpuErrchk(cudaFree(d_reverseEdgeDistsAux));
    gpuErrchk(cudaFree(d_reverseEdgeStatus));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    // printf("Reverse edge pruning finished in %d iterations.\n", iter);
}
