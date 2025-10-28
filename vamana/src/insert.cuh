#pragma once

#include "vamana.h"

// void insert_inner(uint8_t* d_graph, float* d_queryVecs, float alpha, unsigned batchStart,
//                   unsigned batchSize) {
//     unsigned* d_visitedSets;
//     unsigned* d_visitedSetCount;
//     uint8_t*  d_reverseEdgeIndex;

//     CPUTimer cputimer;

//     cputimer.Start();
//     gpuErrchk(cudaMalloc(&d_visitedSets, batchSize * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
//     gpuErrchk(cudaMalloc(&d_visitedSetCount, batchSize * sizeof(unsigned)));
//     gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, N * reverseIndexEntrySize * sizeof(uint8_t)));
//     gpuErrchk(cudaMemset(d_visitedSetCount, 0, batchSize * sizeof(unsigned)));
//     cputimer.Stop();
//     printf("insert_inner mallocs: %f sec\n", cputimer.Elapsed());

//     cputimer.Start();
//     greedySearch(d_graph, d_queryVecs, d_visitedSets, d_visitedSetCount, batchStart, batchSize);
//     cputimer.Stop();
//     printf("greedySearch: %f sec\n", cputimer.Elapsed());

//     cputimer.Start();

//     computeOutNeighbors(d_graph, d_queryVecs, d_visitedSets, d_visitedSetCount, alpha,
//                         d_reverseEdgeIndex, batchStart, batchSize);
//     cputimer.Stop();
//     printf("computeOutNeighbors: %f sec\n", cputimer.Elapsed());

//     // if (batchStart == 100000) {
//     //     uint8_t *reverseEdgeIndex = (uint8_t*)malloc(N * reverseIndexEntrySize *
//     //     sizeof(uint8_t)); cudaMemcpy(reverseEdgeIndex, d_reverseEdgeIndex, N *
//     //     reverseIndexEntrySize, cudaMemcpyDeviceToHost); FILE *outFile =
//     //     fopen("build/reverseIndex.bin", "wb"); if (!outFile) {
//     //         printf("Could not open output file.\n");
//     //         return;
//     //     }
//     //     fwrite(reverseEdgeIndex, reverseIndexEntrySize, N, outFile);
//     //     free(reverseEdgeIndex);
//     // }

//     cputimer.Start();
//     computeReverseEdges(d_graph, d_reverseEdgeIndex, alpha);
//     cputimer.Stop();
//     printf("computeReverseEdges: %f sec\n", cputimer.Elapsed());

//     cputimer.Start();
//     gpuErrchk(cudaFree(d_visitedSets));
//     gpuErrchk(cudaFree(d_visitedSetCount));
//     gpuErrchk(cudaFree(d_reverseEdgeIndex));
//     cputimer.Stop();
//     printf("insert_inner frees: %f sec\n", cputimer.Elapsed());
// }

// void insert_outer(uint8_t* d_graph, float alpha) {
//     int batchSize = 10000;

//     // uint8_t* d_graph;
//     float* queryVecs;
//     float* d_queryVecs;

//     // gpuErrchk(cudaMalloc(&d_graph, N * graphEntrySize * sizeof(uint8_t)));
//     gpuErrchk(cudaMalloc(&d_queryVecs, batchSize * D * sizeof(float)));
//     queryVecs = (float*)malloc(batchSize * D * sizeof(float));
//     // gpuErrchk(
//     //     cudaMemcpy(d_graph, graph, N * graphEntrySize * sizeof(uint8_t),
//     //     cudaMemcpyHostToDevice));

//     for (int batchStart = 0; batchStart < N; batchStart += batchSize) {
//         if (batchStart + batchSize > N)
//             batchSize = N - batchStart;
//         printf("Starting iteration: %d to %d.\n", batchStart, batchStart + batchSize - 1);

//         // Load query vectors
//         for (int i = 0; i < batchSize; i++) {
//             float* queryVec = (float*)(graph + (batchStart + i) * graphEntrySize);
//             for (int j = 0; j < D; j++) {
//                 queryVecs[i * D + j] = queryVec[j];
//             }
//         }
//         gpuErrchk(cudaMemcpy(d_queryVecs, queryVecs, batchSize * D * sizeof(float),
//                              cudaMemcpyHostToDevice));
//         insert_inner(d_graph, d_queryVecs, alpha, batchStart, batchSize);
//     }

//     // gpuErrchk(
//     //     cudaMemcpy(graph, d_graph, N * graphEntrySize * sizeof(uint8_t),
//     //     cudaMemcpyDeviceToHost));
//     // cudaFree(d_graph);
//     cudaFree(d_queryVecs);
// }