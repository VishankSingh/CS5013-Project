// // ==============================
// // graphIndexT.cuh
// // ==============================
// #pragma once
// #include "graphT.cuh"
// #include <cstdio>
// #include <cstdlib>

// #ifdef __CUDACC__
// #include <cuda_runtime.h>
// #endif

// template <typename T__>
// struct GraphIndexT {
//     using GraphEntry = GraphT<T__>;

//     GraphEntry *entries = nullptr; // array of graph entries
//     uint8_t *blob = nullptr;       // raw binary blob
//     size_t num_nodes = 0;

//     GraphIndexT() = default;

//     // Construct from binary file (host memory)
//     GraphIndexT(const char *filename) {
//         using namespace FreshVamana::Consts;

//         FILE *fp = fopen(filename, "rb");
//         if (!fp) {
//             perror("fopen");
//             return;
//         }

//         fseek(fp, 0, SEEK_END);
//         long file_size = ftell(fp);
//         fseek(fp, 0, SEEK_SET);

//         blob = static_cast<uint8_t *>(malloc(file_size));
//         if (!blob) {
//             perror("malloc");
//             fclose(fp);
//             return;
//         }

//         size_t bytes_read = fread(blob, 1, file_size, fp);
//         fclose(fp);
//         if (bytes_read != (size_t)file_size) {
//             fprintf(stderr, "File read mismatch\n");
//             free(blob);
//             blob = nullptr;
//             return;
//         }

//         num_nodes = file_size / graph_entry_size_in_bytes;
//         entries = static_cast<GraphEntry *>(malloc(num_nodes * sizeof(GraphEntry)));

//         for (size_t i = 0; i < num_nodes; ++i) {
//             uint8_t *base_ptr = blob + i * graph_entry_size_in_bytes;
//             entries[i] = GraphEntry::from_blob(base_ptr);
//         }
//     }

//     // Load graph directly into unified or device memory
// #ifdef __CUDACC__
//     static GraphIndexT load_to_device(const char *filename, bool unified = false) {
//         using namespace FreshVamana::Consts;

//         GraphIndexT gindex;
//         FILE *fp = fopen(filename, "rb");
//         if (!fp) {
//             perror("fopen");
//             return gindex;
//         }

//         fseek(fp, 0, SEEK_END);
//         long file_size = ftell(fp);
//         fseek(fp, 0, SEEK_SET);

//         uint8_t *host_blob = static_cast<uint8_t *>(malloc(file_size));
//         fread(host_blob, 1, file_size, fp);
//         fclose(fp);

//         gindex.num_nodes = file_size / graph_entry_size_in_bytes;

//         if (unified) {
//             cudaMallocManaged(&gindex.blob, file_size);
//             cudaMallocManaged(&gindex.entries, gindex.num_nodes * sizeof(GraphEntry));
//         } else {
//             cudaMalloc(&gindex.blob, file_size);
//             cudaMallocHost(&gindex.entries, gindex.num_nodes * sizeof(GraphEntry));
//         }

//         cudaMemcpy(gindex.blob, host_blob, file_size, cudaMemcpyHostToDevice);

//         for (size_t i = 0; i < gindex.num_nodes; ++i) {
//             uint8_t *base_ptr = gindex.blob + i * graph_entry_size_in_bytes;
//             gindex.entries[i] = GraphEntry::from_blob(base_ptr);
//         }

//         free(host_blob);
//         return gindex;
//     }
// #endif

//     ~GraphIndexT() {
// #ifdef __CUDACC__
//         cudaFree(blob);
//         cudaFree(entries);
// #else
//         if (entries) free(entries);
//         if (blob) free(blob);
// #endif
//     }
// };
