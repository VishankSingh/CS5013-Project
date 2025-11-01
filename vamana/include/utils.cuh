#pragma once
#include "constants.cuh"
#include "graph.cuh"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

// uncomment to log the function calls
// #define logfuncs

template <typename Func>
auto callWithLog(const char*   func_name,
                 const char*   file,
                 int           line,
                 const char*   caller,
                 Func&&        func,
                 std::ostream& os = std::cout) {
    constexpr const char* cyan  = "\033[36m";
    constexpr const char* reset = "\033[0m";

    os << cyan << "[" << file << ":" << line << ":" << caller << "]" << reset << " Calling " << cyan
       << func_name << reset << '\n';
    os.flush();

    if constexpr (std::is_void_v<decltype(func())>)
        func();
    else
        return func();
}

#if defined(logfuncs)
#define CALL_WITH_LOG(fn_call, ...) \
    callWithLog(#fn_call, __FILE__, __LINE__, __func__, [&]() { return fn_call; }, ##__VA_ARGS__)
#else
#define CALL_WITH_LOG(fn_call, ...) fn_call
#endif

template <typename... Args>
void logHostError(const char*        file,
                  int                line,
                  const char*        caller,
                  std::ostream&      os,
                  const std::string& msg,
                  Args&&... args) {
    constexpr const char* red   = "\033[31m";
    constexpr const char* reset = "\033[0m";

    os << red << "[ERROR] [" << file << ":" << line << ":" << caller << "] " << reset;

    std::ostringstream oss;
    oss << msg;
    ((oss << ' ' << std::forward<Args>(args)), ...);

    os << oss.str() << '\n';
    os.flush();
}

#define LOG_HOST_ERROR(stream, msg, ...) \
    logHostError(__FILE__, __LINE__, __func__, stream, msg, ##__VA_ARGS__)

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

//==================================================================================================

[[nodiscard]] inline bool readBin(const std::filesystem::path& bin_path,
                                  uint8_t*                     buffer,
                                  size_t                       number_of_entries,
                                  size_t                       entry_size) {
    FILE* bin_file = fopen(bin_path.c_str(), "rb");
    if (!bin_file) {
        LOG_HOST_ERROR(std::cerr, "Failed to open bin file ", bin_path.c_str());
        // printf("Could not open bin file.\n");
        return false;
    }

    // constexpr size_t                      IO_BUFFER_SIZE = 4 * 1024 * 1024;
    // static thread_local std::vector<char> io_buffer(IO_BUFFER_SIZE);
    // setvbuf(bin_file, io_buffer.data(), _IOFBF, IO_BUFFER_SIZE);

    const size_t read_count = fread(buffer, entry_size, number_of_entries, bin_file);
    fclose(bin_file);

    if (read_count != number_of_entries) {
        LOG_HOST_ERROR(std::cerr, "Incomplete read:", read_count, "of", number_of_entries);
        return false;
    }
    return true;
}

using namespace FreshVamana::Consts;

/*
The returned graph should have d_graph correctly populated
*/
// don't mark noexcept
template <typename T__>
[[nodiscard]] inline std::unique_ptr<GraphT<T__>> initGraph(
    const std::filesystem::path& graph_bin_path) {
    using GraphType = GraphT<T__>;

    auto graph = std::make_unique<GraphType>();

    uint8_t* h_graph = (uint8_t*)std::calloc(N_g, graph->getGraphEntrySize());
    if (!readBin(graph_bin_path, h_graph, rg_bin_size_g, graph->getGraphEntrySize())) {
        return nullptr;
    }

    graph->d_graph_capacity = N_g;
    graph->d_graph_size     = rg_bin_size_g;

    gpuErrchk(cudaMalloc(&graph->d_graph, N_g * graph->getGraphEntrySize()));
    // TODO: complete this
    gpuErrchk(cudaMemcpy(graph->d_graph, h_graph, rg_bin_size_g * graph->getGraphEntrySize(),
                         cudaMemcpyHostToDevice));

    free(h_graph);
    std::cout << "[initGraph] Graph initialized: "
              << "N=" << N_g << ", D=" << FreshVamana::Consts::D_g
              << ", R=" << FreshVamana::Consts::R_g
              << ", EntrySize=" << GraphType::getGraphEntrySize() << " bytes.\n";

    return graph;
}

__global__ void computeDists(GraphT<FreshVamana::Consts::dtype_g>& graph,
                             unsigned*                             d_nodes,
                             unsigned*                             d_nodeCount,
                             float*                                d_queryVecs,
                             float*                                d_dists,
                             unsigned                              rowSize) {
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
        float*   nodeVec = (float*)(graph.d_graph + FreshVamana::Consts::graph_entry_size_g *
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
