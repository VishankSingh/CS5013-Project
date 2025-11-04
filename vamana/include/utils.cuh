#pragma once
#include "constants.cuh"
#include "globals.cuh"
#include "graphT.cuh"

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
    // graph->d_graph_capacity = rg_bin_size_g * 1.3;
    // graph->d_graph_size     = rg_bin_size_g;

    FreshVamana::Globals::d_graph_capacity = rg_bin_size_g * 1.3;
    FreshVamana::Globals::d_graph_size     = rg_bin_size_g;

    uint8_t* h_graph = (uint8_t*)std::calloc(FreshVamana::Globals::d_graph_capacity,
                                             FreshVamana::Consts::graph_entry_bytes_g);
    if (!readBin(graph_bin_path,
                 h_graph,
                 FreshVamana::Globals::d_graph_size,
                 FreshVamana::Consts::graph_entry_bytes_g)) {
        return nullptr;
    }

    gpuErrchk(
        cudaMalloc(&graph->d_graph,
                   FreshVamana::Globals::d_graph_size * FreshVamana::Consts::graph_entry_bytes_g));
    // TODO: complete this
    gpuErrchk(cudaMemcpy(graph->d_graph,
                         h_graph,
                         rg_bin_size_g * FreshVamana::Consts::graph_entry_bytes_g,
                         cudaMemcpyHostToDevice));

    free(h_graph);
    std::cout << "[ initGraph ] Graph initialized: "
              << "N=" << FreshVamana::Globals::d_graph_size << ", D=" << FreshVamana::Consts::D_g
              << ", R=" << FreshVamana::Consts::R_g
              << ", EntrySize=" << FreshVamana::Consts::graph_entry_bytes_g << " bytes.\n";

    return graph;
}
