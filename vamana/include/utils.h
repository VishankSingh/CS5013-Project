#pragma once

#include "constants.h"
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
auto callWithLog(const char* func_name, const char* file, int line, const char* caller, Func&& func,
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
void logHostError(const char* file, int line, const char* caller, std::ostream& os,
                  const std::string& msg, Args&&... args) {
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

//==================================================================================================
[[nodiscard]] bool read_bin(const std::filesystem::path& bin_path, uint8_t* buffer,
                            size_t number_of_entries, size_t entry_size) {
    FILE* bin_file = fopen(bin_path.c_str(), "rb");
    if (!bin_file) {
        LOG_HOST_ERROR(std::cerr, "Failed to open bin file ", bin_path.c_str());
        // printf("Could not open bin file.\n");
        return false;
    }
    fread(buffer, entry_size, number_of_entries, bin_file);
    fclose(bin_file);
    return true;
}

// don't mark noexcept
[[nodiscard]] std::unique_ptr<Graph_t<dtype_g, D_g, R_g>> init_graph(
    const std::filesystem::path& graph_bin_path, const std::filesystem::path& basepoints_bin_path) {
    auto graph = std::make_unique<Graph_t<dtype_g, D_g, R_g>>();

    graph->h_graph_size     = graph_size_g;
    graph->h_graph_capacity = graph_size_g;
    graph->h_graph          = (uint8_t*)std::calloc(graph_size_g, graph->get_graph_entry_size());
    if (!read_bin(graph_bin_path, graph->h_graph, N_g, graph->get_graph_entry_size())) {
        return nullptr;
    }

    graph->d_graph_size     = N_g;
    graph->d_graph_capacity = N_g;

    return nullptr;
}
