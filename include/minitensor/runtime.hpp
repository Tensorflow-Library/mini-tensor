/// @file runtime.hpp
/// @brief Memory statistics and thread and SIMD settings.
#pragma once

#include <cstddef>

namespace minitensor {

/// Alignment, in bytes, of every tensor buffer. One cache line, which also
/// fits the widest SIMD register the library uses.
inline constexpr std::size_t kAlignment = 64;

/// A snapshot of the library's heap use.
struct MemoryStats {
    std::size_t live_bytes;   ///< Bytes currently allocated.
    std::size_t peak_bytes;   ///< Largest value `live_bytes` has reached.
    std::size_t allocations;  ///< Number of allocations made so far.
};

/// Returns the current memory statistics.
MemoryStats memory_stats() noexcept;

/// Returns the library's cache of freed blocks to the OS. Large freed blocks
/// are normally kept, up to 1 GiB in total, so that new tensors can reuse them
/// without page faults.
void release_cached_memory() noexcept;

/// Number of threads the library uses for large operations.
int num_threads() noexcept;

/// Sets the number of threads for large operations. Values below 1 restore
/// the default.
void set_num_threads(int threads) noexcept;

/// Name of the SIMD code in use: "avx512", "avx2" or "generic". It's chosen
/// once at startup from what the CPU supports. Setting the MINITENSOR_ISA
/// environment variable selects a lower level instead.
const char* simd_backend() noexcept;

namespace detail {

void* aligned_malloc(std::size_t bytes);
void* aligned_calloc(std::size_t bytes);
void aligned_free(void* ptr) noexcept;

// Decides whether an operation on this many elements of addition should use
// all the threads. See src/runtime.cpp.
bool worth_waking_threads(std::size_t elements) noexcept;

// Creates the thread team, the first time it's called.
void start_threads_once() noexcept;

}  // namespace detail
}  // namespace minitensor
