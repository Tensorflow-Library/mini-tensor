// Thread settings and the choice of SIMD kernels.
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "kernels/kernels.hpp"
#include "minitensor/runtime.hpp"

namespace minitensor {
namespace {

std::atomic<int> g_threads{0};

enum class Isa { Generic, Avx2, Avx512 };

Isa best_isa() noexcept {
#if defined(MINITENSOR_HAVE_AVX2) || defined(MINITENSOR_HAVE_AVX512)
    __builtin_cpu_init();
#endif
#if defined(MINITENSOR_HAVE_AVX512)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("fma")) {
        return Isa::Avx512;
    }
#endif
#if defined(MINITENSOR_HAVE_AVX2)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return Isa::Avx2;
#endif
    return Isa::Generic;
}

Isa choose_isa() noexcept {
    const Isa best = best_isa();
    const char* requested = std::getenv("MINITENSOR_ISA");
    if (requested == nullptr) return best;
    if (std::strcmp(requested, "generic") == 0) return Isa::Generic;
    if (std::strcmp(requested, "avx2") == 0 && best != Isa::Generic) return Isa::Avx2;
    return best;
}

Isa active_isa() noexcept {
    static const Isa isa = choose_isa();
    return isa;
}

#ifdef _OPENMP
// Counts physical cores. Hyper-threads of the same core list the same siblings.
int physical_core_count() noexcept {
    try {
        std::set<std::string> cores;
        for (int cpu = 0;; ++cpu) {
            std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                               "/topology/thread_siblings_list");
            std::string siblings;
            if (!(file >> siblings)) break;
            cores.insert(siblings);
        }
        return static_cast<int>(cores.size());
    } catch (...) {
        return 0;
    }
}

// Hyper-threads share a core's SIMD units, so they add no speed to this kind of
// work, and every thread they add must be waited for at each synchronization
// point. One thread per core is the default, unless OMP_NUM_THREADS says otherwise.
int default_thread_count() noexcept {
    const int available = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS") != nullptr) return available;
    const int cores = physical_core_count();
    return cores > 0 ? std::min(cores, available) : available;
}
#endif

}  // namespace

int num_threads() noexcept {
    const int chosen = g_threads.load(std::memory_order_relaxed);
    if (chosen > 0) return chosen;
#ifdef _OPENMP
    static const int default_count = default_thread_count();
    return std::min(default_count, omp_get_max_threads());
#else
    return 1;
#endif
}

void set_num_threads(int threads) noexcept {
    g_threads.store(threads > 0 ? threads : 0, std::memory_order_relaxed);
}

const char* simd_backend() noexcept {
    switch (active_isa()) {
        case Isa::Avx512:
            return "avx512";
        case Isa::Avx2:
            return "avx2";
        default:
            return "generic";
    }
}

namespace detail {
namespace {

template <typename T>
KernelSet<T> build_kernels() noexcept {
    KernelSet<T> set{};
    switch (active_isa()) {
#if defined(MINITENSOR_HAVE_AVX512)
        case Isa::Avx512:
            avx512::load_kernels(set);
            break;
#endif
#if defined(MINITENSOR_HAVE_AVX2)
        case Isa::Avx2:
            avx2::load_kernels(set);
            break;
#endif
        default:
            generic::load_kernels(set);
    }
    return set;
}

}  // namespace

template <typename T>
const KernelSet<T>& kernels() {
    static const KernelSet<T> set = build_kernels<T>();
    return set;
}

template const KernelSet<float>& kernels<float>();
template const KernelSet<double>& kernels<double>();
template const KernelSet<std::int32_t>& kernels<std::int32_t>();
template const KernelSet<std::int64_t>& kernels<std::int64_t>();

}  // namespace detail
}  // namespace minitensor
