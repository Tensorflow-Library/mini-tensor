// Thread settings and the choice of SIMD kernels.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
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

// Worker threads go to sleep after about 2 ms without work, and waking them takes
// much longer than using them awake: about 210 us against 13 us in
// docs/EFFICIENCY.md. So medium-sized work only uses them when calls keep
// arriving. One call stays on this thread, and a loop soon moves to all of them.
constexpr std::size_t kMinThreadedElements = std::size_t{1} << 16;
constexpr std::size_t kAlwaysThreadedElements = std::size_t{1} << 19;
constexpr int kCallsBeforeWaking = 3;
constexpr std::int64_t kThreadsStayAwakeNs = 2000000;

std::atomic<std::int64_t> g_last_call_ns{0};
std::atomic<int> g_calls_in_a_row{0};

}  // namespace

bool worth_waking_threads(std::size_t elements) noexcept {
    if (elements < kMinThreadedElements) return false;

    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const std::int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    const std::int64_t previous_ns = g_last_call_ns.exchange(now_ns, std::memory_order_relaxed);

    if (elements >= kAlwaysThreadedElements) {
        g_calls_in_a_row.store(kCallsBeforeWaking, std::memory_order_relaxed);
        return true;
    }
    if (now_ns - previous_ns > kThreadsStayAwakeNs) {
        g_calls_in_a_row.store(0, std::memory_order_relaxed);
        return false;
    }
    const int calls = g_calls_in_a_row.load(std::memory_order_relaxed);
    if (calls >= kCallsBeforeWaking) return true;
    g_calls_in_a_row.store(calls + 1, std::memory_order_relaxed);
    return false;
}

void start_threads_once() noexcept {
#ifdef _OPENMP
    static const bool started = [] {
        const int threads = num_threads();
        // The compiler removes an empty parallel block, so this needs a real effect.
        std::atomic<int> arrived{0};
#pragma omp parallel num_threads(threads)
        arrived.fetch_add(1, std::memory_order_relaxed);
        return arrived.load(std::memory_order_relaxed) > 0;
    }();
    (void)started;
#endif
}

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
