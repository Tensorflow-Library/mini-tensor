// Cache-aware matrix multiplication, in the style of the GotoBLAS and BLIS
// libraries. See docs/DESIGN.md for an explanation of the algorithm.
//
// To compute C += A * B, the matrices are cut into pieces that fit each level
// of the CPU's cache. The pieces are copied ("packed") into buffers laid out so
// the micro-kernel can read them in a straight line:
//
//   for each block of up to NC rows of A and NC columns of B    (fits L3)
//     for each block of KC columns of A and rows of B
//       pack the A block into panels of MR rows
//       pack the B block into panels of NR columns
//       for each tile of A panels and B panels, on any thread   (fits L2)
//         for each A panel in the tile                          (fits L1)
//           for each B panel in the tile
//             micro-kernel: update MR x NR entries of C
//
// C is stored by rows, so its columns are what SIMD registers can load at once.
// That's why this is the mirror image of BLIS, which stores C by columns.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

#include "kernels/kernels.hpp"
#include "minitensor/tensor.hpp"
#include "nd_loop.hpp"

namespace minitensor::detail {
namespace {

// Below this many multiplications, packing costs more than it saves.
constexpr double kSmallWork = 32.0 * 32.0 * 32.0;
// worth_waking_threads() counts work in elements of addition. One of those takes
// about as long as this many multiplications here.
constexpr double kMultiplicationsPerAddition = 14.0;
// Threads take tiles from a shared list. Having several tiles per thread means
// a thread held up by something else on the machine doesn't hold up the others.
constexpr double kTilesPerThread = 4.0;

struct CacheSizes {
    std::size_t l1;
    std::size_t l2;
    std::size_t l3;
};

// Reads a cache size in bytes from `variable`, then from the OS. The variables
// help on virtual machines that report wrong sizes, and let the tests force
// matrices to be split into many blocks.
std::size_t cache_size(const char* variable, int query, std::size_t fallback) noexcept {
    if (const char* text = std::getenv(variable)) {
        const long long size = std::atoll(text);
        if (size > 0) return static_cast<std::size_t>(size);
    }
#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
    const long size = sysconf(query);
    if (size > 0) return static_cast<std::size_t>(size);
#else
    (void)query;
#endif
    return fallback;
}

CacheSizes detect_cache_sizes() noexcept {
#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
    const int l1 = _SC_LEVEL1_DCACHE_SIZE;
    const int l2 = _SC_LEVEL2_CACHE_SIZE;
    const int l3 = _SC_LEVEL3_CACHE_SIZE;
#else
    const int l1 = 0;
    const int l2 = 0;
    const int l3 = 0;
#endif
    return {cache_size("MINITENSOR_L1_CACHE", l1, 32 << 10),
            cache_size("MINITENSOR_L2_CACHE", l2, 256 << 10),
            cache_size("MINITENSOR_L3_CACHE", l3, 8 << 20)};
}

std::size_t round_up(std::size_t value, std::size_t multiple) noexcept {
    return (value + multiple - 1) / multiple * multiple;
}

struct BlockSizes {
    std::size_t kc;
    std::size_t mc;
    std::size_t nc;
};

// Sizes each block to take about half of its cache level, leaving room for the
// other data the loops touch.
template <typename T>
BlockSizes block_sizes(const KernelSet<T>& set) noexcept {
    static const CacheSizes caches = detect_cache_sizes();
    const std::size_t kc = std::clamp<std::size_t>(caches.l1 / 2 / (set.mr * sizeof(T)), 32, 512);
    std::size_t mc = caches.l2 / 2 / (kc * sizeof(T));
    mc = std::max(mc - mc % set.nr, 2 * set.nr);
    std::size_t nc = caches.l3 / 2 / (kc * sizeof(T));
    nc = std::max(nc - nc % (set.mr * set.nr), 8 * set.mr * set.nr);
    return {kc, mc, nc};
}

// How one block is divided into tiles.
struct TileGrid {
    std::size_t rows;     // Groups of A panels.
    std::size_t columns;  // Groups of B panels.
};

// Square tiles reuse the most data per byte brought into L2, so aim for those,
// without letting the part of B in a tile outgrow L2.
TileGrid choose_tiles(std::size_t rows, std::size_t cols, std::size_t a_panels,
                      std::size_t b_panels, std::size_t mc, std::size_t nr, int threads) noexcept {
    if (threads <= 1) return {1, (cols + mc - 1) / mc};
    const double tiles = kTilesPerThread * threads;
    const double side = std::sqrt(static_cast<double>(rows) * static_cast<double>(cols) / tiles);
    const std::size_t tile_cols =
        std::clamp<std::size_t>(round_up(static_cast<std::size_t>(side), nr), nr, mc);
    const std::size_t columns = std::min((cols + tile_cols - 1) / tile_cols, b_panels);
    const auto wanted_rows =
        static_cast<std::size_t>(std::ceil(tiles / static_cast<double>(columns)));
    return {std::min(wanted_rows, a_panels), columns};
}

// A buffer kept for the life of the thread, so repeated multiplications don't
// pay for allocation and page faults every time.
class ScratchBuffer {
public:
    ScratchBuffer() = default;
    ScratchBuffer(const ScratchBuffer&) = delete;
    ScratchBuffer& operator=(const ScratchBuffer&) = delete;
    ~ScratchBuffer() { aligned_free(data_); }

    template <typename T>
    T* get(std::size_t count) {
        const std::size_t bytes = count * sizeof(T);
        if (bytes > capacity_) {
            aligned_free(data_);
            data_ = nullptr;
            capacity_ = 0;
            data_ = aligned_malloc(bytes);
            capacity_ = bytes;
        }
        return static_cast<T*>(data_);
    }

private:
    void* data_ = nullptr;
    std::size_t capacity_ = 0;
};

thread_local ScratchBuffer t_packed_a;
thread_local ScratchBuffer t_packed_b;

}  // namespace

template <typename T>
void gemm(std::size_t m, std::size_t n, std::size_t k, const T* a, std::ptrdiff_t rs_a,
          std::ptrdiff_t cs_a, const T* b, std::ptrdiff_t rs_b, std::ptrdiff_t cs_b, T* c,
          std::ptrdiff_t rs_c) {
    if (m == 0 || n == 0 || k == 0) return;

    const KernelSet<T>& set = kernels<T>();
    const double work = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    if (work <= kSmallWork) {
        set.small_gemm(m, n, k, a, rs_a, cs_a, b, rs_b, cs_b, c, rs_c);
        return;
    }

    const std::size_t mr = set.mr;
    const std::size_t nr = set.nr;
    const BlockSizes blocks = block_sizes(set);

    int threads = 1;
#ifdef _OPENMP
    const double additions = std::min(work / kMultiplicationsPerAddition, 1e18);
    if (worth_waking_threads(static_cast<std::size_t>(additions))) threads = num_threads();
#endif

    T* packed_a = t_packed_a.get<T>(round_up(std::min(m, blocks.nc), mr) * blocks.kc);
    T* packed_b = t_packed_b.get<T>(round_up(std::min(n, blocks.nc), nr) * blocks.kc);

#pragma omp parallel num_threads(threads) if (threads > 1)
    {
        alignas(kAlignment) T edge[kMaxMr * kMaxNr];

        for (std::size_t ic = 0; ic < m; ic += blocks.nc) {
            const std::size_t rows = std::min(blocks.nc, m - ic);
            const std::size_t a_panels = (rows + mr - 1) / mr;

            for (std::size_t jc = 0; jc < n; jc += blocks.nc) {
                const std::size_t cols = std::min(blocks.nc, n - jc);
                const std::size_t b_panels = (cols + nr - 1) / nr;
                const TileGrid grid =
                    choose_tiles(rows, cols, a_panels, b_panels, blocks.mc, nr, threads);

                for (std::size_t pc = 0; pc < k; pc += blocks.kc) {
                    const std::size_t kc = std::min(blocks.kc, k - pc);

#pragma omp for schedule(static) nowait
                    for (std::size_t panel = 0; panel < a_panels; ++panel) {
                        const std::size_t row = panel * mr;
                        set.pack_a(std::min(mr, rows - row), kc,
                                   a + static_cast<std::ptrdiff_t>(ic + row) * rs_a +
                                       static_cast<std::ptrdiff_t>(pc) * cs_a,
                                   rs_a, cs_a, packed_a + panel * mr * kc);
                    }

#pragma omp for schedule(static)
                    for (std::size_t panel = 0; panel < b_panels; ++panel) {
                        const std::size_t col = panel * nr;
                        set.pack_b(std::min(nr, cols - col), kc,
                                   b + static_cast<std::ptrdiff_t>(pc) * rs_b +
                                       static_cast<std::ptrdiff_t>(jc + col) * cs_b,
                                   rs_b, cs_b, packed_b + panel * nr * kc);
                    }

#pragma omp for schedule(dynamic) collapse(2)
                    for (std::size_t tile_row = 0; tile_row < grid.rows; ++tile_row) {
                        for (std::size_t tile_col = 0; tile_col < grid.columns; ++tile_col) {
                            const std::size_t first_a = a_panels * tile_row / grid.rows;
                            const std::size_t last_a = a_panels * (tile_row + 1) / grid.rows;
                            const std::size_t first_b = b_panels * tile_col / grid.columns;
                            const std::size_t last_b = b_panels * (tile_col + 1) / grid.columns;

                            for (std::size_t pa = first_a; pa < last_a; ++pa) {
                                const std::size_t row = pa * mr;
                                const std::size_t panel_rows = std::min(mr, rows - row);
                                const T* a_panel = packed_a + pa * mr * kc;
                                T* c_rows = c + static_cast<std::ptrdiff_t>(ic + row) * rs_c +
                                            static_cast<std::ptrdiff_t>(jc);

                                for (std::size_t pb = first_b; pb < last_b; ++pb) {
                                    const std::size_t col = pb * nr;
                                    const std::size_t panel_cols = std::min(nr, cols - col);
                                    const T* b_panel = packed_b + pb * nr * kc;

                                    if (panel_rows == mr && panel_cols == nr) {
                                        set.micro_kernel(kc, a_panel, b_panel, c_rows + col, rs_c);
                                        continue;
                                    }
                                    // At the edges of the matrix the kernel still
                                    // works on a full block, so it writes to a side
                                    // buffer and only the real part is kept.
                                    std::memset(edge, 0, mr * nr * sizeof(T));
                                    set.micro_kernel(kc, a_panel, b_panel, edge,
                                                     static_cast<std::ptrdiff_t>(nr));
                                    for (std::size_t r = 0; r < panel_rows; ++r) {
                                        T* row_out =
                                            c_rows + static_cast<std::ptrdiff_t>(r) * rs_c + col;
                                        for (std::size_t j = 0; j < panel_cols; ++j) {
                                            row_out[j] += edge[r * nr + j];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void gemv(std::size_t m, std::size_t n, const T* a, std::ptrdiff_t rs_a, std::ptrdiff_t cs_a,
          const T* x, std::ptrdiff_t inc_x, T* y) {
    const KernelSet<T>& set = kernels<T>();
    const bool contiguous_rows = cs_a == 1 && inc_x == 1;

    // Each row is a dot product, which takes about half as long per element as
    // addition.
    for_each_chunk(m, worth_waking_threads(m * n / 2), [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const T* row = a + static_cast<std::ptrdiff_t>(i) * rs_a;
            if (contiguous_rows) {
                y[i] = set.dot(row, x, n);
                continue;
            }
            T total = T(0);
            for (std::size_t j = 0; j < n; ++j) {
                const auto index = static_cast<std::ptrdiff_t>(j);
                total += row[index * cs_a] * x[index * inc_x];
            }
            y[i] = total;
        }
    });
}

#define MINITENSOR_INSTANTIATE(T)                                                             \
    template void gemm<T>(std::size_t, std::size_t, std::size_t, const T*, std::ptrdiff_t,    \
                          std::ptrdiff_t, const T*, std::ptrdiff_t, std::ptrdiff_t, T*,       \
                          std::ptrdiff_t);                                                    \
    template void gemv<T>(std::size_t, std::size_t, const T*, std::ptrdiff_t, std::ptrdiff_t, \
                          const T*, std::ptrdiff_t, T*);

MINITENSOR_INSTANTIATE(float)
MINITENSOR_INSTANTIATE(double)
MINITENSOR_INSTANTIATE(std::int32_t)
MINITENSOR_INSTANTIATE(std::int64_t)

#undef MINITENSOR_INSTANTIATE

}  // namespace minitensor::detail
