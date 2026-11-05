// Runs an operation over tensors of any shape and memory layout.
//
// Every element-wise operation reduces to "walk these strided arrays in step".
// make_loop() rewrites that walk into the fewest and flattest loops possible,
// and run_loop() executes it, in parallel if asked.
#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "minitensor/runtime.hpp"
#include "minitensor/shape.hpp"

namespace minitensor::detail {

// Calls body(begin, end) for pieces of [0, count), one piece per thread.
//
// The team size stays the same across calls even for small `count`, because
// libgomp destroys and recreates threads when the team size changes.
template <typename Body>
void for_each_chunk(std::size_t count, bool parallel, Body&& body) {
#ifdef _OPENMP
    const int threads = parallel && count > 1 ? num_threads() : 1;
    if (threads > 1) {
#pragma omp parallel num_threads(threads)
        {
            const auto team = static_cast<std::size_t>(omp_get_num_threads());
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const std::size_t begin = count * id / team;
            const std::size_t end = count * (id + 1) / team;
            if (begin < end) body(begin, end);
        }
        return;
    }
#else
    (void)parallel;
#endif
    if (count > 0) body(std::size_t{0}, count);
}

// A set of nested loops. Dimension 0 is the outermost. `stride[op][d]` is how
// far operand `op` moves per step of dimension `d`.
template <std::size_t Ops>
struct NdLoop {
    std::size_t rank = 0;
    std::size_t total = 1;
    std::array<std::size_t, kMaxRank> size{};
    std::array<std::array<std::ptrdiff_t, kMaxRank>, Ops> stride{};

    std::ptrdiff_t inner_stride(std::size_t op) const noexcept {
        return rank == 0 ? 0 : stride[op][rank - 1];
    }
};

template <std::size_t Ops>
NdLoop<Ops> make_loop(std::size_t rank, const std::size_t* dims,
                      const std::array<const std::ptrdiff_t*, Ops>& strides) noexcept {
    NdLoop<Ops> loop;

    // Dimensions of size 1 don't move any pointer.
    for (std::size_t d = 0; d < rank; ++d) {
        loop.total *= dims[d];
        if (dims[d] == 1) continue;
        loop.size[loop.rank] = dims[d];
        for (std::size_t op = 0; op < Ops; ++op) loop.stride[op][loop.rank] = strides[op][d];
        ++loop.rank;
    }

    // Put the smallest strides innermost, so memory is walked in order even
    // for something like a transposed matrix.
    const auto weight = [&loop](std::size_t d) {
        std::ptrdiff_t total = 0;
        for (std::size_t op = 0; op < Ops; ++op) total += std::abs(loop.stride[op][d]);
        return total;
    };
    const auto swap_dims = [&loop](std::size_t x, std::size_t y) {
        std::swap(loop.size[x], loop.size[y]);
        for (std::size_t op = 0; op < Ops; ++op) std::swap(loop.stride[op][x], loop.stride[op][y]);
    };
    for (std::size_t i = 1; i < loop.rank; ++i) {
        for (std::size_t j = i; j > 0 && weight(j - 1) < weight(j); --j) swap_dims(j - 1, j);
    }

    // Join loops that continue each other. A contiguous 1000 x 1000 matrix
    // becomes a single loop of 1,000,000 steps.
    if (loop.rank > 0) {
        std::size_t kept = 0;
        for (std::size_t d = 1; d < loop.rank; ++d) {
            bool joins = true;
            for (std::size_t op = 0; op < Ops; ++op) {
                joins = joins && loop.stride[op][kept] ==
                                     loop.stride[op][d] * static_cast<std::ptrdiff_t>(loop.size[d]);
            }
            if (joins) {
                loop.size[kept] *= loop.size[d];
            } else {
                ++kept;
                loop.size[kept] = loop.size[d];
            }
            for (std::size_t op = 0; op < Ops; ++op) loop.stride[op][kept] = loop.stride[op][d];
        }
        loop.rank = kept + 1;
    }
    return loop;
}

// Calls inner(offsets, count) for each run of the innermost loop. `offsets`
// holds each operand's position at the start of the run.
template <std::size_t Ops, typename Inner>
void run_loop(const NdLoop<Ops>& loop, bool parallel, Inner&& inner) {
    using Offsets = std::array<std::ptrdiff_t, Ops>;

    if (loop.rank <= 1) {
        // A single flat run can be split anywhere.
        const std::size_t count = loop.rank == 0 ? 1 : loop.size[0];
        for_each_chunk(count, parallel, [&](std::size_t begin, std::size_t end) {
            Offsets offsets{};
            for (std::size_t op = 0; op < Ops; ++op) {
                offsets[op] = static_cast<std::ptrdiff_t>(begin) * loop.inner_stride(op);
            }
            inner(offsets, end - begin);
        });
        return;
    }

    const std::size_t last = loop.rank - 1;
    for_each_chunk(loop.size[0], parallel, [&](std::size_t begin, std::size_t end) {
        std::array<std::size_t, kMaxRank> index{};
        index[0] = begin;
        Offsets offsets{};
        for (std::size_t op = 0; op < Ops; ++op) {
            offsets[op] = static_cast<std::ptrdiff_t>(begin) * loop.stride[op][0];
        }
        while (index[0] < end) {
            inner(offsets, loop.size[last]);
            // Step the outer loops like an odometer.
            for (std::size_t d = last; d-- > 0;) {
                ++index[d];
                for (std::size_t op = 0; op < Ops; ++op) offsets[op] += loop.stride[op][d];
                if (d == 0 || index[d] < loop.size[d]) break;
                for (std::size_t op = 0; op < Ops; ++op) {
                    offsets[op] -= static_cast<std::ptrdiff_t>(loop.size[d]) * loop.stride[op][d];
                }
                index[d] = 0;
            }
        }
    });
}

}  // namespace minitensor::detail
