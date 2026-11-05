// Element-wise operations: these choose a kernel and split work across threads.
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "kernels/kernels.hpp"
#include "minitensor/tensor.hpp"
#include "nd_loop.hpp"

namespace minitensor::detail {
namespace {

// worth_waking_threads() counts work in elements of addition. Reductions read
// one or two arrays and write nothing, so they get through more elements in the
// same time.
constexpr std::size_t kSumElementsPerAddition = 4;
constexpr std::size_t kDotElementsPerAddition = 2;

// Lines up an operand's strides with the output's last dimensions, the way
// broadcasting lines up shapes. Stretched dimensions get a stride of 0, so the
// walk stays on the same element.
template <typename T>
Strides broadcast_strides(const View<const T>& operand, std::size_t rank) noexcept {
    Strides strides{};
    const std::size_t missing = rank - operand.rank;
    for (std::size_t d = 0; d < operand.rank; ++d) {
        strides[missing + d] = operand.dims[d] == 1 ? 0 : operand.strides[d];
    }
    return strides;
}

// Gives each piece of the array its own slot for the result. All the threads
// finish at about the same moment, so sharing one lock instead made them queue
// and sleep, which took longer than the arithmetic.
template <typename T, typename Reduce, typename Combine>
T parallel_reduce(std::size_t size, std::size_t work, Reduce&& reduce, Combine&& combine) {
    if (!worth_waking_threads(work)) return reduce(std::size_t{0}, size);

    const std::size_t pieces = std::min(static_cast<std::size_t>(num_threads()), size);
    std::vector<T> results(pieces);
    for_each_chunk(pieces, true, [&](std::size_t first, std::size_t last) {
        for (std::size_t piece = first; piece < last; ++piece) {
            const std::size_t begin = size * piece / pieces;
            const std::size_t end = size * (piece + 1) / pieces;
            results[piece] = reduce(begin, end - begin);
        }
    });

    T result = results[0];
    for (std::size_t piece = 1; piece < pieces; ++piece) result = combine(result, results[piece]);
    return result;
}

// True if `view` has the same dimensions as `out`, and its elements are next to
// each other in row-major order.
template <typename T, typename U>
bool is_flat_like(const View<T>& view, const View<U>& out) noexcept {
    if (view.rank != out.rank) return false;
    std::ptrdiff_t expected = 1;
    for (std::size_t d = out.rank; d-- > 0;) {
        if (view.dims[d] != out.dims[d] || view.strides[d] != expected) return false;
        expected *= static_cast<std::ptrdiff_t>(out.dims[d]);
    }
    return true;
}

}  // namespace

template <typename T>
void binary_into(BinaryOp op, View<T> out, View<const T> a, View<const T> b) {
    const KernelSet<T>& set = kernels<T>();
    const int index = static_cast<int>(op);

    // Most calls are on contiguous tensors of one shape, which need no planning.
    if (is_flat_like(out, out) && is_flat_like(a, out) && is_flat_like(b, out)) {
        std::size_t total = 1;
        for (std::size_t d = 0; d < out.rank; ++d) total *= out.dims[d];
        for_each_chunk(total, worth_waking_threads(total), [&](std::size_t begin, std::size_t end) {
            set.binary_vv[index](a.data + begin, b.data + begin, out.data + begin, end - begin);
        });
        return;
    }

    const Strides strides_a = broadcast_strides(a, out.rank);
    const Strides strides_b = broadcast_strides(b, out.rank);
    const NdLoop<3> loop =
        make_loop<3>(out.rank, out.dims, {out.strides, strides_a.data(), strides_b.data()});
    if (loop.total == 0) return;

    const std::ptrdiff_t so = loop.inner_stride(0);
    const std::ptrdiff_t sa = loop.inner_stride(1);
    const std::ptrdiff_t sb = loop.inner_stride(2);

    run_loop(loop, worth_waking_threads(loop.total),
             [&](const std::array<std::ptrdiff_t, 3>& at, std::size_t count) {
                 T* target = out.data + at[0];
                 const T* left = a.data + at[1];
                 const T* right = b.data + at[2];
                 if (so == 1 && sa == 1 && sb == 1) {
                     set.binary_vv[index](left, right, target, count);
                 } else if (so == 1 && sa == 1 && sb == 0) {
                     set.binary_vs[index](left, *right, target, count);
                 } else if (so == 1 && sa == 0 && sb == 1) {
                     set.binary_sv[index](*left, right, target, count);
                 } else {
                     set.binary_strided[index](left, sa, right, sb, target, so, count);
                 }
             });
}

template <typename T>
void copy_into(View<T> out, View<const T> source) {
    const Strides strides = broadcast_strides(source, out.rank);
    const NdLoop<2> loop = make_loop<2>(out.rank, out.dims, {out.strides, strides.data()});
    if (loop.total == 0) return;

    const KernelSet<T>& set = kernels<T>();
    const std::ptrdiff_t so = loop.inner_stride(0);
    const std::ptrdiff_t ss = loop.inner_stride(1);

    run_loop(loop, worth_waking_threads(loop.total),
             [&](const std::array<std::ptrdiff_t, 2>& at, std::size_t count) {
                 T* target = out.data + at[0];
                 const T* from = source.data + at[1];
                 if (so == 1 && ss == 1) {
                     std::memmove(target, from, count * sizeof(T));
                 } else if (so == 1 && ss == 0) {
                     set.fill(target, *from, count);
                 } else {
                     set.copy_strided(from, ss, target, so, count);
                 }
             });
}

template <typename T>
void fill_into(View<T> out, T value) {
    const NdLoop<1> loop = make_loop<1>(out.rank, out.dims, {out.strides});
    if (loop.total == 0) return;

    const KernelSet<T>& set = kernels<T>();
    const std::ptrdiff_t so = loop.inner_stride(0);

    run_loop(loop, worth_waking_threads(loop.total),
             [&](const std::array<std::ptrdiff_t, 1>& at, std::size_t count) {
                 if (so == 1) {
                     set.fill(out.data + at[0], value, count);
                 } else {
                     set.copy_strided(&value, 0, out.data + at[0], so, count);
                 }
             });
}

template <typename T>
T reduce_sum(const T* data, std::size_t size) {
    const KernelSet<T>& set = kernels<T>();
    return parallel_reduce<T>(
        size, size / kSumElementsPerAddition,
        [&](std::size_t begin, std::size_t count) { return set.sum(data + begin, count); },
        [](T x, T y) { return x + y; });
}

template <typename T>
T reduce_min(const T* data, std::size_t size) {
    const KernelSet<T>& set = kernels<T>();
    return parallel_reduce<T>(
        size, size / kSumElementsPerAddition,
        [&](std::size_t begin, std::size_t count) { return set.min(data + begin, count); },
        [](T x, T y) { return y < x ? y : x; });
}

template <typename T>
T reduce_max(const T* data, std::size_t size) {
    const KernelSet<T>& set = kernels<T>();
    return parallel_reduce<T>(
        size, size / kSumElementsPerAddition,
        [&](std::size_t begin, std::size_t count) { return set.max(data + begin, count); },
        [](T x, T y) { return x < y ? y : x; });
}

template <typename T>
T dot_product(const T* a, const T* b, std::size_t size) {
    const KernelSet<T>& set = kernels<T>();
    return parallel_reduce<T>(
        size, size / kDotElementsPerAddition,
        [&](std::size_t begin, std::size_t count) { return set.dot(a + begin, b + begin, count); },
        [](T x, T y) { return x + y; });
}

#define MINITENSOR_INSTANTIATE(T)                                                  \
    template void binary_into<T>(BinaryOp, View<T>, View<const T>, View<const T>); \
    template void copy_into<T>(View<T>, View<const T>);                            \
    template void fill_into<T>(View<T>, T);                                        \
    template T reduce_sum<T>(const T*, std::size_t);                               \
    template T reduce_min<T>(const T*, std::size_t);                               \
    template T reduce_max<T>(const T*, std::size_t);                               \
    template T dot_product<T>(const T*, const T*, std::size_t);

MINITENSOR_INSTANTIATE(float)
MINITENSOR_INSTANTIATE(double)
MINITENSOR_INSTANTIATE(std::int32_t)
MINITENSOR_INSTANTIATE(std::int64_t)

#undef MINITENSOR_INSTANTIATE

}  // namespace minitensor::detail
