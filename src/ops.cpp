// Element-wise operations: these choose a kernel and split work across threads.
#include <array>
#include <cstring>
#include <mutex>

#include "kernels/kernels.hpp"
#include "minitensor/tensor.hpp"
#include "nd_loop.hpp"

namespace minitensor::detail {
namespace {

// Starting threads costs several microseconds, so smaller tensors run on one.
constexpr std::size_t kParallelElements = std::size_t{1} << 16;

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

// Combines one result per thread. Only used for large arrays, so the lock is
// taken just a few times per call.
template <typename T, typename Reduce, typename Combine>
T parallel_reduce(std::size_t size, Reduce&& reduce, Combine&& combine) {
    std::mutex mutex;
    bool has_result = false;
    T result{};
    for_each_chunk(size, size >= kParallelElements, [&](std::size_t begin, std::size_t end) {
        const T piece = reduce(begin, end - begin);
        const std::lock_guard<std::mutex> lock(mutex);
        result = has_result ? combine(result, piece) : piece;
        has_result = true;
    });
    return result;
}

}  // namespace

template <typename T>
void binary_into(BinaryOp op, View<T> out, View<const T> a, View<const T> b) {
    const Strides strides_a = broadcast_strides(a, out.rank);
    const Strides strides_b = broadcast_strides(b, out.rank);
    const NdLoop<3> loop =
        make_loop<3>(out.rank, out.dims, {out.strides, strides_a.data(), strides_b.data()});
    if (loop.total == 0) return;

    const KernelSet<T>& set = kernels<T>();
    const int index = static_cast<int>(op);
    const std::ptrdiff_t so = loop.inner_stride(0);
    const std::ptrdiff_t sa = loop.inner_stride(1);
    const std::ptrdiff_t sb = loop.inner_stride(2);

    run_loop(loop, loop.total >= kParallelElements,
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

    run_loop(loop, loop.total >= kParallelElements,
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

    run_loop(loop, loop.total >= kParallelElements,
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
        size, [&](std::size_t begin, std::size_t count) { return set.sum(data + begin, count); },
        [](T x, T y) { return x + y; });
}

template <typename T>
T reduce_min(const T* data, std::size_t size) {
    const KernelSet<T>& set = kernels<T>();
    return parallel_reduce<T>(
        size, [&](std::size_t begin, std::size_t count) { return set.min(data + begin, count); },
        [](T x, T y) { return y < x ? y : x; });
}

template <typename T>
T reduce_max(const T* data, std::size_t size) {
    const KernelSet<T>& set = kernels<T>();
    return parallel_reduce<T>(
        size, [&](std::size_t begin, std::size_t count) { return set.max(data + begin, count); },
        [](T x, T y) { return x < y ? y : x; });
}

template <typename T>
T dot_product(const T* a, const T* b, std::size_t size) {
    const KernelSet<T>& set = kernels<T>();
    return parallel_reduce<T>(
        size,
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
