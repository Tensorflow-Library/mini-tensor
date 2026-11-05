/// @file ops.hpp
/// @brief Arithmetic, matrix multiplication and other tensor operations.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "minitensor/tensor.hpp"

namespace minitensor {

namespace detail {

// Stops `T` being deduced from a scalar argument, so `tensor_of_double + 1`
// compiles even though 1 is an int.
template <typename T>
struct NonDeduced {
    using type = T;
};
template <typename T>
using NonDeduced_t = typename NonDeduced<T>::type;

template <typename T>
Tensor<T> binary(BinaryOp op, const Tensor<T>& a, const Tensor<T>& b) {
    Tensor<T> out = Tensor<T>::uninitialized(broadcast_shapes(a.shape(), b.shape()));
    binary_into<T>(op, out.ref(), a.cref(), b.cref());
    return out;
}

template <typename T>
Tensor<T> binary(BinaryOp op, const Tensor<T>& a, T b) {
    Tensor<T> out = Tensor<T>::uninitialized(a.shape());
    binary_into<T>(op, out.ref(), a.cref(), scalar_view(b));
    return out;
}

template <typename T>
Tensor<T> binary(BinaryOp op, T a, const Tensor<T>& b) {
    Tensor<T> out = Tensor<T>::uninitialized(b.shape());
    binary_into<T>(op, out.ref(), scalar_view(a), b.cref());
    return out;
}

// An expression like `a + b + c` makes a temporary for `a + b`. When an operand
// is such a temporary, its buffer can hold the result, which saves an allocation.
template <typename T>
bool can_reuse(const Tensor<T>& operand, const Shape& result) noexcept {
    return operand.is_unique() && operand.is_contiguous() && operand.shape() == result;
}

template <typename T>
Tensor<T> binary(BinaryOp op, Tensor<T>&& a, const Tensor<T>& b) {
    if (!can_reuse(a, broadcast_shapes(a.shape(), b.shape()))) {
        return binary(op, static_cast<const Tensor<T>&>(a), b);
    }
    binary_into<T>(op, a.ref(), a.cref(), b.cref());
    return std::move(a);
}

template <typename T>
Tensor<T> binary(BinaryOp op, const Tensor<T>& a, Tensor<T>&& b) {
    if (!can_reuse(b, broadcast_shapes(a.shape(), b.shape()))) {
        return binary(op, a, static_cast<const Tensor<T>&>(b));
    }
    binary_into<T>(op, b.ref(), a.cref(), b.cref());
    return std::move(b);
}

template <typename T>
Tensor<T> binary(BinaryOp op, Tensor<T>&& a, Tensor<T>&& b) {
    return binary(op, std::move(a), static_cast<const Tensor<T>&>(b));
}

template <typename T>
Tensor<T> binary(BinaryOp op, Tensor<T>&& a, T b) {
    if (!can_reuse(a, a.shape())) return binary(op, static_cast<const Tensor<T>&>(a), b);
    binary_into<T>(op, a.ref(), a.cref(), scalar_view(b));
    return std::move(a);
}

template <typename T>
Tensor<T> binary(BinaryOp op, T a, Tensor<T>&& b) {
    if (!can_reuse(b, b.shape())) return binary(op, a, static_cast<const Tensor<T>&>(b));
    binary_into<T>(op, b.ref(), scalar_view(a), b.cref());
    return std::move(b);
}

}  // namespace detail

/// @name Element-wise arithmetic
/// These work on tensors of the same shape, tensors that broadcast to a common
/// shape, and scalars. `*` multiplies element by element; use `matmul` for
/// matrix multiplication.
/// @{

#define MINITENSOR_DEFINE_OPERATOR(symbol, op)                                   \
    template <typename T>                                                        \
    Tensor<T> operator symbol(const Tensor<T>& a, const Tensor<T>& b) {          \
        return detail::binary(detail::BinaryOp::op, a, b);                       \
    }                                                                            \
    template <typename T>                                                        \
    Tensor<T> operator symbol(Tensor<T>&& a, const Tensor<T>& b) {               \
        return detail::binary(detail::BinaryOp::op, std::move(a), b);            \
    }                                                                            \
    template <typename T>                                                        \
    Tensor<T> operator symbol(const Tensor<T>& a, Tensor<T>&& b) {               \
        return detail::binary(detail::BinaryOp::op, a, std::move(b));            \
    }                                                                            \
    template <typename T>                                                        \
    Tensor<T> operator symbol(Tensor<T>&& a, Tensor<T>&& b) {                    \
        return detail::binary(detail::BinaryOp::op, std::move(a), std::move(b)); \
    }                                                                            \
    template <typename T>                                                        \
    Tensor<T> operator symbol(const Tensor<T>& a, detail::NonDeduced_t<T> b) {   \
        return detail::binary<T>(detail::BinaryOp::op, a, b);                    \
    }                                                                            \
    template <typename T>                                                        \
    Tensor<T> operator symbol(Tensor<T>&& a, detail::NonDeduced_t<T> b) {        \
        return detail::binary<T>(detail::BinaryOp::op, std::move(a), b);         \
    }                                                                            \
    template <typename T>                                                        \
    Tensor<T> operator symbol(detail::NonDeduced_t<T> a, const Tensor<T>& b) {   \
        return detail::binary<T>(detail::BinaryOp::op, a, b);                    \
    }                                                                            \
    template <typename T>                                                        \
    Tensor<T> operator symbol(detail::NonDeduced_t<T> a, Tensor<T>&& b) {        \
        return detail::binary<T>(detail::BinaryOp::op, a, std::move(b));         \
    }

MINITENSOR_DEFINE_OPERATOR(+, Add)
MINITENSOR_DEFINE_OPERATOR(-, Sub)
MINITENSOR_DEFINE_OPERATOR(*, Mul)
MINITENSOR_DEFINE_OPERATOR(/, Div)

#undef MINITENSOR_DEFINE_OPERATOR

/// Returns `-a`.
template <typename T>
Tensor<T> operator-(const Tensor<T>& a) {
    return T(0) - a;
}

/// Returns the smaller of each pair of elements.
template <typename T>
Tensor<T> minimum(const Tensor<T>& a, const Tensor<T>& b) {
    return detail::binary(detail::BinaryOp::Min, a, b);
}

/// Returns the larger of each pair of elements.
template <typename T>
Tensor<T> maximum(const Tensor<T>& a, const Tensor<T>& b) {
    return detail::binary(detail::BinaryOp::Max, a, b);
}

/// @}
/// @name Linear algebra
/// @{

/// Returns the dot product of two 1-D tensors of the same size.
template <typename T>
T dot(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.rank() != 1 || b.rank() != 1 || a.size() != b.size()) {
        throw std::invalid_argument("minitensor: dot needs two 1-D tensors of the same size, not " +
                                    a.shape().to_string() + " and " + b.shape().to_string());
    }
    const Tensor<T> flat_a = a.contiguous();
    const Tensor<T> flat_b = b.contiguous();
    return detail::dot_product<T>(flat_a.data(), flat_b.data(), a.size());
}

/// Writes the matrix product `a * b` into `out`, which must already have shape
/// `(a.dim(0), b.dim(1))` and must not share data with `a` or `b`. This lets a
/// loop reuse one output tensor instead of allocating a new one each time.
template <typename T>
void matmul_into(Tensor<T>& out, const Tensor<T>& a, const Tensor<T>& b) {
    if (a.rank() != 2 || b.rank() != 2 || out.rank() != 2 || a.dim(1) != b.dim(0) ||
        out.dim(0) != a.dim(0) || out.dim(1) != b.dim(1)) {
        throw std::invalid_argument("minitensor: matmul_into cannot write " +
                                    a.shape().to_string() + " x " + b.shape().to_string() +
                                    " into " + out.shape().to_string());
    }
    if (out.shares_data_with(a) || out.shares_data_with(b)) {
        throw std::invalid_argument("minitensor: matmul_into output overlaps an input");
    }
    if (!out.is_contiguous()) {
        Tensor<T> result(out.shape());
        matmul_into(result, a, b);
        out.assign(result);
        return;
    }
    // Small outputs skip the threaded fill, whose setup would cost more than the
    // multiplication itself.
    if (out.size() < (std::size_t{1} << 16)) {
        std::fill_n(out.data(), out.size(), T(0));
    } else {
        out.fill(T(0));
    }
    detail::gemm<T>(a.dim(0), b.dim(1), a.dim(1), a.data(), a.strides()[0], a.strides()[1],
                    b.data(), b.strides()[0], b.strides()[1], out.data(), out.strides()[0]);
}

/// Multiplies matrices or vectors, with the same rules as NumPy's `matmul`:
///   - matrix x matrix: `(m, k) x (k, n)` gives `(m, n)`
///   - matrix x vector: `(m, k) x (k)` gives `(m)`
///   - vector x matrix: `(k) x (k, n)` gives `(n)`
///   - vector x vector: `(k) x (k)` gives their dot product, as a rank-0 tensor
///
/// Throws std::invalid_argument if the shapes don't fit.
template <typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    const bool sizes_match = a.rank() >= 1 && a.rank() <= 2 && b.rank() >= 1 && b.rank() <= 2 &&
                             a.dim(a.rank() - 1) == b.dim(0);
    if (!sizes_match) {
        throw std::invalid_argument("minitensor: cannot multiply shapes " + a.shape().to_string() +
                                    " and " + b.shape().to_string());
    }
    if (a.rank() == 2 && b.rank() == 2) {
        Tensor<T> out = Tensor<T>::uninitialized(Shape{a.dim(0), b.dim(1)});
        matmul_into(out, a, b);
        return out;
    }
    if (a.rank() == 2) {
        Tensor<T> out = Tensor<T>::uninitialized(Shape{a.dim(0)});
        detail::gemv<T>(a.dim(0), a.dim(1), a.data(), a.strides()[0], a.strides()[1], b.data(),
                        b.strides()[0], out.data());
        return out;
    }
    if (b.rank() == 2) {
        return matmul(a.unsqueeze(0), b).squeeze(0);
    }
    Tensor<T> out = Tensor<T>::uninitialized(Shape{});
    out.fill(dot(a, b));
    return out;
}

/// Returns the Euclidean length of all the elements of `a`.
template <typename T>
double norm(const Tensor<T>& a) {
    const Tensor<T> flat = a.contiguous();
    return std::sqrt(
        static_cast<double>(detail::dot_product<T>(flat.data(), flat.data(), flat.size())));
}

/// @}
/// @name Combining and transforming
/// @{

/// Joins tensors along an existing dimension. All other dimensions must match.
template <typename T>
Tensor<T> concat(const std::vector<Tensor<T>>& tensors, std::size_t dim = 0) {
    if (tensors.empty())
        throw std::invalid_argument("minitensor: concat needs at least one tensor");
    std::size_t total = 0;
    for (const Tensor<T>& t : tensors) {
        if (t.rank() != tensors[0].rank() ||
            t.shape().with(dim, 0) != tensors[0].shape().with(dim, 0)) {
            throw std::invalid_argument("minitensor: cannot concat shapes " +
                                        tensors[0].shape().to_string() + " and " +
                                        t.shape().to_string());
        }
        total += t.dim(dim);
    }
    Tensor<T> out = Tensor<T>::uninitialized(tensors[0].shape().with(dim, total));
    std::ptrdiff_t position = 0;
    for (const Tensor<T>& t : tensors) {
        const std::ptrdiff_t next = position + static_cast<std::ptrdiff_t>(t.dim(dim));
        out.slice(dim, position, next).assign(t);
        position = next;
    }
    return out;
}

/// Joins tensors of the same shape along a new dimension.
template <typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, std::size_t dim = 0) {
    std::vector<Tensor<T>> expanded;
    expanded.reserve(tensors.size());
    for (const Tensor<T>& t : tensors) expanded.push_back(t.unsqueeze(dim));
    return concat(expanded, dim);
}

/// Returns a tensor with `function` applied to every element, e.g.
/// `map(t, [](double x) { return std::tanh(x); })`. Large tensors are processed
/// in parallel, so `function` must be safe to call from several threads and
/// must not throw.
template <typename T, typename Function>
Tensor<T> map(const Tensor<T>& a, Function&& function) {
    const Tensor<T> flat = a.contiguous();
    Tensor<T> out = Tensor<T>::uninitialized(a.shape());
    const T* in = flat.data();
    T* result = out.data();
    const std::ptrdiff_t size = static_cast<std::ptrdiff_t>(a.size());
#pragma omp parallel for num_threads(num_threads()) if (size >= (1 << 14))
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        result[i] = static_cast<T>(function(in[i]));
    }
    return out;
}

/// True if both tensors have the same shape and elements.
template <typename T>
bool equal(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape()) return false;
    const Tensor<T> flat_a = a.contiguous();
    const Tensor<T> flat_b = b.contiguous();
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (flat_a.data()[i] != flat_b.data()[i]) return false;
    }
    return true;
}

/// True if both tensors have the same shape, and every pair of elements
/// differs by at most `tolerance`.
template <typename T>
bool all_close(const Tensor<T>& a, const Tensor<T>& b, double tolerance = 1e-6) {
    if (a.shape() != b.shape()) return false;
    const Tensor<T> flat_a = a.contiguous();
    const Tensor<T> flat_b = b.contiguous();
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double difference =
            static_cast<double>(flat_a.data()[i]) - static_cast<double>(flat_b.data()[i]);
        if (!(std::fabs(difference) <= tolerance)) return false;
    }
    return true;
}

/// @}

}  // namespace minitensor
