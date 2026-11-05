/// @file shape.hpp
/// @brief Tensor dimensions and NumPy-style broadcasting rules.
#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace minitensor {

/// Largest number of dimensions a tensor can have.
inline constexpr std::size_t kMaxRank = 8;

/// Distance, in elements, between neighbours along each dimension.
using Strides = std::array<std::ptrdiff_t, kMaxRank>;

/// One entry of a dimension list. Going through this class lets a brace list mix
/// integer types, as in `{rows, 3}` where `rows` is a std::size_t.
class Dim {
public:
    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    constexpr Dim(I value) noexcept : value_(static_cast<std::ptrdiff_t>(value)) {}

    constexpr std::ptrdiff_t value() const noexcept { return value_; }

private:
    std::ptrdiff_t value_;
};

/// The size of each tensor dimension.
///
/// Dimensions live inside the object, so creating or copying a Shape never
/// touches the heap.
class Shape {
public:
    /// Creates a rank-0 shape, which describes a single scalar.
    Shape() noexcept = default;

    /// Creates a shape from a brace list, e.g. `Shape{2, 3, 4}`.
    Shape(std::initializer_list<Dim> dims) {
        check_rank(dims.size());
        for (const Dim dim : dims) append(dim.value());
    }

    /// Creates a shape from a vector of dimension sizes.
    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Shape(const std::vector<I>& dims) {
        check_rank(dims.size());
        for (const I dim : dims) append(static_cast<std::ptrdiff_t>(dim));
    }

    /// Returns the shape of an empty tensor, `(0)`.
    static Shape empty() noexcept {
        Shape shape;
        shape.rank_ = 1;
        return shape;
    }

    /// Number of dimensions.
    std::size_t rank() const noexcept { return rank_; }

    /// Size of dimension `i`. Not bounds checked.
    std::size_t operator[](std::size_t i) const noexcept { return dims_[i]; }

    /// Pointer to the first dimension size.
    const std::size_t* data() const noexcept { return dims_.data(); }

    const std::size_t* begin() const noexcept { return dims_.data(); }
    const std::size_t* end() const noexcept { return dims_.data() + rank_; }

    /// Total number of elements. Throws std::length_error on overflow.
    std::size_t num_elements() const {
        std::size_t total = 1;
        for (std::size_t i = 0; i < rank_; ++i) {
            if (__builtin_mul_overflow(total, dims_[i], &total)) {
                throw std::length_error("minitensor: tensor shape " + to_string() +
                                        " is too large");
            }
        }
        return total;
    }

    /// Returns a copy with `dim` removed.
    Shape without(std::size_t dim) const noexcept {
        Shape out;
        for (std::size_t i = 0; i < rank_; ++i) {
            if (i != dim) out.dims_[out.rank_++] = dims_[i];
        }
        return out;
    }

    /// Returns a copy with `dim` set to `size`.
    Shape with(std::size_t dim, std::size_t size) const noexcept {
        Shape out = *this;
        out.dims_[dim] = size;
        return out;
    }

    /// Formats the shape as text, e.g. `(2, 3, 4)`.
    std::string to_string() const {
        std::string text = "(";
        for (std::size_t i = 0; i < rank_; ++i) {
            if (i != 0) text += ", ";
            text += std::to_string(dims_[i]);
        }
        return text + ")";
    }

    friend bool operator==(const Shape& a, const Shape& b) noexcept {
        if (a.rank_ != b.rank_) return false;
        for (std::size_t i = 0; i < a.rank_; ++i) {
            if (a.dims_[i] != b.dims_[i]) return false;
        }
        return true;
    }

    friend bool operator!=(const Shape& a, const Shape& b) noexcept { return !(a == b); }

private:
    static void check_rank(std::size_t count) {
        if (count > kMaxRank) {
            throw std::invalid_argument("minitensor: tensors can have at most " +
                                        std::to_string(kMaxRank) + " dimensions");
        }
    }

    void append(std::ptrdiff_t size) {
        if (size < 0) throw std::invalid_argument("minitensor: dimension sizes cannot be negative");
        dims_[rank_++] = static_cast<std::size_t>(size);
    }

    std::array<std::size_t, kMaxRank> dims_{};
    std::size_t rank_ = 0;
};

/// Returns the shape two tensors broadcast to, following NumPy's rules:
/// shapes are aligned on their last dimension, and a size of 1 stretches to
/// match the other tensor. Throws std::invalid_argument if they don't fit.
inline Shape broadcast_shapes(const Shape& a, const Shape& b) {
    if (a == b) return a;
    const std::size_t rank = a.rank() > b.rank() ? a.rank() : b.rank();
    std::vector<std::size_t> dims(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        const std::size_t da = i < a.rank() ? a[a.rank() - 1 - i] : 1;
        const std::size_t db = i < b.rank() ? b[b.rank() - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) {
            throw std::invalid_argument("minitensor: cannot broadcast shapes " + a.to_string() +
                                        " and " + b.to_string());
        }
        dims[rank - 1 - i] = da == 1 ? db : da;
    }
    return Shape(dims);
}

}  // namespace minitensor
