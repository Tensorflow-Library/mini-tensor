/// @file io.hpp
/// @brief Printing tensors.
#pragma once

#include <cstddef>
#include <ostream>

#include "minitensor/tensor.hpp"

namespace minitensor {

namespace detail {

template <typename T>
void print_nested(std::ostream& out, const Tensor<T>& t, std::size_t depth, std::size_t limit) {
    if (t.rank() == 0) {
        out << t.item();
        return;
    }
    const std::size_t count = t.dim(0);
    out << '[';
    for (std::size_t i = 0; i < count; ++i) {
        if (count > limit && i == limit / 2) {
            out << "...";
            i = count - limit / 2 - 1;
        } else {
            print_nested(out, t[static_cast<std::ptrdiff_t>(i)], depth + 1, limit);
        }
        if (i + 1 == count) break;
        out << ',';
        if (t.rank() == 1) {
            out << ' ';
        } else {
            out << '\n';
            for (std::size_t pad = 0; pad <= depth; ++pad) out << ' ';
        }
    }
    out << ']';
}

}  // namespace detail

/// Prints a tensor as nested brackets. Long dimensions are shortened with `...`.
template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& t) {
    detail::print_nested(out, t, 0, 6);
    return out;
}

}  // namespace minitensor
