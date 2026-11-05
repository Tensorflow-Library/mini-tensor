/// @file binary_op.hpp
/// @brief The element-wise operations. Kept apart from the other headers
/// because the SIMD kernels include it (see src/kernels/kernels_impl.inl).
#pragma once

namespace minitensor::detail {

enum class BinaryOp : int { Add, Sub, Mul, Div, Min, Max };

inline constexpr int kBinaryOpCount = 6;

}  // namespace minitensor::detail
