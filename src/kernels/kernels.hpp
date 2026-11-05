// The table of SIMD loops ("kernels") behind every tensor operation.
//
// Each kernel is a plain single-threaded loop. Splitting work across threads
// is the job of the callers in ops.cpp and matmul.cpp.
//
// The same kernel source is compiled three times with different compiler
// flags, into the namespaces `generic`, `avx2` and `avx512`. At startup,
// `kernels<T>()` picks the best version the CPU can run. That gives one binary
// that runs on any x86-64 Linux machine and still uses the newest instructions
// when they're available.
#pragma once

#include <cstddef>
#include <cstdint>

#include "minitensor/binary_op.hpp"

namespace minitensor::detail {

// The largest register block of any micro-kernel: 14 rows of 32 floats.
inline constexpr std::size_t kMaxMr = 14;
inline constexpr std::size_t kMaxNr = 32;

template <typename T>
struct KernelSet {
    // out[i] = a[i] op b[i]. `out` may be `a` or `b`, but no other overlap.
    void (*binary_vv[kBinaryOpCount])(const T* a, const T* b, T* out, std::size_t n);
    // out[i] = a[i] op b
    void (*binary_vs[kBinaryOpCount])(const T* a, T b, T* out, std::size_t n);
    // out[i] = a op b[i]
    void (*binary_sv[kBinaryOpCount])(T a, const T* b, T* out, std::size_t n);
    // out[i * so] = a[i * sa] op b[i * sb], for any strides.
    void (*binary_strided[kBinaryOpCount])(const T* a, std::ptrdiff_t sa, const T* b,
                                           std::ptrdiff_t sb, T* out, std::ptrdiff_t so,
                                           std::size_t n);

    void (*fill)(T* out, T value, std::size_t n);
    void (*copy_strided)(const T* source, std::ptrdiff_t ss, T* out, std::ptrdiff_t so,
                         std::size_t n);

    T (*sum)(const T* a, std::size_t n);
    T (*min)(const T* a, std::size_t n);
    T (*max)(const T* a, std::size_t n);
    T (*dot)(const T* a, const T* b, std::size_t n);

    // The micro-kernel updates `mr` rows and `nr` columns of the output.
    std::size_t mr;
    std::size_t nr;

    // Copies one panel of `rows` <= mr rows and `kc` columns of A, zero-padded
    // to mr rows.
    void (*pack_a)(std::size_t rows, std::size_t kc, const T* a, std::ptrdiff_t rs,
                   std::ptrdiff_t cs, T* out);
    // Copies one panel of `kc` rows and `cols` <= nr columns of B, zero-padded
    // to nr columns.
    void (*pack_b)(std::size_t cols, std::size_t kc, const T* b, std::ptrdiff_t rs,
                   std::ptrdiff_t cs, T* out);
    // c[r * rs_c + j] += sum over p of a[p * mr + r] * b[p * nr + j]
    void (*micro_kernel)(std::size_t kc, const T* a, const T* b, T* c, std::ptrdiff_t rs_c);
    // c += a * b using plain loops, for matrices too small to be worth packing.
    void (*small_gemm)(std::size_t m, std::size_t n, std::size_t k, const T* a, std::ptrdiff_t rs_a,
                       std::ptrdiff_t cs_a, const T* b, std::ptrdiff_t rs_b, std::ptrdiff_t cs_b,
                       T* c, std::ptrdiff_t rs_c);
};

// Returns the kernels chosen for this CPU.
template <typename T>
const KernelSet<T>& kernels();

#define MINITENSOR_DECLARE_ISA(isa)                  \
    namespace isa {                                  \
    void load_kernels(KernelSet<float>& set);        \
    void load_kernels(KernelSet<double>& set);       \
    void load_kernels(KernelSet<std::int32_t>& set); \
    void load_kernels(KernelSet<std::int64_t>& set); \
    }

MINITENSOR_DECLARE_ISA(generic)
MINITENSOR_DECLARE_ISA(avx2)
MINITENSOR_DECLARE_ISA(avx512)

#undef MINITENSOR_DECLARE_ISA

}  // namespace minitensor::detail
