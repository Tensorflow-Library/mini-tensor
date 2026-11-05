// Kernel source, compiled once per instruction set. Before including this
// file, define:
//   MINITENSOR_ISA           namespace for this version
//   MINITENSOR_VECTOR_BYTES  SIMD register width
//   MINITENSOR_MR            rows updated by the micro-kernel
//   MINITENSOR_FOR_ROWS(X)   expands X(r) for each of those rows
//
// Only include small C headers here. Inline functions from a C++ header would
// be compiled with this file's instruction set, and the linker could then use
// that copy everywhere, which would crash on older CPUs.
#include "kernels.hpp"

namespace minitensor::detail::MINITENSOR_ISA {
namespace {

using std::ptrdiff_t;
using std::size_t;

// GCC and Clang compile arithmetic on these types directly to SIMD instructions.
template <typename T>
struct VectorOf;
template <>
struct VectorOf<float> {
    typedef float type __attribute__((vector_size(MINITENSOR_VECTOR_BYTES)));
};
template <>
struct VectorOf<double> {
    typedef double type __attribute__((vector_size(MINITENSOR_VECTOR_BYTES)));
};
template <>
struct VectorOf<std::int32_t> {
    typedef std::int32_t type __attribute__((vector_size(MINITENSOR_VECTOR_BYTES)));
};
template <>
struct VectorOf<std::int64_t> {
    typedef std::int64_t type __attribute__((vector_size(MINITENSOR_VECTOR_BYTES)));
};

template <typename T>
using Vector = typename VectorOf<T>::type;

template <typename T>
inline constexpr size_t kLanes = MINITENSOR_VECTOR_BYTES / sizeof(T);

template <typename T>
inline __attribute__((always_inline)) Vector<T> load(const T* source) {
    Vector<T> value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

template <typename T>
inline __attribute__((always_inline)) void store(T* target, const Vector<T>& value) {
    __builtin_memcpy(target, &value, sizeof(value));
}

template <BinaryOp Op, typename T>
inline __attribute__((always_inline)) T apply(T a, T b) {
    if constexpr (Op == BinaryOp::Add) return a + b;
    if constexpr (Op == BinaryOp::Sub) return a - b;
    if constexpr (Op == BinaryOp::Mul) return a * b;
    if constexpr (Op == BinaryOp::Div) return a / b;
    if constexpr (Op == BinaryOp::Min) return b < a ? b : a;
    if constexpr (Op == BinaryOp::Max) return a < b ? b : a;
}

template <BinaryOp Op, typename T>
void binary_vv(const T* a, const T* b, T* out, size_t n) {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) out[i] = apply<Op>(a[i], b[i]);
}

template <BinaryOp Op, typename T>
void binary_vs(const T* a, T b, T* out, size_t n) {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) out[i] = apply<Op>(a[i], b);
}

template <BinaryOp Op, typename T>
void binary_sv(T a, const T* b, T* out, size_t n) {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) out[i] = apply<Op>(a, b[i]);
}

template <BinaryOp Op, typename T>
void binary_strided(const T* a, ptrdiff_t sa, const T* b, ptrdiff_t sb, T* out, ptrdiff_t so,
                    size_t n) {
    for (size_t i = 0; i < n; ++i) {
        *out = apply<Op>(*a, *b);
        a += sa;
        b += sb;
        out += so;
    }
}

template <typename T>
void fill(T* out, T value, size_t n) {
#pragma omp simd
    for (size_t i = 0; i < n; ++i) out[i] = value;
}

template <typename T>
void copy_strided(const T* source, ptrdiff_t ss, T* out, ptrdiff_t so, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        *out = *source;
        source += ss;
        out += so;
    }
}

template <typename T>
inline T horizontal_sum(const Vector<T>& value) {
    T total = T(0);
    for (size_t lane = 0; lane < kLanes<T>; ++lane) total += value[lane];
    return total;
}

// A SIMD add takes several cycles to finish, so a single running total limits
// the loop to one add at a time. Four totals let the CPU run four at once.
template <typename T>
T sum(const T* a, size_t n) {
    constexpr size_t lanes = kLanes<T>;
    Vector<T> s0 = {}, s1 = {}, s2 = {}, s3 = {};
    size_t i = 0;
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        s0 += load(a + i);
        s1 += load(a + i + lanes);
        s2 += load(a + i + 2 * lanes);
        s3 += load(a + i + 3 * lanes);
    }
    T total = horizontal_sum<T>(s0 + s1 + s2 + s3);
    for (; i < n; ++i) total += a[i];
    return total;
}

template <typename T>
T dot(const T* a, const T* b, size_t n) {
    constexpr size_t lanes = kLanes<T>;
    Vector<T> s0 = {}, s1 = {}, s2 = {}, s3 = {};
    size_t i = 0;
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        s0 += load(a + i) * load(b + i);
        s1 += load(a + i + lanes) * load(b + i + lanes);
        s2 += load(a + i + 2 * lanes) * load(b + i + 2 * lanes);
        s3 += load(a + i + 3 * lanes) * load(b + i + 3 * lanes);
    }
    T total = horizontal_sum<T>(s0 + s1 + s2 + s3);
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

// Finds the smallest or largest element. Written with explicit vectors because
// compilers don't all vectorize this as a plain loop.
template <bool Smallest, typename T>
T extreme(const T* a, size_t n) {
    constexpr size_t lanes = kLanes<T>;
    const auto beats = [](auto x, auto y) { return Smallest ? x < y : y < x; };

    T best = a[0];
    size_t i = 0;
    if (n >= lanes) {
        Vector<T> running = load(a);
        for (i = lanes; i + lanes <= n; i += lanes) {
            const Vector<T> next = load(a + i);
            auto mask = beats(next, running);
            typedef decltype(mask) Mask;
            running = (Vector<T>)((((Mask)next) & mask) | (((Mask)running) & ~mask));
        }
        for (size_t lane = 0; lane < lanes; ++lane) {
            if (beats(running[lane], best)) best = running[lane];
        }
    }
    for (; i < n; ++i) {
        if (beats(a[i], best)) best = a[i];
    }
    return best;
}

// Matrix multiplication works on a block of the output of MR rows and NR
// columns, where NR is two SIMD registers wide. The running totals for this
// block stay in registers for the whole loop, which is the point of the design:
// no memory access is needed to update them.
template <typename T>
inline constexpr size_t kNr = 2 * kLanes<T>;

template <typename T>
void micro_kernel(size_t kc, const T* a, const T* b, T* c, ptrdiff_t rs_c) {
    using V = Vector<T>;
    constexpr size_t lanes = kLanes<T>;

#define MINITENSOR_DECLARE(r) V row##r##_left = {}, row##r##_right = {};
    MINITENSOR_FOR_ROWS(MINITENSOR_DECLARE)
#undef MINITENSOR_DECLARE

    for (size_t p = 0; p < kc; ++p) {
        const V b_left = load(b);
        const V b_right = load(b + lanes);
#define MINITENSOR_UPDATE(r)        \
    row##r##_left += a[r] * b_left; \
    row##r##_right += a[r] * b_right;
        MINITENSOR_FOR_ROWS(MINITENSOR_UPDATE)
#undef MINITENSOR_UPDATE
        a += MINITENSOR_MR;
        b += 2 * lanes;
    }

#define MINITENSOR_STORE(r)                                     \
    {                                                           \
        T* row = c + r * rs_c;                                  \
        store(row, load(row) + row##r##_left);                  \
        store(row + lanes, load(row + lanes) + row##r##_right); \
    }
    MINITENSOR_FOR_ROWS(MINITENSOR_STORE)
#undef MINITENSOR_STORE
}

template <typename T>
void pack_a(size_t rows, size_t kc, const T* a, ptrdiff_t rs, ptrdiff_t cs, T* out) {
    constexpr size_t mr = MINITENSOR_MR;
    for (size_t r = 0; r < rows; ++r) {
        const T* source = a + static_cast<ptrdiff_t>(r) * rs;
        T* target = out + r;
        for (size_t p = 0; p < kc; ++p) {
            *target = *source;
            source += cs;
            target += mr;
        }
    }
    for (size_t r = rows; r < mr; ++r) {
        T* target = out + r;
        for (size_t p = 0; p < kc; ++p) target[p * mr] = T(0);
    }
}

template <typename T>
void pack_b(size_t cols, size_t kc, const T* b, ptrdiff_t rs, ptrdiff_t cs, T* out) {
    constexpr size_t nr = kNr<T>;
    for (size_t p = 0; p < kc; ++p) {
        if (cs == 1 && cols == nr) {
            store(out, load(b));
            store(out + kLanes<T>, load(b + kLanes<T>));
        } else {
            for (size_t j = 0; j < cols; ++j) out[j] = b[static_cast<ptrdiff_t>(j) * cs];
            for (size_t j = cols; j < nr; ++j) out[j] = T(0);
        }
        b += rs;
        out += nr;
    }
}

template <typename T>
void small_gemm(size_t m, size_t n, size_t k, const T* a, ptrdiff_t rs_a, ptrdiff_t cs_a,
                const T* b, ptrdiff_t rs_b, ptrdiff_t cs_b, T* c, ptrdiff_t rs_c) {
    for (size_t i = 0; i < m; ++i) {
        T* row = c + static_cast<ptrdiff_t>(i) * rs_c;
        const T* a_row = a + static_cast<ptrdiff_t>(i) * rs_a;
        for (size_t p = 0; p < k; ++p) {
            const T scale = a_row[static_cast<ptrdiff_t>(p) * cs_a];
            const T* b_row = b + static_cast<ptrdiff_t>(p) * rs_b;
            if (cs_b == 1) {
#pragma omp simd
                for (size_t j = 0; j < n; ++j) row[j] += scale * b_row[j];
            } else {
                for (size_t j = 0; j < n; ++j)
                    row[j] += scale * b_row[static_cast<ptrdiff_t>(j) * cs_b];
            }
        }
    }
}

template <typename T, BinaryOp Op>
void set_binary(KernelSet<T>& set) {
    const int index = static_cast<int>(Op);
    set.binary_vv[index] = &binary_vv<Op, T>;
    set.binary_vs[index] = &binary_vs<Op, T>;
    set.binary_sv[index] = &binary_sv<Op, T>;
    set.binary_strided[index] = &binary_strided<Op, T>;
}

template <typename T>
void load_all(KernelSet<T>& set) {
    static_assert(MINITENSOR_MR <= kMaxMr && kNr<T> <= kMaxNr);
    set_binary<T, BinaryOp::Add>(set);
    set_binary<T, BinaryOp::Sub>(set);
    set_binary<T, BinaryOp::Mul>(set);
    set_binary<T, BinaryOp::Div>(set);
    set_binary<T, BinaryOp::Min>(set);
    set_binary<T, BinaryOp::Max>(set);
    set.fill = &fill<T>;
    set.copy_strided = &copy_strided<T>;
    set.sum = &sum<T>;
    set.min = &extreme<true, T>;
    set.max = &extreme<false, T>;
    set.dot = &dot<T>;
    set.mr = MINITENSOR_MR;
    set.nr = kNr<T>;
    set.pack_a = &pack_a<T>;
    set.pack_b = &pack_b<T>;
    set.micro_kernel = &micro_kernel<T>;
    set.small_gemm = &small_gemm<T>;
}

}  // namespace

void load_kernels(KernelSet<float>& set) {
    load_all(set);
}
void load_kernels(KernelSet<double>& set) {
    load_all(set);
}
void load_kernels(KernelSet<std::int32_t>& set) {
    load_all(set);
}
void load_kernels(KernelSet<std::int64_t>& set) {
    load_all(set);
}

}  // namespace minitensor::detail::MINITENSOR_ISA
