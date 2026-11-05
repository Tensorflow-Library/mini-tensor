// Kernels for x86-64 CPUs with AVX-512. Built with -mavx512f -mavx512dq
// -mavx512vl -mfma.
//
// There are 32 registers: 28 hold the totals for 14 rows of 2 vectors, 2 hold
// the current row of B, and 1 holds the current element of A.
#define MINITENSOR_ISA avx512
#define MINITENSOR_VECTOR_BYTES 64
#define MINITENSOR_MR 14
#define MINITENSOR_FOR_ROWS(X) \
    X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13)

#include "kernels_impl.inl"
