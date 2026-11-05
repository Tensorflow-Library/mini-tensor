// Kernels for x86-64 CPUs with AVX2 and FMA. Built with -mavx2 -mfma.
//
// There are 16 registers: 12 hold the totals for 6 rows of 2 vectors, 2 hold
// the current row of B, and 1 holds the current element of A.
#define MINITENSOR_ISA avx2
#define MINITENSOR_VECTOR_BYTES 32
#define MINITENSOR_MR 6
#define MINITENSOR_FOR_ROWS(X) X(0) X(1) X(2) X(3) X(4) X(5)

#include "kernels_impl.inl"
