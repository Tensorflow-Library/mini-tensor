// Kernels for any CPU: SSE2 on x86-64, NEON on 64-bit ARM.
#define MINITENSOR_ISA generic
#define MINITENSOR_VECTOR_BYTES 16
#define MINITENSOR_MR 4
#define MINITENSOR_FOR_ROWS(X) X(0) X(1) X(2) X(3)

#include "kernels_impl.inl"
