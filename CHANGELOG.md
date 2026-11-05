# Changelog

## 1.0.0 - 2026-11-05

First public release.

- `Tensor<T>` for `float`, `double`, `std::int32_t` and `std::int64_t`, with up to 8 dimensions, in
  contiguous, 64-byte aligned, row-major storage.
- Element access, with and without bounds checking.
- Element-wise arithmetic with scalars and NumPy-style broadcasting.
- Cache-blocked, parallel matrix multiplication with SIMD micro-kernels.
- Views that don't copy data: `reshape`, `transpose`, `permute`, `slice`, `flip`, `squeeze`,
  `unsqueeze` and `flatten`.
- Reductions, `concat`, `stack`, `map` and `cast`.
- AVX-512, AVX2 and baseline kernels, selected when the program starts.
- OpenMP multithreading, which can be turned off.
- A cache of freed blocks, which avoids repeated page faults.
- CMake packaging, tests, a benchmark, examples and documentation.
