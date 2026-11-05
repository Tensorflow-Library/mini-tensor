# Changelog

## Unreleased

Added:

- An efficiency test: `benchmarks/run_efficiency_test.sh` times every operation the challenge asks
  for against plain C++ and writes a report. Results are in `docs/EFFICIENCY.md`.
- Automated efficiency checks in the test suite, which fail if the library loses its speed.

Changed:

- `bench_tensor` now compares every operation with a baseline, and reports fastest and median times.
- Medium-sized work only wakes the worker threads when calls keep arriving, so a single call no
  longer pays for waking them.
- The worker threads start when the first large tensor is created, instead of during the first
  threaded operation.

Fixed:

- `sum`, `min`, `max` and `dot` no longer make threads queue on a lock, which was slowing them down.
- Small matrix multiplications and element-wise operations spend less time on setup.

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
