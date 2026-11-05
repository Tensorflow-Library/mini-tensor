# Efficiency test

Execution time is the main evaluation criterion for this project, so this test measures it directly.
Every operation the challenge asks for is timed and compared with the plain C++ a developer would
otherwise write, on the same machine and in the same run.

## Summary

| Task | Result |
|---|---|
| 1. Create tensors | Filling a 2048 x 2048 tensor is 289x faster than `std::vector`, and copying it is 20x faster |
| 2. Contiguous storage | Summing a 2048 x 2048 matrix is 65x faster than a vector of vectors |
| 3. Access elements | `t(i, j)` is as fast as indexing raw memory (1.0x) |
| 4 and 5. Add, subtract, multiply | 5x faster at 100,000 elements, about 1.1x at 10 million, where memory speed is the limit |
| 6. Matrix multiplication | At 1024 x 1024, 54x faster than the best plain loop, and 155x faster than the textbook loop |
| 7. Reshape and transpose | Views take about 30 ns at any size, because no data is copied |
| 8. Memory and threads | Recycled memory makes large operations 1.5x faster, and 16 threads make matmul 7.1x faster |

## How to run it

Build the project, then run one script. It takes about a minute.

```bash
cmake -B build
cmake --build build -j
benchmarks/run_efficiency_test.sh
```

The script runs the automated checks, records the machine's details, runs the benchmark, compares
the SIMD levels, and writes everything to `efficiency-report.md`.

There are two smaller ways to run parts of it:

```bash
./build/benchmarks/bench_tensor --quick
```

```bash
ctest --test-dir build -R test_efficiency
```

## Automated checks

`tests/test_efficiency.cpp` runs as part of the normal tests and CI, and fails if the library loses
its speed. It compares the library with plain C++ on the same machine, so it doesn't depend on how
fast the machine is. The limits are far looser than the real results, so a busy machine won't fail
them.

| Check | Fails if |
|---|---|
| Matrix multiplication on one thread | It's not at least 2x faster than the best plain loop |
| Matrix multiplication on all threads | It's not faster than on one thread |
| Addition | It's more than 2x slower than a plain loop |
| Sum | It's not at least 1.5x faster than a plain loop, which would mean SIMD has stopped working |
| Element access | It's more than 3x slower than raw memory |
| Views | One allocates memory or copies data |
| Views at different sizes | A 4096 x 4096 view takes more than 4x as long as a 16 x 16 view |
| Memory reuse | A loop of large additions stops reusing its block |

## How it measures

- Each operation runs repeatedly for at least 200 ms, with at least 3 to 5 samples.
- **Fastest** is the run least disturbed by the rest of the machine. **Median** is a typical run.
- Very short operations are timed in batches, because a clock can't time a single call of 30 ns.
- Every result is written to a `volatile` variable, so the compiler can't remove the work.
- The library uses all its threads and the best SIMD level, unless a table says otherwise.
- Plain loops run on one thread, with the same compiler flags as normal Release code: `-O3`, without
  `-march=native`.

**A stricter baseline.** Plain loops built with `-O3 -march=native` get AVX-512 too, so they're
faster. They're still single-threaded and cache-unaware, and the library stays well ahead:

| Plain i-k-j loop, `double` | Without `-march=native` | With `-march=native` | mini-tensor | Speedup over the native loop |
|---|---:|---:|---:|---:|
| 512 x 512 | 32.8 ms | 16.3 ms | 0.71 ms | 23x |
| 1024 x 1024 | 269 ms | 136 ms | 4.98 ms | 27x |
| 2048 x 2048 | 2.91 s | 2.41 s | 34.8 ms | 69x |

## Results

Measured on an AMD Ryzen 9 7945HX with 16 cores, 2 threads per core and 7.6 GiB of memory available,
in WSL2 on Ubuntu 24.04, with GCC 13.3. The library chose AVX-512 and 16 threads. Your numbers will
differ, and a virtual machine adds noise, so compare results from the same run only. Small
differences mean nothing: a plain loop on 1,000 elements took 115 ns or 208 ns in the same run,
depending on the operation.

### Task 1: create and manage tensors

Compared with `std::vector<double>`.

| Operation | Size | Fastest | Median | `std::vector` | Speedup |
|---|---|---:|---:|---:|---:|
| Create filled with zeros | 2048 x 2048 | 9.86 us | 10.66 us | 14.64 ms | 1485x |
| Create filled with a value | 2048 x 2048 | 40.90 us | 48.30 us | 11.83 ms | 289x |
| Copy | 2048 x 2048 | 696 us | 943 us | 14.24 ms | 20.5x |

The zeros row overstates the gain. The OS supplies zeroed pages when they're first used, so that
cost moves to the first operation on the tensor. The other two rows are real, because they recycle
memory and fill it on all threads.

### Task 2: contiguous storage

Compared with a vector of vectors, the usual way to write a matrix in C++.

| Operation | Size | Fastest | Median | Vector of vectors | Speedup |
|---|---|---:|---:|---:|---:|
| Sum every element | 2048 x 2048 | 39.24 us | 47.03 us | 2.54 ms | 64.7x |

A 2048 x 2048 tensor uses 1 block of memory. A vector of vectors uses 2,049 blocks.

### Task 3: access and modify elements

Compared with indexing a flat `std::vector` by hand, on one thread. Here, matching the baseline is
the goal.

| Operation | Size | Fastest | Median | `std::vector` | Speedup |
|---|---|---:|---:|---:|---:|
| Read every element, `t(i, j)` | 2048 x 2048 | 2.55 ms | 2.58 ms | 2.57 ms | 1.0x |
| Read every element, `t.at(i, j)` | 2048 x 2048 | 2.82 ms | 2.93 ms | 2.57 ms | 0.9x |
| Write every element, `t(i, j)` | 2048 x 2048 | 726 us | 819 us | 740 us | 1.0x |

`t(i, j)` costs nothing extra. Bounds checking with `at` costs about 10%.

### Tasks 4 and 5: addition, subtraction and element-wise multiplication

Compared with one plain loop over `std::vector`, on one thread.

| Operation | Size | Fastest | Median | Plain loop | Speedup |
|---|---|---:|---:|---:|---:|
| Add, `a + b` | 1,000 elements | 137 ns | 141 ns | 208 ns | 1.5x |
| Add in place, `c += a` | 1,000 elements | 81 ns | 82 ns | 116 ns | 1.4x |
| Subtract, `a - b` | 1,000 elements | 136 ns | 140 ns | 115 ns | 0.8x |
| Multiply, `a * b` | 1,000 elements | 137 ns | 140 ns | 222 ns | 1.6x |
| Add, `a + b` | 100,000 elements | 4.09 us | 10.03 us | 21.72 us | 5.3x |
| Add in place, `c += a` | 100,000 elements | 3.21 us | 6.62 us | 15.87 us | 4.9x |
| Subtract, `a - b` | 100,000 elements | 4.48 us | 9.25 us | 22.59 us | 5.0x |
| Multiply, `a * b` | 100,000 elements | 4.77 us | 10.71 us | 23.27 us | 4.9x |
| Add, `a + b` | 10,000,000 elements | 6.74 ms | 6.88 ms | 7.29 ms | 1.1x |
| Add in place, `c += a` | 10,000,000 elements | 4.09 ms | 4.23 ms | 5.51 ms | 1.3x |
| Subtract, `a - b` | 10,000,000 elements | 6.72 ms | 6.85 ms | 7.21 ms | 1.1x |
| Multiply, `a * b` | 10,000,000 elements | 6.75 ms | 6.85 ms | 7.30 ms | 1.1x |

**At 1,000 elements,** the two are about even. `a + b` allocates a result and checks shapes, and the
plain loop does neither, while the plain times themselves vary from 115 to 222 ns.

**At 100,000 elements,** the arrays fit in cache, and threads give about 5x.

**At 10 million elements,** the 240 MB of data doesn't fit in cache. Time goes on moving data through
main memory, and the plain loop already uses most of its bandwidth, so threads can't add much.

### Task 6: matrix multiplication

Compared with the fastest simple loop: flat arrays in i-k-j order, on one thread.

| Operation | Size | Fastest | Median | i-k-j loop | Speedup |
|---|---|---:|---:|---:|---:|
| `double` | 8 x 8 | 142 ns | 155 ns | 177 ns | 1.2x |
| `double` | 32 x 32 | 3.75 us | 3.80 us | 8.38 us | 2.2x |
| `double` | 128 x 128 | 28.93 us | 39.54 us | 501 us | 17.3x |
| `double` | 512 x 512 | 708 us | 839 us | 32.82 ms | 46.3x |
| `double` | 1024 x 1024 | 4.98 ms | 5.50 ms | 269 ms | 54.0x |
| `double` | 2048 x 2048 | 34.82 ms | 37.17 ms | 2.91 s | 83.6x |
| `float` | 512 x 512 | 314 us | 378 us | | |
| `float` | 1024 x 1024 | 2.09 ms | 2.33 ms | | |
| `float` | 2048 x 2048 | 15.55 ms | 16.03 ms | | |

Compared with the textbook loops: a vector of vectors in i-j-k order, on one thread.

| Operation | Size | Fastest | Textbook loop | Speedup |
|---|---|---:|---:|---:|
| `double` | 32 x 32 | 3.75 us | 11.69 us | 3.1x |
| `double` | 128 x 128 | 28.93 us | 1.04 ms | 36.0x |
| `double` | 512 x 512 | 708 us | 95.66 ms | 135x |
| `double` | 1024 x 1024 | 4.98 ms | 772 ms | 155x |

| Operation | Size | Fastest | Median | Plain loop | Speedup |
|---|---|---:|---:|---:|---:|
| Matrix x vector | 2048 x 2048 | 64.40 us | 77.26 us | 2.69 ms | 41.8x |

The speedup grows with size, because a plain loop falls out of cache while the library's blocks don't.
At very small sizes, the time taken to check shapes matters as much as the arithmetic.

### Task 7: reshape and manipulate tensors

Views share memory, so they cost the same at any size. They're compared with making the same result
as a new copy, using plain loops on one thread.

| Operation | Size | Fastest | Median | Plain copy |
|---|---|---:|---:|---:|
| Reshape (view) | 2048 x 2048 | 28.4 ns | 33.8 ns | 14.29 ms |
| Flatten (view) | 2048 x 2048 | 13.8 ns | 14.0 ns | 14.29 ms |
| Transpose (view) | 2048 x 2048 | 32.7 ns | 34.4 ns | 28.92 ms |
| Transpose, then copy | 2048 x 2048 | 2.19 ms | 2.79 ms | 28.92 ms |
| Slice half the rows (view) | 2048 x 2048 | 9.6 ns | 10.4 ns | 526 us |

### Task 8: memory management and performance

**Recycled memory.** `a + b` on 10 million elements takes 6.78 ms with recycled memory, and 10.03 ms
when every call gets fresh memory from the OS, so recycling makes it 1.5x faster.

**Temporaries.** `c + d + c + d` has 3 operators but allocates 1 buffer, because each later operator
reuses the temporary.

**Thread scaling.** Each row is the same work on more threads.

| Threads | matmul 1024 x 1024 | Billion operations/s | Speedup | Add 10 million elements | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.13 ms | 61.1 | 1.0x | 10.89 ms | 1.0x |
| 2 | 19.62 ms | 109.4 | 1.8x | 9.15 ms | 1.2x |
| 4 | 9.67 ms | 222.1 | 3.6x | 9.28 ms | 1.2x |
| 8 | 5.28 ms | 406.3 | 6.6x | 8.07 ms | 1.3x |
| 16 | 4.95 ms | 434.0 | 7.1x | 7.47 ms | 1.5x |

Matrix multiplication stays in cache, so it scales well up to 8 threads. It falls short of 16x
because a laptop CPU lowers its clock speed when all cores are busy, and a virtual machine can put
two threads on one physical core. Adding 10 million elements barely scales, because all threads
share the same memory bandwidth.

**SIMD levels.** Each row is a 1024 x 1024 `double` matrix multiplication on one thread.

| SIMD | Time | Billion operations/s |
|---|---:|---:|
| Generic (SSE2) | 101.20 ms | 21.2 |
| AVX2 | 35.54 ms | 60.4 |
| AVX-512 | 34.72 ms | 61.8 |

AVX-512 gains little here because this CPU splits 512-bit work into two 256-bit halves.

**Single calls and repeated calls.** Worker threads sleep after about 2 ms without work, and waking
them takes time, so a timing of one call can look very different from a timing of a loop. Single
calls are the median of 25, with a pause before each. Plain loops run on one thread.

| Operation | Size | Single call | Repeated calls | Plain loop |
|---|---|---:|---:|---:|
| Add in place, `c += a` | 100,000 elements | 26.58 us | 2.82 us | 21.91 us |
| Add in place, `c += a` | 1,000,000 elements | 302 us | 13.34 us | 205 us |
| Add in place, `c += a` | 10,000,000 elements | 5.37 ms | 4.04 ms | 5.29 ms |
| Matrix multiply | 128 x 128 | 73.91 us | 29.80 us | 450 us |
| Matrix multiply | 256 x 256 | 350 us | 125 us | 3.82 ms |
| Matrix multiply | 1024 x 1024 | 5.23 ms | 4.89 ms | 264 ms |

Single calls of element-wise work are about the same as a plain loop, or a little slower, and
repeated calls are much faster. Matrix multiplication is faster than a plain loop in every case.

## Problems the test found

Writing this test exposed real slowdowns, which are now fixed. Times are from this machine.

| Problem | Before | After | Fix |
|---|---:|---:|---|
| `sum` of 65,536 elements, median | 41.6 us | 3.6 us | All threads shared one lock, so they queued and slept. Each now has its own result slot. |
| Single add of 100,000 elements | about 230 us | 26.6 us | Medium-sized work now wakes the threads only when calls keep arriving. |
| First threaded call in a program, 1,000,000 elements | about 1,530 us | about 500 us | Threads now start when the first large tensor is created, before most timed work. |
| 8 x 8 matrix multiply | 172 ns | 142 ns | Small outputs skip the threaded fill, and shape checks are cheaper. |
| `c += a`, 1,000 elements | 95 ns | 81 ns | Contiguous tensors of one shape skip the loop planner. |

## Known limits

- **Very small tensors.** The time taken to check shapes and allocate memory is similar to the
  arithmetic, so the library roughly matches a plain loop.
- **Arrays larger than the cache.** Element-wise speed is limited by memory bandwidth.
- **The first use of new memory.** The first operation on a new large tensor pays for page faults.
  For 8 MB, that was about 2.8 ms here. Later operations reuse memory and don't pay this.
- **Single calls of element-wise work.** A one-off call of 1,000,000 elements is slower than a plain
  loop, because the threads have to wake up. It was 302 us against 205 us here.
- **This machine.** All results come from one laptop CPU in a virtual machine. We haven't measured
  native Linux or a server CPU, where AVX-512 should gain more and waking threads should cost less.
