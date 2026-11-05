# Design notes

This document explains how mini-tensor works and why it's built this way, following the seven
phases of the challenge. The main goal was execution speed, followed by code that a developer can
read and explain.

All measurements below come from one machine: an AMD Ryzen 9 7945HX with 16 cores, in WSL2 on
Ubuntu 24.04, using GCC 13.3. A virtual machine makes timings noisy, so small differences mean
nothing.

## Phase 1: Tensor representation and memory management

### What a tensor is

A `Tensor<T>` has five members:

```text
storage_   shared_ptr that owns the memory block
data_      pointer to this tensor's first element
shape_     size of each dimension
strides_   distance between neighbouring elements of each dimension
size_      total number of elements
```

`shape_` and `strides_` live inside the object as fixed arrays, up to rank 8. That means creating a
view or a shape never touches the heap.

### Why `data_` is separate from `storage_`

A view is a new `Tensor` pointing at the same block, with a different `data_`, `shape_` and
`strides_`. The `shared_ptr` counts the tensors using the block and frees it after the last one is
destroyed. So `reshape`, `transpose` and `slice` cost a few nanoseconds no matter how big the
tensor is.

### Copies and moves

Copying a `Tensor` copies its data, like `std::vector`, because that's what C++ programmers expect.
Moving only transfers pointers. Views are only ever created by calling a method that says so.

### Aligned allocation

Every buffer starts on a 64-byte boundary, the size of a cache line and of the widest SIMD register
used here. `memory.cpp` over-allocates, rounds up, and stores a small header before the block to
remember what to free.

### The block cache

Page faults turned out to be the largest hidden cost in the library. The allocator gives large
blocks back to the OS as soon as they're freed, so code like `x = a + b` in a loop gets fresh pages
each time, and each page faults on first use.

So `memory.cpp` keeps freed blocks of 64 KiB or more, up to 1 GiB in total, on shelves of
power-of-two sizes. The next tensor of a similar size reuses a block. The `neural_network` example
went from **21 seconds to 7 seconds**, and its time in the kernel fell from 84 seconds (summed over
threads) to 0.3 seconds.

`aligned_calloc`, used for tensors of zeros, skips the cache on purpose. A new block from `calloc`
is free until it's used, because the OS supplies zeroed pages, whereas a recycled block would have
to be cleared by hand.

Large blocks also ask the OS for 2 MiB pages, which reduces misses in the CPU's address translation
cache. That gave a small gain: about 15% on first use of a large block, and 2 to 3% on large matrix
multiplications.

## Phase 2: Multidimensional indexing and memory layout

### Row-major order and strides

The position of element `(i, j, k)` is:

```text
offset = i * strides[0] + j * strides[1] + k * strides[2]
```

For a contiguous `(2, 3, 4)` tensor, the strides are `(12, 4, 1)`. Row-major order was chosen
because it's what C arrays and NumPy use, so loops over the last index walk memory in order.

Strides are what make views possible:

| Operation | What changes |
|---|---|
| `transpose` | Swaps entries of `shape_` and `strides_` |
| `slice(dim, start, stop, step)` | Moves `data_` and multiplies that stride by `step` |
| `flip` | Moves `data_` to the last element and negates that stride |
| `unsqueeze` | Inserts a dimension of size 1 |

`reshape` only changes the shape, so it needs contiguous data. If the tensor isn't contiguous, it
makes a copy first, and this is documented.

### Element access

`operator()` is a small inline function. In release builds it does no bounds checking, so it's
just a multiply and add per dimension, and the compiler can optimize it inside a loop. Debug builds
assert on the rank. `at()` is the checked version, and throws on a bad index.

## Phase 3: Mathematical operations

Element-wise work is divided into three layers, so each one stays simple.

**Kernels** (`src/kernels/`) are plain single-threaded loops such as `out[i] = a[i] + b[i]`.

**The loop planner** (`src/nd_loop.hpp`) reduces any shape and layout to the simplest possible set
of nested loops:

1. It drops dimensions of size 1.
2. It sorts dimensions so the smallest strides are innermost, so that even a transposed matrix is
   walked in memory order.
3. It joins loops that continue each other. A contiguous 1000 x 1000 matrix becomes one loop of a
   million steps, so it runs the fastest kernel directly.

**The drivers** (`src/ops.cpp`) look at the innermost strides, choose a kernel, and split the
outermost loop across threads.

### Broadcasting

Broadcasting needs almost no special code. Shapes are lined up on their last dimension, and a
dimension being stretched gets a **stride of 0**, so the walk simply stays on the same element. A
scalar is a rank-0 tensor, which goes through the same path. So `a + b`, `a + 2`, and
`matrix + row` all share one function.

### Avoiding temporaries

In `a + b + c`, the result of `a + b` is a temporary. Overloads of the operators for rvalue
references detect that and write the next result into the temporary's buffer. This saves one
allocation per operator, and a test confirms the exact number of allocations.

A buffer is only reused if nothing else can see it: the tensor must be the only owner, contiguous,
and already the right shape. `Tensor::wrap` creates a shared pointer with no owner, so its memory
is never reused.

### Overlapping data

`a += a.transpose()` reads and writes the same memory in different orders. Tensors can detect that
they share a block with a different layout, and copy the right-hand side first in that case. The
same layout, as in `a += a`, is safe and needs no copy.

## Phase 4: Advanced tensor operations

### Why matrix multiplication needs care

The textbook three loops do `2 * n * n * n` floating-point operations, but most of the running time
goes to waiting for memory. At 1024 x 1024, they ran at 2.8 billion operations per second on this
machine. The CPU can do 65 billion on one core.

The solution is the approach used by GotoBLAS and BLIS: cut the matrices into blocks that fit each
cache level, and copy them into a layout the CPU can stream through.

```text
for each block of up to NC rows of A and NC columns of B     (fits L3)
  for each block of KC columns of A and rows of B
    pack the A block into panels of MR rows
    pack the B block into panels of NR columns
    for each tile of A panels and B panels, on any thread    (fits L2)
      for each A panel in the tile                           (fits L1)
        for each B panel in the tile
          micro-kernel: update MR x NR entries of C
```

### Packing

Packing copies a block into a new buffer where the micro-kernel reads every value in a straight
line. This has four benefits:

- All data the kernel reads sits next to each other, so hardware prefetching works perfectly.
- Each value can be reused without touching its original location.
- Views work with no extra code, since a transposed or sliced matrix is just another set of strides
  for the packing routines to read.
- Partial panels at the matrix edges can be padded with zeros, so there's only one kernel to write.

### The micro-kernel

The kernel in `kernels_impl.inl` is the heart of the library. With AVX-512 and `double`, it updates
a block of 14 rows and 16 columns of C:

```text
for each of the KC steps:
    load two registers from B                  (8 values each, so 16 columns)
    for each of the 14 rows:
        total[row][left]  += A[row] * B[left]
        total[row][right] += A[row] * B[right]
```

The totals stay in registers for the entire loop. There are 32 registers in AVX-512: 28 totals, 2
for B, and 1 for the current element of A. Choosing 14 rows is what makes that fit. AVX2 has 16
registers, so it uses 6 rows with the same structure. Each multiply-and-add is a single FMA
instruction.

`objdump` confirms that the double-precision AVX-512 kernel contains the 28 expected `vfmadd231pd`
instructions.

The kernel uses GCC and Clang vector types instead of intrinsics, so one source file serves AVX-512,
AVX2, SSE2 and ARM NEON. A macro expands the code once per row, which gives every total its own
named variable, so the compiler keeps it in a register.

### Why this is a mirror image of BLIS

BLIS stores C by columns, and loads SIMD registers along columns. Here, C is stored by rows, and
registers can only load neighbouring elements, so a register holds part of a row. All the roles of
A and B swap: registers load from B, and elements of A are broadcast.

### Small matrices

Packing isn't worth it for tiny matrices. Up to the work of a 32 x 32 multiplication, a simple
i-k-j loop is used. The threshold came from measuring 4 x 4 up to 128 x 128.

### Other advanced operations

`concat` and `stack` are built from `slice` and `assign`. `sum(dim)` adds slices, so it benefits
from SIMD and needs no code of its own. `matmul` of a matrix and a vector uses the `dot` kernel on
each row.

## Phase 5: Performance optimization and parallel computing

### SIMD selection at runtime

`-march=native` would make a binary crash on any CPU older than the one that built it, which goes
against the portability requirement. Instead, the kernels are compiled three times, each with
different flags:

| File | Flags | Used on |
|---|---|---|
| `kernels_generic.cpp` | none | Any CPU: SSE2 on x86-64, NEON on ARM64 |
| `kernels_avx2.cpp` | `-mavx2 -mfma` | Most x86-64 CPUs since 2013 |
| `kernels_avx512.cpp` | `-mavx512f -mavx512dq -mavx512vl -mfma` | Recent servers, AMD Zen 4 and later |

`runtime.cpp` asks the CPU what it supports at startup, and fills a table of function pointers. So
the same library runs anywhere, with the best code for each machine.

There's one trap with this design. If a kernel file included a C++ header, the compiler might build
an inline function from that header with AVX-512 instructions, and the linker could then use that
copy everywhere, which would crash older CPUs. So `kernels_impl.inl` only includes small C headers,
and everything in it is in an anonymous namespace.

| Kernels, one thread, 1024 x 1024 `double` | Time (ms) | Billions of operations per second |
|---|---:|---:|
| Generic (SSE2) | 94 | 23 |
| AVX2 | 34 | 64 |
| AVX-512 | 33 | 64 |

AVX-512 isn't faster than AVX2 here because this CPU splits 512-bit work into two 256-bit halves.
On CPUs with true 512-bit units, it should be about twice as fast, though we couldn't test that.

### Threads

Threads come from OpenMP, and matrix multiplication works like this:

1. All threads pack A and B together into shared buffers.
2. The output is divided into tiles, and each thread takes the next free tile.

Tiles are roughly square, about 4 per thread, because a square tile reuses the most data for each
byte it brings into L2. An earlier version cut the output into narrow columns instead, with one
packed copy of B per thread. Both performed about the same on this machine, and the shared version
is simpler and handles tall matrices with no extra code, so it was kept.

### When to use the threads

Worker threads go to sleep after about 2 ms without work. Measured here, using them while they're
awake costs about 13 us, but waking them costs about 210 us. So a fixed size limit can't suit both a
loop and a single call: at 100,000 elements, threads make a loop 5x faster but would make one call
10x slower.

So `worth_waking_threads()` in `runtime.cpp` looks at the size and at recent calls:

| Work, in elements of addition | Decision |
|---|---|
| Below 65,536 | One thread |
| 524,288 or more | All threads |
| In between | All threads, but only after 3 calls that each arrive within 2 ms of the last |

So one call stays on one thread, and a loop soon moves to all of them. Cheaper work counts for less:
a sum element counts as a quarter of an element of addition, and it takes about 14 multiplications
in `matmul` to equal one.

The threads themselves start when the first tensor of 256 KiB or more is created. That normally
happens before the work a program wants to time, and it cut the first threaded call from about
1,530 us to about 500 us.

| Threads, 1024 x 1024 `double` | Time (ms) | Billions of operations per second |
|---|---:|---:|
| 1 | 33.3 | 65 |
| 2 | 19.5 | 110 |
| 4 | 9.4 | 228 |
| 8 | 7.4 | 289 |
| 16 | 4.7 | 456 |

Scaling stops short of 16 times, even though each thread works independently. Changing the
algorithm and the block sizes made no measurable difference, which points to the hardware rather
than the code: all-core clock speeds are lower than single-core ones on a laptop CPU, and a virtual
machine can place two threads on one physical core.

### One thread per physical core

The default is one thread per physical core, not per hyper-thread. Two hyper-threads share a core's
FMA units, so the extra threads add little to this kind of work, and every thread has to be waited
for at each synchronization point. On this machine, 32 threads made a 128 x 128 multiplication
more than 10 times slower. They did help at 2048 x 2048 (33 ms against 41 ms), so setting
`OMP_NUM_THREADS` is worthwhile for very large matrices.

### Keeping the thread team steady

libgomp, the GCC OpenMP library, destroys threads when a team shrinks and creates them when it
grows. Choosing a thread count based on tensor size would therefore cause constant thread creation.
So each parallel section uses the full team or runs on one thread, and nothing in between.

### Ideas that weren't used

| Idea | Finding | Decision |
|---|---|---|
| Different `KC` values, from 100 to 400 | Measured: within noise | Keep the simple cache-based formula |
| Non-temporal stores for huge arrays | Not measured. They would only help above L3 size, and would cut memory traffic by a quarter at most | Not implemented |
| Expression templates | Not measured. They add a lot of complexity, and temporaries are already reused | Not implemented |

### What limits element-wise speed

Adding 100,000 elements is 5 times faster than a plain loop, but adding 10 million isn't faster at
all. Those arrays don't fit in cache, so the limit is memory bandwidth, and the plain loop already
uses most of it. Threads can't add more memory channels.

## Phase 6: Testing and application development

Tests have no dependencies. `tests/test_harness.hpp` is a small framework, so anyone with a compiler
can run them.

The checks that matter most compare results with slow, simple reference code:

- Element-wise operations run on every size from 0 to 70, and around the threading threshold, so
  every kind of loop tail is covered, for all four element types.
- Matrix multiplication is compared with textbook loops on hundreds of size combinations around
  each kernel's block sizes, again for all four types. Inputs are small integers, so even `float`
  results are exact, and `==` can be used.
- Transposed, sliced, stepped and flipped inputs are all tested.
- Results are checked on 1, 2, 3 and 7 threads, from several threads at once, and from inside an
  OpenMP parallel region.

CTest runs every test against each set of kernels through `MINITENSOR_ISA`. It also runs the matrix
tests with tiny fake cache sizes, which forces matrices to split across every blocking loop.

Speed is tested too. `tests/test_efficiency.cpp` fails if the library loses its speed, and
`benchmarks/run_efficiency_test.sh` measures every operation against plain C++. The results are in
[`EFFICIENCY.md`](EFFICIENCY.md).

Testing covered GCC 13, Clang 18, a build without OpenMP, a shared library build, AddressSanitizer
and UndefinedBehaviorSanitizer. Valgrind also ran on part of the suite with the AVX2 kernels, because
it doesn't support AVX-512.

Applications in `examples/`:

| Example | Shows |
|---|---|
| `quickstart` | A tour of the main features |
| `linear_regression` | Gradient descent on 100,000 samples |
| `neural_network` | A two-hidden-layer network trained by hand-written backpropagation; it reaches 100% accuracy on interleaved spirals |
| `image_blur` | Stencils built from views, on a 2000 x 3000 RGB image |

## Phase 7: Packaging, documentation and public release

- CMake builds a library target, `minitensor::minitensor`, for use by `add_subdirectory`,
  `FetchContent` or `find_package`. The CI installs the library and builds a separate program
  against it.
- Kernels are always built with `-O3`, even in a Debug build of the program that uses them.
- GitHub Actions builds and tests on Ubuntu 22.04 and 24.04 with GCC and Clang, and runs the
  sanitizers.
- Doxygen comments on the public headers generate the API reference.

## Limitations

- Up to 8 dimensions.
- Four element types. Adding another means adding it to the `KernelSet` instantiations.
- `matmul` works on matrices and vectors, not batches of matrices.
- Linux is the target. The code uses GCC and Clang extensions.
- Elements are `T`, not automatic differentiation variables, so gradients are written by hand.
- Views made from a `const Tensor` can write to its data, as with a `shared_ptr`.

## Questions you might be asked

**Why not use `std::vector` for storage?** It zero-fills on creation, can't guarantee 64-byte
alignment without a custom allocator, and can't be shared between views.

**Why is the packed copy worth the cost of copying?** Each packed value is used many times, while
copying touches it once. For a 1024 x 1024 multiply, packing is a tiny part of the 2 billion
floating-point operations, and it's what lets them run at full speed.

**Why 14 rows in the AVX-512 kernel?** 32 registers, minus 2 for B and 1 for A, leaves 29. Two
registers per row gives 14 rows.

**Why is `kc` based on L1, and blocks on L2?** The kernel reuses one A panel for every B panel in a
tile, so that panel should stay in L1. The B panels are reused for every A panel in the tile, so
they should stay in L2.

**Why is multiplying `float` twice as fast as `double`?** A SIMD register holds twice as many
floats.

**Why does `a + b` allocate, but `a + b + c` allocate only once?** The second `+` receives a
temporary, and reuses its buffer.

**What happens if the CPU has no AVX?** The generic kernels run. Nothing needs recompiling.

**Why is a single call slower than a call in a loop?** Between calls, the worker threads go to sleep,
and waking them takes about 210 us here. In a loop, they stay awake.

**Why not always use every thread?** Below a certain size, waking the threads takes longer than the
work. See "When to use the threads".

**Is it safe to call from several threads?** Yes, as long as they don't write to the same tensor.
Packing buffers are per thread, the kernel table is set up once, and the block cache has a lock.
