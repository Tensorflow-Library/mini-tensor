# mini-tensor

A small, fast tensor library written from scratch in C++17. It stores multidimensional arrays in
contiguous memory and provides element-wise arithmetic, broadcasting, cache-aware matrix
multiplication, views, and reductions, with no third-party dependencies.

Built for the Diwali 2026 Challenge: Design of Mini Tensor Library.

```cpp
#include <iostream>
#include <minitensor/minitensor.hpp>

int main() {
    using Tensor = mt::Tensor<double>;

    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = Tensor::ones({2, 3});

    Tensor sum = a + b;                             // element-wise
    Tensor product = mt::matmul(a, a.transpose());  // matrix multiplication
    Tensor flat = a.reshape({-1});                  // a view: no data is copied

    a(1, 2) = 10.0;  // element access
    std::cout << product << "\n";
}
```

## Features

- **Tensors of up to 8 dimensions** of `float`, `double`, `std::int32_t` or `std::int64_t`.
- **One contiguous, 64-byte aligned buffer** in row-major order, with strides.
- **Element access** with `t(i, j, k)`, or the bounds-checked `t.at(i, j, k)`.
- **Arithmetic**: `+ - * /` between tensors and with scalars, in place or not, with NumPy-style
  broadcasting.
- **Matrix multiplication** using the same approach as high-performance BLAS libraries:
  cache blocking, packing, and SIMD micro-kernels.
- **Views that don't copy**: `reshape`, `transpose`, `permute`, `slice`, `flip`, `squeeze`,
  `unsqueeze`, `flatten`.
- **Reductions**: `sum`, `mean`, `min`, `max`, `argmax`, `dot`, `norm`, over all elements, and
  `sum` and `mean` along one dimension.
- **Runtime SIMD selection**: one binary uses AVX-512, AVX2 or baseline code, whichever the CPU
  supports. There's no need to compile with `-march=native`.
- **OpenMP multithreading** for large tensors. It's optional, and the library works without it.
- **Memory reuse**: large freed buffers are recycled, so loops don't pay for page faults.

## Requirements

- Linux. Developed on Ubuntu 24.04, and CI runs on Ubuntu 22.04 and 24.04.
- GCC 11+ or Clang 14+
- CMake 3.20+ (3.16 is enough to build; `ctest --test-dir` needs 3.20)
- OpenMP is optional. GCC includes it; for Clang, install `libomp-dev`.

On Ubuntu:

```bash
sudo apt install build-essential cmake libomp-dev
```

## Build and test

```bash
git clone https://github.com/Tensorflow-Library/mini-tensor.git
cd mini-tensor
cmake -B build
cmake --build build -j
ctest --test-dir build
```

Then try the benchmark and the examples:

```bash
./build/benchmarks/bench_tensor
./build/examples/quickstart
./build/examples/neural_network
```

The build type defaults to `Release`. These CMake options are available:

| Option | Default | Meaning |
|---|---|---|
| `MINITENSOR_OPENMP` | `ON` | Use OpenMP threads |
| `MINITENSOR_BUILD_TESTS` | `ON` when top-level | Build the tests |
| `MINITENSOR_BUILD_EXAMPLES` | `ON` when top-level | Build the examples |
| `MINITENSOR_BUILD_BENCHMARKS` | `ON` when top-level | Build the benchmark |
| `MINITENSOR_INSTALL` | `ON` when top-level | Generate the install target |
| `MINITENSOR_SANITIZE` | `OFF` | Build with ASan and UBSan |
| `BUILD_SHARED_LIBS` | `OFF` | Build a shared library |

## Use it in your project

**Install it, then use `find_package`:**

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j
cmake --install build
```

```cmake
find_package(minitensor 1.0 REQUIRED)
target_link_libraries(my_app PRIVATE minitensor::minitensor)
```

A complete example is in [`examples/consumer`](examples/consumer).

**Or let CMake download it:**

```cmake
include(FetchContent)
FetchContent_Declare(minitensor
    GIT_REPOSITORY https://github.com/Tensorflow-Library/mini-tensor.git
    GIT_TAG v1.0.0)
FetchContent_MakeAvailable(minitensor)
target_link_libraries(my_app PRIVATE minitensor::minitensor)
```

## API overview

Everything is in the namespace `minitensor`, which is also available as `mt`. Include
`<minitensor/minitensor.hpp>`. Full reference pages can be generated with `doxygen docs/Doxyfile`.

### Creating tensors

| Code | Result |
|---|---|
| `Tensor<double> t({2, 3})` | Filled with zeros |
| `Tensor<double> t({2, 3}, 1.5)` | Filled with 1.5 |
| `Tensor<double> t({2, 2}, {1, 2, 3, 4})` | From a list, in row-major order |
| `Tensor<double>::zeros(shape)`, `ones`, `full` | Named versions of the above |
| `Tensor<double>::uninitialized(shape)` | Fastest; contents are undefined |
| `Tensor<double>::eye(n)` | Identity matrix |
| `Tensor<double>::arange(start, stop, step)` | `start, start + step, ...` |
| `Tensor<double>::linspace(start, stop, count)` | Evenly spaced values |
| `Tensor<double>::rand(shape, seed)` | Uniform in `[0, 1)` |
| `Tensor<double>::randn(shape, seed)` | Standard normal |
| `Tensor<double>::from_data(shape, pointer)` | Copies from memory |
| `Tensor<double>::wrap(pointer, shape)` | Uses existing memory, without copying |

### Reading a tensor

`t.rank()`, `t.shape()`, `t.dim(i)`, `t.size()`, `t.strides()`, `t.data()`, `t.is_contiguous()`,
`t.item()`, `t(i, j)`, `t.at(i, j)`, `t[i]`

### Views

These share memory with the original, so writing through a view changes the original.

| Code | Result |
|---|---|
| `t.reshape({2, -1})` | New shape; `-1` is inferred. Copies only if `t` isn't contiguous. |
| `t.flatten()` | 1-D version |
| `t.transpose()` | Reverses the dimensions |
| `t.transpose(d0, d1)` | Swaps two dimensions |
| `t.permute({2, 0, 1})` | Reorders dimensions |
| `t.slice(dim, start, stop, step)` | A range of one dimension. Negative positions count from the end. |
| `t[i]` | One index of the first dimension |
| `t.flip(dim)` | Reverses one dimension |
| `t.squeeze()`, `t.unsqueeze(dim)` | Removes or adds dimensions of size 1 |

Copying a tensor (`Tensor b = a` or `a.clone()`) copies its data, like `std::vector`. Moving is free.

### Operations

| Code | Result |
|---|---|
| `a + b`, `a - b`, `a * b`, `a / b` | Element-wise, with broadcasting |
| `a + 1`, `2 * a`, `-a` | With a scalar |
| `a += b`, `a *= 2` | In place |
| `mt::minimum(a, b)`, `mt::maximum(a, b)` | Element-wise smaller or larger value |
| `mt::matmul(a, b)` | Matrix and vector products, as in NumPy |
| `mt::matmul_into(out, a, b)` | Matrix product into an existing tensor |
| `mt::dot(a, b)`, `mt::norm(a)` | Dot product and length |
| `t.sum()`, `t.mean()`, `t.min()`, `t.max()`, `t.argmax()` | Over all elements |
| `t.sum(dim)`, `t.mean(dim)` | Along one dimension |
| `mt::concat<double>({a, b}, dim)`, `mt::stack<double>({a, b}, dim)` | Joins tensors |
| `mt::map(t, function)` | Applies a function to each element |
| `t.cast<float>()` | Converts the element type |
| `t.assign(other)`, `t.fill(value)` | Writes into existing elements, through views too |
| `mt::equal(a, b)`, `mt::all_close(a, b, tolerance)` | Compares tensors |
| `std::cout << t` | Prints a tensor |

Wrong shapes and out-of-range arguments throw `std::invalid_argument` or `std::out_of_range`.

### Runtime settings

| Setting | Effect |
|---|---|
| `mt::set_num_threads(n)` or `OMP_NUM_THREADS=n` | Number of threads. The default is one per physical core. |
| `mt::simd_backend()` | Name of the SIMD code in use |
| `MINITENSOR_ISA=generic` or `avx2` | Forces a lower SIMD level |
| `mt::memory_stats()` | Current, peak and total memory use |
| `mt::release_cached_memory()` | Returns recycled buffers to the OS |

## Performance

Measured with `bench_tensor` on an AMD Ryzen 9 7945HX (16 cores), in WSL2 on Ubuntu 24.04, compiled
with GCC 13.3. Each time is the fastest of several runs, in milliseconds. "Plain loop" is the
obvious single-threaded C++ code: a vector of vectors with three nested loops for matrix
multiplication, and one loop for addition.

| Operation | Size | mini-tensor | Plain loop | Speedup |
|---|---|---:|---:|---:|
| Matrix multiply, `double` | 128 x 128 | 0.019 | 1.12 | 59x |
| Matrix multiply, `double` | 512 x 512 | 0.57 | 97.8 | 171x |
| Matrix multiply, `double` | 1024 x 1024 | 4.5 | 768 | 170x |
| Matrix multiply, `double` | 2048 x 2048 | 41 | | |
| Matrix multiply, `float` | 1024 x 1024 | 2.2 | | |
| Add, `double` | 100,000 elements | 0.0037 | 0.0195 | 5.2x |
| Add, `double` | 10,000,000 elements | 8.2 | 8.4 | 1.0x |
| Sum, `double` | 10,000,000 elements | 0.77 | | |
| Dot product, `double` | 10,000,000 elements | 3.2 | | |
| Matrix x vector, `double` | 2048 x 2048 | 0.074 | | |
| Transpose (view) | 2048 x 2048 | 0.0001 | | |

Adding 10 million elements gains nothing because that work is limited by memory bandwidth, not by
the CPU: the three arrays don't fit in cache, so adding threads doesn't help.

A single core reaches about 65 billion floating-point operations per second in matrix
multiplication, close to this CPU's limit of 16 per clock cycle. See
[`docs/DESIGN.md`](docs/DESIGN.md) for how this works and for more measurements. Your results will
depend on your hardware, and a virtual machine adds noise, so run `bench_tensor` to measure yours.

## Project layout

```text
include/minitensor/   Public headers
  tensor.hpp            The Tensor class: storage, indexing, views
  ops.hpp               Arithmetic, matmul, joining tensors
  shape.hpp             Shapes and broadcasting rules
  runtime.hpp           Memory, thread and SIMD settings
  io.hpp                Printing
src/
  ops.cpp               Element-wise drivers
  nd_loop.hpp           Runs an operation over any shape and layout
  matmul.cpp            Cache-blocked, parallel matrix multiplication
  memory.cpp            Aligned allocation and the cache of freed blocks
  runtime.cpp           Thread count and SIMD selection
  kernels/              SIMD loops, compiled once per instruction set
tests/                Test cases, with no external test framework
benchmarks/           Timing program
examples/             Tutorials and small applications
docs/                 Design notes and Doxygen settings
```

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) explains how the library works, phase by phase, and why.
- `doxygen docs/Doxyfile` generates API reference pages in `build/docs/html`.
- [`examples/`](examples) has a feature tour, linear regression, a neural network, and image
  processing.

## License

[MIT](LICENSE)
