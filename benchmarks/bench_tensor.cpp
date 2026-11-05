// The efficiency test. It times every operation the challenge asks for, and
// compares each one with the plain C++ a developer would otherwise write.
//
//   bench_tensor           runs every size, in a minute or two
//   bench_tensor --quick   runs small sizes, in a few seconds
//   bench_tensor --simd    prints one table row, for comparing SIMD levels
//
// benchmarks/run_efficiency_test.sh runs all of this and writes a report.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "minitensor/minitensor.hpp"
#include "timing.hpp"

namespace {

using Tensor = mt::Tensor<double>;
using Rows = std::vector<std::vector<double>>;
using bench::measure;
using bench::sink;
using bench::Timing;

constexpr Timing kNoBaseline{0.0, 0.0};

std::string format(const char* pattern, double value) {
    char text[32];
    std::snprintf(text, sizeof(text), pattern, value);
    return text;
}

std::string format_time(double ms) {
    if (ms <= 0.0) return "";
    if (ms < 0.001) return format("%.1f ns", ms * 1e6);
    if (ms < 1.0) return format("%.2f us", ms * 1e3);
    if (ms < 1000.0) return format("%.2f ms", ms);
    return format("%.2f s", ms / 1e3);
}

std::string format_speedup(double ratio) {
    if (ratio <= 0.0) return "";
    return format(ratio < 100.0 ? "%.1fx" : "%.0fx", ratio);
}

std::string matrix_size(std::size_t n) {
    return std::to_string(n) + " x " + std::to_string(n);
}

void print_heading(const std::string& title, const std::string& note) {
    std::cout << "\n## " << title << "\n\n" << note << "\n\n";
}

void print_table_top(const std::string& baseline) {
    std::cout << "| Operation | Size | Fastest | Median | " << baseline << " | Speedup |\n"
              << "|---|---|---:|---:|---:|---:|\n";
}

void print_row(const std::string& name, const std::string& size, Timing ours,
               Timing baseline = kNoBaseline) {
    const double ratio = baseline.min_ms > 0.0 ? baseline.min_ms / ours.min_ms : 0.0;
    std::cout << "| " << name << " | " << size << " | " << format_time(ours.min_ms) << " | "
              << format_time(ours.median_ms) << " | " << format_time(baseline.min_ms) << " | "
              << format_speedup(ratio) << " |\n";
}

// The textbook loops, on a vector of vectors.
void textbook_matmul(const Rows& a, const Rows& b, Rows& c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double total = 0.0;
            for (std::size_t k = 0; k < n; ++k) total += a[i][k] * b[k][j];
            c[i][j] = total;
        }
    }
}

// The fastest simple version: flat arrays, with the loops in the order that
// walks memory in a straight line.
void plain_ikj_matmul(const double* a, const double* b, double* c, std::size_t n) {
    std::fill(c, c + n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double* c_row = c + i * n;
        for (std::size_t k = 0; k < n; ++k) {
            const double scale = a[i * n + k];
            const double* b_row = b + k * n;
            for (std::size_t j = 0; j < n; ++j) c_row[j] += scale * b_row[j];
        }
    }
}

Rows to_rows(const Tensor& t, std::size_t n) {
    Rows rows(n);
    for (std::size_t i = 0; i < n; ++i) rows[i].assign(t.data() + i * n, t.data() + (i + 1) * n);
    return rows;
}

void test_creation(std::size_t n) {
    print_heading("Task 1: create and manage tensors",
                  "Compared with `std::vector<double>`. The zeros row is cheap at large sizes "
                  "because the OS supplies zeroed pages, so that cost moves to their first use.");
    print_table_top("std::vector");
    const std::string size = matrix_size(n);
    const Tensor a = Tensor::rand({n, n}, 1);
    const std::vector<double> plain(a.data(), a.data() + n * n);

    print_row("Create filled with zeros", size, measure([&] {
                  Tensor t({n, n});
                  sink = t(0, 0);
              }),
              measure([&] {
                  std::vector<double> v(n * n);
                  sink = v[0];
              }));
    print_row("Create filled with a value", size, measure([&] {
                  Tensor t({n, n}, 1.0);
                  sink = t(0, 0);
              }),
              measure([&] {
                  std::vector<double> v(n * n, 1.0);
                  sink = v[0];
              }));
    print_row("Copy", size, measure([&] {
                  Tensor t = a.clone();
                  sink = t(0, 0);
              }),
              measure([&] {
                  std::vector<double> v(plain);
                  sink = v[0];
              }));
}

void test_storage(std::size_t n) {
    print_heading("Task 2: contiguous storage",
                  "Compared with a vector of vectors, the usual way to write a matrix in C++.");
    print_table_top("Vector of vectors");
    const Tensor a = Tensor::rand({n, n}, 1);
    const Rows rows = to_rows(a, n);
    print_row("Sum every element", matrix_size(n), measure([&] { sink = a.sum(); }), measure([&] {
                  double total = 0.0;
                  for (const std::vector<double>& row : rows) {
                      for (const double value : row) total += value;
                  }
                  sink = total;
              }));

    std::cout << "\nA " << matrix_size(n) << " tensor uses 1 block of memory. A vector of vectors "
              << "uses " << n + 1 << " blocks, spread across the heap.\n";
}

void test_access(std::size_t n) {
    print_heading("Task 3: access and modify elements",
                  "Compared with indexing a flat `std::vector` by hand, on one thread. Here, "
                  "matching the baseline is the goal. `at` also checks the bounds.");
    print_table_top("std::vector");
    const std::string size = matrix_size(n);
    Tensor a = Tensor::rand({n, n}, 1);
    std::vector<double> plain(a.data(), a.data() + n * n);

    const Timing plain_read = measure([&] {
        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) total += plain[i * n + j];
        }
        sink = total;
    });
    print_row("Read every element, `t(i, j)`", size, measure([&] {
                  double total = 0.0;
                  for (std::size_t i = 0; i < n; ++i) {
                      for (std::size_t j = 0; j < n; ++j) total += a(i, j);
                  }
                  sink = total;
              }),
              plain_read);
    print_row("Read every element, `t.at(i, j)`", size, measure([&] {
                  double total = 0.0;
                  for (std::size_t i = 0; i < n; ++i) {
                      for (std::size_t j = 0; j < n; ++j) total += a.at(i, j);
                  }
                  sink = total;
              }),
              plain_read);
    print_row("Write every element, `t(i, j)`", size, measure([&] {
                  for (std::size_t i = 0; i < n; ++i) {
                      for (std::size_t j = 0; j < n; ++j) a(i, j) = 1.5;
                  }
                  sink = a(0, 0);
              }),
              measure([&] {
                  for (std::size_t i = 0; i < n; ++i) {
                      for (std::size_t j = 0; j < n; ++j) plain[i * n + j] = 1.5;
                  }
                  sink = plain[0];
              }));
}

void test_arithmetic(const std::vector<std::size_t>& sizes) {
    print_heading("Tasks 4 and 5: addition, subtraction and element-wise multiplication",
                  "Compared with one plain loop over `std::vector`, on one thread.");
    print_table_top("Plain loop");
    for (const std::size_t n : sizes) {
        const std::string size = std::to_string(n) + " elements";
        const Tensor a = Tensor::rand({n}, 1);
        const Tensor b = Tensor::rand({n}, 2);
        Tensor total({n});
        const std::vector<double> x(a.data(), a.data() + n);
        const std::vector<double> y(b.data(), b.data() + n);
        std::vector<double> z(n);

        const Timing plain_add = measure([&] {
            for (std::size_t i = 0; i < n; ++i) z[i] = x[i] + y[i];
            sink = z[0];
        });
        print_row("Add, `a + b`", size, measure([&] {
                      Tensor c = a + b;
                      sink = c(0);
                  }),
                  plain_add);
        print_row("Add in place, `c += a`", size, measure([&] {
                      total += a;
                      sink = total(0);
                  }),
                  measure([&] {
                      for (std::size_t i = 0; i < n; ++i) z[i] += x[i];
                      sink = z[0];
                  }));
        print_row("Subtract, `a - b`", size, measure([&] {
                      Tensor c = a - b;
                      sink = c(0);
                  }),
                  measure([&] {
                      for (std::size_t i = 0; i < n; ++i) z[i] = x[i] - y[i];
                      sink = z[0];
                  }));
        print_row("Multiply, `a * b`", size, measure([&] {
                      Tensor c = a * b;
                      sink = c(0);
                  }),
                  measure([&] {
                      for (std::size_t i = 0; i < n; ++i) z[i] = x[i] * y[i];
                      sink = z[0];
                  }));
    }
}

void test_matmul(const std::vector<std::size_t>& sizes, std::size_t vector_size) {
    std::vector<Timing> ours;
    std::vector<Timing> floats;
    std::vector<Timing> best_plain;
    for (const std::size_t n : sizes) {
        const Tensor a = Tensor::rand({n, n}, 1);
        const Tensor b = Tensor::rand({n, n}, 2);
        Tensor out({n, n});
        ours.push_back(measure([&] {
            mt::matmul_into(out, a, b);
            sink = out(0, 0);
        }));

        const mt::Tensor<float> fa = mt::Tensor<float>::rand({n, n}, 1);
        const mt::Tensor<float> fb = mt::Tensor<float>::rand({n, n}, 2);
        mt::Tensor<float> fout({n, n});
        floats.push_back(measure([&] {
            mt::matmul_into(fout, fa, fb);
            sink = fout(0, 0);
        }));

        best_plain.push_back(measure(
            [&] {
                plain_ikj_matmul(a.data(), b.data(), out.data(), n);
                sink = out(0, 0);
            },
            200.0, 3));
    }

    print_heading("Task 6: matrix multiplication",
                  "Compared with the fastest simple loop: flat arrays in i-k-j order, on one "
                  "thread.");
    print_table_top("i-k-j loop");
    for (std::size_t s = 0; s < sizes.size(); ++s) {
        print_row("`double`", matrix_size(sizes[s]), ours[s], best_plain[s]);
    }
    for (std::size_t s = 0; s < sizes.size(); ++s) {
        print_row("`float`", matrix_size(sizes[s]), floats[s]);
    }

    std::cout << "\nThe same `double` results, compared with the textbook loops: a vector of "
                 "vectors in i-j-k order, on one thread.\n\n";
    print_table_top("Textbook loop");
    for (std::size_t s = 0; s < sizes.size(); ++s) {
        const std::size_t n = sizes[s];
        if (n > 1024) continue;
        const Tensor a = Tensor::rand({n, n}, 1);
        const Tensor b = Tensor::rand({n, n}, 2);
        const Rows x = to_rows(a, n);
        const Rows y = to_rows(b, n);
        Rows z(n, std::vector<double>(n));
        print_row("`double`", matrix_size(n), ours[s],
                  measure(
                      [&] {
                          textbook_matmul(x, y, z, n);
                          sink = z[0][0];
                      },
                      200.0, 3));
    }

    std::cout << "\nMatrix x vector, compared with one plain loop per row, on one thread.\n\n";
    print_table_top("Plain loop");
    const std::size_t n = vector_size;
    const Tensor a = Tensor::rand({n, n}, 1);
    const Tensor x = Tensor::rand({n}, 2);
    std::vector<double> y(n);
    print_row("Matrix x vector", matrix_size(n), measure([&] {
                  Tensor out = mt::matmul(a, x);
                  sink = out(0);
              }),
              measure([&] {
                  for (std::size_t i = 0; i < n; ++i) {
                      double total = 0.0;
                      for (std::size_t j = 0; j < n; ++j) total += a(i, j) * x(j);
                      y[i] = total;
                  }
                  sink = y[0];
              }));
}

void test_reshape(std::size_t n) {
    print_heading("Task 7: reshape and manipulate tensors",
                  "Views share memory, so they cost the same at any size. They're compared with "
                  "making the same result as a new copy with plain loops, on one thread.");
    print_table_top("Plain copy");
    const std::string size = matrix_size(n);
    const Tensor a = Tensor::rand({n, n}, 1);
    const double* data = a.data();
    std::vector<double> out(n * n);

    const Timing flat_copy = measure([&] {
        std::vector<double> v(data, data + n * n);
        sink = v[0];
    });
    const Timing transposed_copy = measure([&] {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) out[j * n + i] = data[i * n + j];
        }
        sink = out[0];
    });

    print_row("Reshape (view)", size, measure([&] {
                  Tensor t = a.reshape({-1, 1});
                  sink = t(0, 0);
              }),
              flat_copy);
    print_row("Flatten (view)", size, measure([&] {
                  Tensor t = a.flatten();
                  sink = t(0);
              }),
              flat_copy);
    print_row("Transpose (view)", size, measure([&] {
                  Tensor t = a.transpose();
                  sink = t(0, 0);
              }),
              transposed_copy);
    print_row("Transpose, then copy", size, measure([&] {
                  Tensor t = a.transpose().clone();
                  sink = t(0, 0);
              }),
              transposed_copy);
    print_row("Slice half the rows (view)", size, measure([&] {
                  Tensor t = a.slice(0, 0, static_cast<std::ptrdiff_t>(n / 2));
                  sink = t(0, 0);
              }),
              measure([&] {
                  std::vector<double> v(data, data + n * (n / 2));
                  sink = v[0];
              }));
}

void test_memory_and_threads(std::size_t elements, std::size_t matrix) {
    print_heading("Task 8: memory management",
                  "`a + b` with recycled memory, compared with getting fresh memory from the OS "
                  "every time.");
    print_table_top("Fresh memory");
    const Tensor a = Tensor::rand({elements}, 1);
    const Tensor b = Tensor::rand({elements}, 2);
    print_row("Add, `a + b`", std::to_string(elements) + " elements", measure([&] {
                  Tensor c = a + b;
                  sink = c(0);
              }),
              measure([&] {
                  {
                      Tensor c = a + b;
                      sink = c(0);
                  }
                  mt::release_cached_memory();
              }));

    const Tensor c = Tensor::rand({1000}, 3);
    const Tensor d = Tensor::rand({1000}, 4);
    const std::size_t before = mt::memory_stats().allocations;
    const Tensor total = c + d + c + d;
    std::cout << "\n`c + d + c + d` has 3 operators but allocates "
              << mt::memory_stats().allocations - before
              << " buffer, because each later operator reuses the temporary.\n";
    sink = total(0);

    print_heading("Task 8: thread scaling", "Each row is the same work on more threads.");
    std::cout << "| Threads | matmul " << matrix_size(matrix) << " | Billion operations/s | "
              << "Speedup | Add " << elements << " elements | Speedup |\n"
              << "|---:|---:|---:|---:|---:|---:|\n";
    const Tensor x = Tensor::rand({matrix, matrix}, 1);
    const Tensor y = Tensor::rand({matrix, matrix}, 2);
    Tensor product({matrix, matrix});
    Tensor sum({elements});
    const double operations = 2.0 * static_cast<double>(matrix) * static_cast<double>(matrix) *
                              static_cast<double>(matrix);

    std::vector<int> thread_counts;
    const int most = mt::num_threads();
    for (int threads = 1; threads < most; threads *= 2) thread_counts.push_back(threads);
    thread_counts.push_back(most);

    double one_matmul = 0.0;
    double one_add = 0.0;
    for (const int threads : thread_counts) {
        mt::set_num_threads(threads);
        const double matmul_ms = measure([&] {
                                     mt::matmul_into(product, x, y);
                                     sink = product(0, 0);
                                 }).min_ms;
        const double add_ms = measure([&] {
                                  sum.assign(a);
                                  sum += b;
                                  sink = sum(0);
                              }).min_ms;
        if (threads == 1) {
            one_matmul = matmul_ms;
            one_add = add_ms;
        }
        std::cout << "| " << threads << " | " << format_time(matmul_ms) << " | "
                  << format("%.1f", operations / (matmul_ms * 1e-3) / 1e9) << " | "
                  << format_speedup(one_matmul / matmul_ms) << " | " << format_time(add_ms) << " | "
                  << format_speedup(one_add / add_ms) << " |\n";
    }
    mt::set_num_threads(0);
}

// Times one call at a time, with a pause before each that's long enough for the
// worker threads to go to sleep. This is what a program sees when it does one
// operation, rather than a loop of them.
template <typename Work>
double single_call_ms(Work&& work) {
    std::vector<double> samples;
    for (int i = 0; i < 25; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        samples.push_back(bench::run_ms(work, 1));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void test_single_calls(const std::vector<std::size_t>& element_counts,
                       const std::vector<std::size_t>& matrix_sizes) {
    print_heading("Task 8: single calls and repeated calls",
                  "Worker threads sleep after about 2 ms without work, and waking them takes "
                  "time, so a single call can be slower than a call in a loop. Single calls are "
                  "the median of 25, with a pause before each. Plain loops run on one thread.");
    std::cout << "| Operation | Size | Single call | Repeated calls | Plain loop |\n"
              << "|---|---|---:|---:|---:|\n";

    const auto print = [](const std::string& name, const std::string& size, double single,
                          double repeated, double plain) {
        std::cout << "| " << name << " | " << size << " | " << format_time(single) << " | "
                  << format_time(repeated) << " | " << format_time(plain) << " |\n";
    };

    for (const std::size_t n : element_counts) {
        const Tensor a = Tensor::rand({n}, 1);
        const Tensor b = Tensor::rand({n}, 2);
        Tensor out({n}, 0.0);
        const std::vector<double> x(a.data(), a.data() + n);
        const std::vector<double> y(b.data(), b.data() + n);
        std::vector<double> z(n);
        const auto add = [&] {
            out += a;
            sink = out(0);
        };
        print("Add in place, `c += a`", std::to_string(n) + " elements", single_call_ms(add),
              measure(add).min_ms, measure([&] {
                                       for (std::size_t i = 0; i < n; ++i) z[i] += x[i];
                                       sink = z[0];
                                   }).min_ms);
    }
    for (const std::size_t n : matrix_sizes) {
        const Tensor a = Tensor::rand({n, n}, 1);
        const Tensor b = Tensor::rand({n, n}, 2);
        Tensor out({n, n}, 0.0);
        const auto multiply = [&] {
            mt::matmul_into(out, a, b);
            sink = out(0, 0);
        };
        const double plain = measure(
                                 [&] {
                                     plain_ikj_matmul(a.data(), b.data(), out.data(), n);
                                     sink = out(0, 0);
                                 },
                                 200.0, 3)
                                 .min_ms;
        print("Matrix multiply", matrix_size(n), single_call_ms(multiply), measure(multiply).min_ms,
              plain);
    }
}

// Prints one row for the SIMD comparison in run_efficiency_test.sh.
void print_simd_row() {
    const std::size_t n = 1024;
    const Tensor a = Tensor::rand({n, n}, 1);
    const Tensor b = Tensor::rand({n, n}, 2);
    Tensor out({n, n});
    const double ms = measure([&] {
                          mt::matmul_into(out, a, b);
                          sink = out(0, 0);
                      }).min_ms;
    const double operations =
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    std::cout << "| " << mt::simd_backend() << " | " << format_time(ms) << " | "
              << format("%.1f", operations / (ms * 1e-3) / 1e9) << " |\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string option = argc > 1 ? argv[1] : "";
    if (option == "--simd") {
        print_simd_row();
        return 0;
    }
    if (!option.empty() && option != "--quick") {
        std::cerr << "usage: bench_tensor [--quick | --simd]\n";
        return 2;
    }
    const bool quick = option == "--quick";

    std::cout << "SIMD: " << mt::simd_backend() << ", threads: " << mt::num_threads() << "\n";

    const std::size_t matrix = quick ? 256 : 2048;
    test_creation(matrix);
    test_storage(matrix);
    test_access(matrix);
    test_arithmetic(quick ? std::vector<std::size_t>{1000, 100000}
                          : std::vector<std::size_t>{1000, 100000, 10000000});
    test_matmul(quick ? std::vector<std::size_t>{8, 32, 128, 512}
                      : std::vector<std::size_t>{8, 32, 128, 512, 1024, 2048},
                matrix);
    test_reshape(matrix);
    test_memory_and_threads(quick ? 100000 : 10000000, quick ? 256 : 1024);
    test_single_calls(
        quick ? std::vector<std::size_t>{100000}
              : std::vector<std::size_t>{100000, 1000000, 10000000},
        quick ? std::vector<std::size_t>{128} : std::vector<std::size_t>{128, 256, 1024});
    return 0;
}
