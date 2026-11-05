// Times the library's main operations and compares them with simple loops.
//
//   bench_tensor            runs the full set of sizes
//   bench_tensor --quick    runs smaller sizes
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "minitensor/minitensor.hpp"

namespace {

using Clock = std::chrono::steady_clock;

volatile double g_sink = 0.0;

// Runs `work` repeatedly and returns the fastest time in milliseconds. The
// fastest run is the one least disturbed by whatever else the machine is doing.
template <typename Work>
double best_time_ms(Work&& work, double budget_ms = 300.0) {
    double best = 1e300;
    double spent = 0.0;
    int runs = 0;
    do {
        const auto start = Clock::now();
        work();
        const double elapsed =
            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        best = std::min(best, elapsed);
        spent += elapsed;
        ++runs;
    } while ((spent < budget_ms && runs < 1000) || runs < 3);
    return best;
}

void print_row(const std::string& name, const std::string& size, double ms,
               double baseline_ms = 0.0) {
    std::cout << "| " << std::left << std::setw(26) << name << " | " << std::setw(14) << size
              << " | " << std::right << std::setw(11) << std::fixed << std::setprecision(4) << ms
              << " | ";
    if (baseline_ms > 0.0) {
        std::cout << std::setw(11) << baseline_ms << " | " << std::setw(7) << std::setprecision(1)
                  << baseline_ms / ms << "x |\n";
    } else {
        std::cout << std::setw(11) << "" << " | " << std::setw(8) << "" << " |\n";
    }
}

// The textbook loops, on a vector of vectors, as a point of comparison.
double naive_matmul_ms(std::size_t n) {
    using Matrix = std::vector<std::vector<double>>;
    Matrix a(n, std::vector<double>(n, 1.5));
    Matrix b(n, std::vector<double>(n, 0.5));
    Matrix c(n, std::vector<double>(n, 0.0));
    return best_time_ms(
        [&] {
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    double total = 0.0;
                    for (std::size_t k = 0; k < n; ++k) total += a[i][k] * b[k][j];
                    c[i][j] = total;
                }
            }
            g_sink = c[n / 2][n / 2];
        },
        100.0);
}

double naive_add_ms(std::size_t n) {
    std::vector<double> a(n, 1.5), b(n, 0.5), c(n);
    return best_time_ms([&] {
        for (std::size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
        g_sink = c[n / 2];
    });
}

void bench_elementwise(std::size_t n) {
    using T = mt::Tensor<double>;
    const std::string size = std::to_string(n);
    const T a = T::rand({n}, 1);
    const T b = T::rand({n}, 2);
    T out({n});

    const double baseline = naive_add_ms(n);
    print_row("add (a + b)", size, best_time_ms([&] {
                  T c = a + b;
                  g_sink = c(0);
              }),
              baseline);
    print_row("add in place (c += a)", size, best_time_ms([&] {
                  out += a;
                  g_sink = out(0);
              }));
    print_row("subtract (a - b)", size, best_time_ms([&] {
                  T c = a - b;
                  g_sink = c(0);
              }));
    print_row("multiply (a * b)", size, best_time_ms([&] {
                  T c = a * b;
                  g_sink = c(0);
              }));
    print_row("scale (a * 2)", size, best_time_ms([&] {
                  T c = a * 2.0;
                  g_sink = c(0);
              }));
    print_row("sum", size, best_time_ms([&] { g_sink = a.sum(); }));
    print_row("dot", size, best_time_ms([&] { g_sink = mt::dot(a, b); }));
}

void bench_matmul(std::size_t n) {
    using T = mt::Tensor<double>;
    const std::string size = std::to_string(n) + " x " + std::to_string(n);
    const T a = T::rand({n, n}, 1);
    const T b = T::rand({n, n}, 2);
    T out({n, n});
    const double baseline = n <= 1500 ? naive_matmul_ms(n) : 0.0;
    print_row("matmul double", size, best_time_ms([&] {
                  mt::matmul_into(out, a, b);
                  g_sink = out(0, 0);
              }),
              baseline);

    using F = mt::Tensor<float>;
    const F fa = F::rand({n, n}, 1);
    const F fb = F::rand({n, n}, 2);
    F fout({n, n});
    print_row("matmul float", size, best_time_ms([&] {
                  mt::matmul_into(fout, fa, fb);
                  g_sink = fout(0, 0);
              }));
}

void bench_other(std::size_t n) {
    using T = mt::Tensor<double>;
    const std::string size = std::to_string(n) + " x " + std::to_string(n);
    const T a = T::rand({n, n}, 1);
    const T v = T::rand({n}, 2);

    print_row("create zeros", size, best_time_ms([&] {
                  T t({n, n});
                  g_sink = t(0, 0);
              }));
    print_row("create and fill", size, best_time_ms([&] {
                  T t({n, n}, 1.0);
                  g_sink = t(0, 0);
              }));
    print_row("copy (clone)", size, best_time_ms([&] {
                  T t = a.clone();
                  g_sink = t(0, 0);
              }));
    print_row("reshape (view)", size, best_time_ms([&] {
                  T t = a.reshape({-1});
                  g_sink = t(0);
              }));
    print_row("transpose (view)", size, best_time_ms([&] {
                  T t = a.transpose();
                  g_sink = t(0, 0);
              }));
    print_row("transpose and copy", size, best_time_ms([&] {
                  T t = a.transpose().clone();
                  g_sink = t(0, 0);
              }));
    print_row("matrix x vector", size, best_time_ms([&] {
                  T t = mt::matmul(a, v);
                  g_sink = t(0);
              }));
    print_row("read every element", size, best_time_ms([&] {
                  double total = 0.0;
                  for (std::size_t i = 0; i < n; ++i) {
                      for (std::size_t j = 0; j < n; ++j) total += a(i, j);
                  }
                  g_sink = total;
              }));
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;

    std::cout << "SIMD: " << mt::simd_backend() << ", threads: " << mt::num_threads() << "\n\n"
              << "| Operation                  | Size           |   Time (ms) |  Plain loop |  "
                 "Faster  |\n"
              << "|----------------------------|----------------|-------------|-------------|------"
                 "----|\n";

    for (const std::size_t n : std::vector<std::size_t>{1000, 100000, 10000000}) {
        if (quick && n > 100000) continue;
        bench_elementwise(n);
    }
    for (const std::size_t n : std::vector<std::size_t>{8, 32, 128, 512, 1024, 2048}) {
        if (quick && n > 512) continue;
        bench_matmul(n);
    }
    bench_other(quick ? 256 : 2048);
    return 0;
}
