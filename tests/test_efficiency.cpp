// Fails if the library loses its speed, for example if a kernel stops using
// SIMD or a view starts copying data.
//
// Each timed check compares the library with plain C++ on the same machine, so
// it doesn't depend on how fast the machine is. The limits are far looser than
// the real results in docs/EFFICIENCY.md, so a busy machine won't fail them.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "minitensor/minitensor.hpp"
#include "test_harness.hpp"
#include "timing.hpp"

using bench::measure;
using bench::sink;
using D = mt::Tensor<double>;

namespace {

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

// Runs on one thread, so only the algorithm and SIMD are being compared.
class OneThread {
public:
    OneThread() { mt::set_num_threads(1); }
    ~OneThread() { mt::set_num_threads(0); }
};

}  // namespace

TEST(matmul_beats_the_best_plain_loop_on_one_thread) {
    const OneThread one_thread;
    const std::size_t n = 512;
    const D a = D::rand({n, n}, 1);
    const D b = D::rand({n, n}, 2);
    D out({n, n});

    const double ours = measure([&] { mt::matmul_into(out, a, b); }, 100.0).min_ms;
    const double plain =
        measure([&] { plain_ikj_matmul(a.data(), b.data(), out.data(), n); }, 100.0, 3).min_ms;
    CHECK(ours * 2.0 < plain);
}

TEST(matmul_gets_faster_with_more_threads) {
    if (mt::num_threads() < 2) return;
    const std::size_t n = 512;
    const D a = D::rand({n, n}, 1);
    const D b = D::rand({n, n}, 2);
    D out({n, n});

    const double threaded = measure([&] { mt::matmul_into(out, a, b); }, 100.0).min_ms;
    const OneThread one_thread;
    const double single = measure([&] { mt::matmul_into(out, a, b); }, 100.0).min_ms;
    CHECK(threaded < single);
}

TEST(addition_is_not_slower_than_a_plain_loop) {
    const OneThread one_thread;
    const std::size_t n = 10000;
    const D a = D::rand({n}, 1);
    const D b = D::rand({n}, 2);
    const std::vector<double> x(a.data(), a.data() + n);
    const std::vector<double> y(b.data(), b.data() + n);
    std::vector<double> z(n);

    const double ours = measure([&] {
                            const D c = a + b;
                            sink = c(0);
                        }).min_ms;
    const double plain = measure([&] {
                             for (std::size_t i = 0; i < n; ++i) z[i] = x[i] + y[i];
                             sink = z[0];
                         }).min_ms;
    CHECK(ours < plain * 2.0);
}

TEST(sum_uses_simd) {
    const OneThread one_thread;
    const std::size_t n = 10000;
    const D a = D::rand({n}, 1);
    const std::vector<double> x(a.data(), a.data() + n);

    const double ours = measure([&] { sink = a.sum(); }).min_ms;
    const double plain = measure([&] {
                             double total = 0.0;
                             for (const double value : x) total += value;
                             sink = total;
                         }).min_ms;
    CHECK(ours * 1.5 < plain);
}

TEST(element_access_costs_about_the_same_as_raw_memory) {
    const std::size_t n = 500;
    const D a = D::rand({n, n}, 1);
    const std::vector<double> plain(a.data(), a.data() + n * n);

    const double ours = measure([&] {
                            double total = 0.0;
                            for (std::size_t i = 0; i < n; ++i) {
                                for (std::size_t j = 0; j < n; ++j) total += a(i, j);
                            }
                            sink = total;
                        }).min_ms;
    const double raw = measure([&] {
                           double total = 0.0;
                           for (std::size_t i = 0; i < n; ++i) {
                               for (std::size_t j = 0; j < n; ++j) total += plain[i * n + j];
                           }
                           sink = total;
                       }).min_ms;
    CHECK(ours < raw * 3.0);
}

TEST(views_never_allocate_or_copy) {
    const D a({1000, 1000});
    const std::size_t before = mt::memory_stats().allocations;
    {
        const D reshaped = a.reshape({-1, 10});
        const D flipped = a.transpose();
        const D part = a.slice(0, 10, 20);
        const D line = a.flatten();
        const D reversed = a.flip(0);
        const D expanded = a.unsqueeze(0).squeeze();
        CHECK(reshaped.shares_data_with(a));
        CHECK(flipped.shares_data_with(a));
        CHECK(part.shares_data_with(a));
        CHECK(line.shares_data_with(a));
        CHECK(reversed.shares_data_with(a));
        CHECK(expanded.shares_data_with(a));
    }
    CHECK_EQ(mt::memory_stats().allocations, before);
}

TEST(views_take_the_same_time_at_any_size) {
    const D small({16, 16});
    const D large({4096, 4096});
    const auto view_ms = [](const D& tensor) {
        return measure([&] {
                   const D reshaped = tensor.reshape({-1});
                   const D flipped = tensor.transpose();
                   sink = static_cast<double>(reshaped.size() + flipped.size());
               })
            .min_ms;
    };
    CHECK(view_ms(large) < view_ms(small) * 4.0);
}

TEST(large_tensors_reuse_memory_in_a_loop) {
    const D a({200000}, 1.0);
    const D b({200000}, 2.0);
    const double* first = nullptr;
    for (int repeat = 0; repeat < 5; ++repeat) {
        const D c = a + b;
        if (first == nullptr) first = c.data();
        CHECK(c.data() == first);
    }
}

TEST_MAIN()
