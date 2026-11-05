// Phase 3: element-wise arithmetic, broadcasting and reductions.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "minitensor/minitensor.hpp"
#include "test_harness.hpp"

using mt::Shape;
using mt::Tensor;
using D = Tensor<double>;
using I = Tensor<std::int32_t>;

namespace {

// Values from 1 to 13, so they're never 0 and division is always valid.
template <typename T>
Tensor<T> pattern(const Shape& shape, int seed) {
    Tensor<T> t = Tensor<T>::uninitialized(shape);
    for (std::size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = static_cast<T>((static_cast<int>(i % 1000) * 7 + seed) % 13 + 1);
    }
    return t;
}

// Sizes on both sides of the vector widths and of the threading threshold.
std::vector<std::size_t> test_sizes() {
    std::vector<std::size_t> sizes;
    for (std::size_t n = 0; n <= 70; ++n) sizes.push_back(n);
    for (std::size_t n : {127, 128, 129, 1000, 65535, 65536, 65537, 200003}) sizes.push_back(n);
    return sizes;
}

template <typename T, typename Tensors, typename Scalars>
void check_operator(Tensors&& on_tensors, Scalars&& on_scalars) {
    for (const std::size_t n : test_sizes()) {
        const Tensor<T> a = pattern<T>({n}, 3);
        const Tensor<T> b = pattern<T>({n}, 5);
        const Tensor<T> result = on_tensors(a, b);
        CHECK(result.shape() == (Shape{n}));
        for (std::size_t i = 0; i < n; ++i) {
            CHECK_EQ(result.data()[i], static_cast<T>(on_scalars(a.data()[i], b.data()[i])));
        }
    }
}

template <typename T>
void check_all_operators() {
    check_operator<T>([](const auto& a, const auto& b) { return a + b; },
                      [](T x, T y) { return x + y; });
    check_operator<T>([](const auto& a, const auto& b) { return a - b; },
                      [](T x, T y) { return x - y; });
    check_operator<T>([](const auto& a, const auto& b) { return a * b; },
                      [](T x, T y) { return x * y; });
    check_operator<T>([](const auto& a, const auto& b) { return a / b; },
                      [](T x, T y) { return x / y; });
    check_operator<T>([](const auto& a, const auto& b) { return mt::minimum(a, b); },
                      [](T x, T y) { return std::min(x, y); });
    check_operator<T>([](const auto& a, const auto& b) { return mt::maximum(a, b); },
                      [](T x, T y) { return std::max(x, y); });
    check_operator<T>([](const auto& a, const auto&) { return a + T(2); },
                      [](T x, T) { return x + T(2); });
    check_operator<T>([](const auto&, const auto& b) { return T(9) - b; },
                      [](T, T y) { return T(9) - y; });
    check_operator<T>([](const auto&, const auto& b) { return T(100) / b; },
                      [](T, T y) { return T(100) / y; });
    check_operator<T>([](const auto& a, const auto&) { return a * T(3); },
                      [](T x, T) { return x * T(3); });
}

template <typename T>
void check_reductions() {
    for (const std::size_t n : test_sizes()) {
        if (n == 0) continue;
        const Tensor<T> a = pattern<T>({n}, 1);
        const Tensor<T> b = pattern<T>({n}, 2);
        double sum = 0.0;
        double dot = 0.0;
        T smallest = a.data()[0];
        T largest = a.data()[0];
        std::size_t largest_at = 0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += static_cast<double>(a.data()[i]);
            dot += static_cast<double>(a.data()[i]) * static_cast<double>(b.data()[i]);
            smallest = std::min(smallest, a.data()[i]);
            if (a.data()[i] > largest) {
                largest = a.data()[i];
                largest_at = i;
            }
        }
        CHECK_NEAR(a.sum(), sum, 1e-3 * sum);
        CHECK_NEAR(mt::dot(a, b), dot, 1e-3 * dot);
        CHECK_EQ(a.min(), smallest);
        CHECK_EQ(a.max(), largest);
        CHECK_EQ(a.argmax(), largest_at);
    }
}

}  // namespace

TEST(float_operators) {
    check_all_operators<float>();
}
TEST(double_operators) {
    check_all_operators<double>();
}
TEST(int32_operators) {
    check_all_operators<std::int32_t>();
}
TEST(int64_operators) {
    check_all_operators<std::int64_t>();
}

TEST(float_reductions) {
    check_reductions<float>();
}
TEST(double_reductions) {
    check_reductions<double>();
}
TEST(int32_reductions) {
    check_reductions<std::int32_t>();
}
TEST(int64_reductions) {
    check_reductions<std::int64_t>();
}

TEST(works_on_matrices_and_higher_ranks) {
    const I a({2, 2}, {1, 2, 3, 4});
    const I b({2, 2}, {5, 6, 7, 8});
    CHECK(mt::equal(a + b, I({2, 2}, {6, 8, 10, 12})));
    CHECK(mt::equal(a - b, I({2, 2}, {-4, -4, -4, -4})));
    CHECK(mt::equal(a * b, I({2, 2}, {5, 12, 21, 32})));

    const D c = pattern<double>({2, 3, 4, 5}, 1);
    const D d = c + c;
    CHECK_EQ(d(1, 2, 3, 4), 2.0 * c(1, 2, 3, 4));
}

TEST(mismatched_shapes_throw) {
    CHECK_THROWS(D::ones({2, 3}) + D::ones({3, 2}), std::invalid_argument);
    D a = D::ones({2, 3});
    CHECK_THROWS(a += D::ones({2}), std::invalid_argument);
    CHECK_THROWS(a *= D::ones({4, 2, 3}), std::invalid_argument);
}

TEST(scalars_on_either_side) {
    const D a({2}, {1.0, 2.0});
    CHECK(mt::equal(a + 1, D({2}, {2.0, 3.0})));
    CHECK(mt::equal(1 + a, D({2}, {2.0, 3.0})));
    CHECK(mt::equal(a - 1, D({2}, {0.0, 1.0})));
    CHECK(mt::equal(1 - a, D({2}, {0.0, -1.0})));
    CHECK(mt::equal(a * 2, D({2}, {2.0, 4.0})));
    CHECK(mt::equal(2 * a, D({2}, {2.0, 4.0})));
    CHECK(mt::equal(a / 2, D({2}, {0.5, 1.0})));
    CHECK(mt::equal(2 / a, D({2}, {2.0, 1.0})));
    CHECK(mt::equal(-a, D({2}, {-1.0, -2.0})));
}

TEST(broadcasting_stretches_dimensions) {
    const I matrix({2, 3}, {1, 2, 3, 4, 5, 6});
    const I row({3}, {10, 20, 30});
    const I column({2, 1}, {100, 200});

    CHECK(mt::equal(matrix + row, I({2, 3}, {11, 22, 33, 14, 25, 36})));
    CHECK(mt::equal(row + matrix, I({2, 3}, {11, 22, 33, 14, 25, 36})));
    CHECK(mt::equal(matrix + column, I({2, 3}, {101, 102, 103, 204, 205, 206})));
    CHECK(mt::equal(column - matrix, I({2, 3}, {99, 98, 97, 196, 195, 194})));
    CHECK(mt::equal(row * column, I({2, 3}, {1000, 2000, 3000, 2000, 4000, 6000})));

    const I scalar(Shape{}, 5);
    CHECK(mt::equal(matrix + scalar, matrix + 5));

    const D cube = pattern<double>({3, 4, 5}, 1);
    const D stretched = cube + pattern<double>({4, 1}, 2);
    CHECK_EQ(stretched(2, 3, 4), cube(2, 3, 4) + pattern<double>({4, 1}, 2)(3, 0));
}

TEST(broadcasting_large_tensors_runs_in_parallel) {
    const D matrix = pattern<double>({1000, 1000}, 1);
    const D row = pattern<double>({1000}, 2);
    const D result = matrix + row;
    for (std::size_t i = 0; i < 1000; i += 97) {
        for (std::size_t j = 0; j < 1000; j += 89) {
            CHECK_EQ(result(i, j), matrix(i, j) + row(j));
        }
    }
}

TEST(operators_work_on_views) {
    const I t({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    const I transposed = t.transpose() + t.transpose();
    CHECK(transposed.shape() == (Shape{4, 3}));
    CHECK_EQ(transposed(3, 2), 24);
    CHECK_EQ(transposed(0, 1), 10);

    const I columns = t.slice(1, 1, 3) * t.slice(1, 2, 4);
    CHECK(columns.shape() == (Shape{3, 2}));
    CHECK_EQ(columns(2, 1), 11 * 12);

    const I mixed = t.transpose() + I::ones({4, 3}) * 10;
    CHECK_EQ(mixed(1, 2), 20);

    const I reversed = t.flip(1) - t;
    CHECK_EQ(reversed(0, 0), 3);
    CHECK_EQ(reversed(0, 3), -3);

    const D big = pattern<double>({300, 400}, 1);
    const D doubled = big.transpose() + big.transpose();
    CHECK_EQ(doubled(399, 299), 2.0 * big(299, 399));
    CHECK_EQ(doubled(17, 5), 2.0 * big(5, 17));
}

TEST(in_place_operators) {
    D a({3}, {1.0, 2.0, 3.0});
    const D* before = &a;
    const double* data = a.data();
    a += D({3}, {1.0, 1.0, 1.0});
    a *= 2.0;
    a -= 1.0;
    a /= D({3}, {1.0, 5.0, 7.0});
    CHECK(mt::equal(a, D({3}, {3.0, 1.0, 1.0})));
    CHECK(a.data() == data);
    CHECK(&(a += 1.0) == before);
}

TEST(in_place_operators_broadcast_and_write_through_views) {
    I m({2, 3});
    m += I({3}, {1, 2, 3});
    CHECK(mt::equal(m, I({2, 3}, {1, 2, 3, 1, 2, 3})));

    m.slice(1, 1, 2) *= 10;
    CHECK(mt::equal(m, I({2, 3}, {1, 20, 3, 1, 20, 3})));

    m.transpose() += I::ones({3, 2});
    CHECK_EQ(m(1, 2), 4);
}

TEST(in_place_operators_handle_overlap) {
    I t({4}, {1, 2, 3, 4});
    t += t;
    CHECK(mt::equal(t, I({4}, {2, 4, 6, 8})));

    I m({2, 2}, {1, 2, 3, 4});
    m += m.transpose();
    CHECK(mt::equal(m, I({2, 2}, {2, 5, 5, 8})));

    I shifted({5}, {1, 2, 3, 4, 5});
    shifted.slice(0, 1, 5) += shifted.slice(0, 0, 4);
    CHECK(mt::equal(shifted, I({5}, {1, 3, 5, 7, 9})));
}

TEST(temporaries_are_reused_instead_of_allocating) {
    const D a = pattern<double>({1000}, 1);
    const D b = pattern<double>({1000}, 2);
    const D c = pattern<double>({1000}, 3);

    const std::size_t before = mt::memory_stats().allocations;
    const D result = a + b + c - a * b + 2.0 * c;
    // a + b, then a * b and 2 * c each need a buffer. Everything else reuses one.
    CHECK_EQ(mt::memory_stats().allocations, before + 3);
    for (std::size_t i = 0; i < 1000; i += 11) {
        const double x = a(i), y = b(i), z = c(i);
        CHECK_EQ(result(i), x + y + z - x * y + 2.0 * z);
    }
}

TEST(reused_temporaries_give_correct_results_on_the_right) {
    const I a({3}, {10, 20, 30});
    const I b({3}, {1, 2, 3});
    CHECK(mt::equal(a - (b + 1), I({3}, {8, 17, 26})));
    CHECK(mt::equal(a / (b + 1), I({3}, {5, 6, 7})));
    CHECK(mt::equal((a + 0) - (b + 1), I({3}, {8, 17, 26})));
    CHECK(mt::equal(100 - (b + 1), I({3}, {98, 97, 96})));
}

TEST(a_view_of_a_temporary_is_not_overwritten) {
    I source({3}, {1, 2, 3});
    I view = source.view();
    const I result = std::move(source) + 1;
    CHECK(mt::equal(result, I({3}, {2, 3, 4})));
    CHECK(mt::equal(view, I({3}, {1, 2, 3})));
}

TEST(reductions_work_on_views) {
    const I t({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK_EQ(t.transpose().sum(), 21);
    CHECK_EQ(t.slice(1, 1, 3).max(), 6);
    CHECK_EQ(t.slice(1, 0, 1).min(), 1);
    CHECK_EQ(t.transpose().argmax(), 5u);
    CHECK_EQ(mt::dot(t[0], t[1]), 32);
    CHECK_EQ(mt::dot(t.slice(1, 0, 1).squeeze(1), t.slice(1, 1, 2).squeeze(1)), 2 + 20);
    CHECK_NEAR(t.mean(), 3.5, 1e-12);
    CHECK_NEAR(mt::norm(D({2}, {3.0, 4.0})), 5.0, 1e-12);
    CHECK_THROWS(mt::dot(t, t), std::invalid_argument);
}

TEST(reductions_along_a_dimension) {
    const I t({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(mt::equal(t.sum(0), I({3}, {5, 7, 9})));
    CHECK(mt::equal(t.sum(1), I({2}, {6, 15})));
    CHECK_EQ(I({3}, {1, 2, 3}).sum(0).item(), 6);

    const D d({2, 2}, {1.0, 2.0, 3.0, 6.0});
    CHECK(mt::equal(d.mean(0), D({2}, {2.0, 4.0})));
    CHECK(mt::equal(d.mean(1), D({2}, {1.5, 4.5})));
    CHECK_THROWS(t.sum(2), std::out_of_range);
}

TEST_MAIN()
