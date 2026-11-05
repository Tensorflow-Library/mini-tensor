// Phases 3 and 5: matrix multiplication, on any layout and thread count.
//
// Inputs are small integers, so even float results are exact, and the fast
// code can be compared with the textbook loops using `==`.
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "minitensor/minitensor.hpp"
#include "test_harness.hpp"

using mt::Shape;
using mt::Tensor;
using D = Tensor<double>;
using I = Tensor<std::int32_t>;

namespace {

template <typename T>
Tensor<T> pattern(const Shape& shape, int seed) {
    Tensor<T> t = Tensor<T>::uninitialized(shape);
    for (std::size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = static_cast<T>((static_cast<int>(i % 1009) * 31 + seed) % 13 - 6);
    }
    return t;
}

template <typename T>
Tensor<T> textbook_matmul(const Tensor<T>& a, const Tensor<T>& b) {
    Tensor<T> out({a.dim(0), b.dim(1)});
    for (std::size_t i = 0; i < a.dim(0); ++i) {
        for (std::size_t j = 0; j < b.dim(1); ++j) {
            T total = T(0);
            for (std::size_t k = 0; k < a.dim(1); ++k) total += a(i, k) * b(k, j);
            out(i, j) = total;
        }
    }
    return out;
}

template <typename T>
void check_sizes() {
    // Sizes around each register block, including the ones used by each SIMD level.
    const std::vector<std::size_t> sizes = {1,  2,  3,  4,  5,  6,  7,  8,  9,  13, 14, 15, 16,
                                            17, 28, 29, 31, 32, 33, 48, 49, 63, 64, 65, 97, 130};
    std::size_t next = 0;
    const auto pick = [&] {
        next = (next + 7) % sizes.size();
        return sizes[next];
    };
    for (const std::size_t m : sizes) {
        for (const std::size_t n : sizes) {
            const std::size_t k = pick();
            const Tensor<T> a = pattern<T>({m, k}, 1);
            const Tensor<T> b = pattern<T>({k, n}, 2);
            CHECK(mt::equal(mt::matmul(a, b), textbook_matmul(a, b)));
        }
    }
}

}  // namespace

TEST(small_known_result) {
    const I a({2, 3}, {1, 2, 3, 4, 5, 6});
    const I b({3, 2}, {7, 8, 9, 10, 11, 12});
    CHECK(mt::equal(mt::matmul(a, b), I({2, 2}, {58, 64, 139, 154})));
}

TEST(float_matches_textbook_loops) {
    check_sizes<float>();
}
TEST(double_matches_textbook_loops) {
    check_sizes<double>();
}
TEST(int32_matches_textbook_loops) {
    check_sizes<std::int32_t>();
}
TEST(int64_matches_textbook_loops) {
    check_sizes<std::int64_t>();
}

TEST(large_matrices_use_all_threads_correctly) {
    const D a = pattern<double>({301, 277}, 1);
    const D b = pattern<double>({277, 353}, 2);
    CHECK(mt::equal(mt::matmul(a, b), textbook_matmul(a, b)));
}

TEST(tall_and_wide_matrices) {
    const D tall_a = pattern<double>({1500, 40}, 1);
    const D tall_b = pattern<double>({40, 3}, 2);
    CHECK(mt::equal(mt::matmul(tall_a, tall_b), textbook_matmul(tall_a, tall_b)));

    const D wide_a = pattern<double>({3, 40}, 3);
    const D wide_b = pattern<double>({40, 1500}, 4);
    CHECK(mt::equal(mt::matmul(wide_a, wide_b), textbook_matmul(wide_a, wide_b)));

    const D long_a = pattern<double>({30, 2000}, 5);
    const D long_b = pattern<double>({2000, 30}, 6);
    CHECK(mt::equal(mt::matmul(long_a, long_b), textbook_matmul(long_a, long_b)));
}

TEST(identity_leaves_a_matrix_unchanged) {
    const D a = pattern<double>({100, 100}, 1);
    CHECK(mt::equal(mt::matmul(a, D::eye(100)), a));
    CHECK(mt::equal(mt::matmul(D::eye(100), a), a));
}

TEST(result_is_the_same_on_any_number_of_threads) {
    const D a = pattern<double>({250, 300}, 1);
    const D b = pattern<double>({300, 275}, 2);
    const D expected = textbook_matmul(a, b);
    for (const int threads : {1, 2, 3, 7}) {
        mt::set_num_threads(threads);
        CHECK(mt::equal(mt::matmul(a, b), expected));
    }
    mt::set_num_threads(0);
}

TEST(inputs_can_be_transposed_or_sliced) {
    const D a = pattern<double>({90, 120}, 1);
    const D b = pattern<double>({70, 90}, 2);

    const D ta = a.transpose();
    const D tb = b.transpose();
    CHECK(!ta.is_contiguous());
    CHECK(mt::equal(mt::matmul(ta, tb), textbook_matmul(ta.clone(), tb.clone())));

    const D sliced_a = a.slice(0, 5, 80, 2).slice(1, 10, 110);
    const D sliced_b = pattern<double>({100, 50}, 3);
    CHECK(mt::equal(mt::matmul(sliced_a, sliced_b), textbook_matmul(sliced_a.clone(), sliced_b)));

    const D flipped_a = a.flip(0).flip(1);
    const D flipped_b = pattern<double>({120, 30}, 4).flip(0).flip(1);
    CHECK(mt::equal(mt::matmul(flipped_a, flipped_b),
                    textbook_matmul(flipped_a.clone(), flipped_b.clone())));
}

TEST(small_inputs_can_be_views_too) {
    const I a({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(mt::equal(mt::matmul(a, a.transpose()), I({2, 2}, {14, 32, 32, 77})));
    CHECK(mt::equal(mt::matmul(a.transpose(), a), I({3, 3}, {17, 22, 27, 22, 29, 36, 27, 36, 45})));
}

TEST(vectors) {
    const I m({2, 3}, {1, 2, 3, 4, 5, 6});
    const I x({3}, {1, 0, -1});
    const I y({2}, {2, -1});

    const I mv = mt::matmul(m, x);
    CHECK(mv.shape() == (Shape{2}));
    CHECK(mt::equal(mv, I({2}, {-2, -2})));

    const I vm = mt::matmul(y, m);
    CHECK(vm.shape() == (Shape{3}));
    CHECK(mt::equal(vm, I({3}, {-2, -1, 0})));

    const I vv = mt::matmul(x, x);
    CHECK_EQ(vv.rank(), 0u);
    CHECK_EQ(vv.item(), 2);
}

TEST(matrix_vector_on_views_and_large_sizes) {
    const D a = pattern<double>({400, 300}, 1);
    const D x = pattern<double>({300}, 2);
    const D as_column = x.unsqueeze(1);
    CHECK(mt::equal(mt::matmul(a, x), textbook_matmul(a, as_column).squeeze(1)));

    const D t = a.transpose();
    const D y = pattern<double>({400}, 3);
    CHECK(mt::equal(mt::matmul(t, y), textbook_matmul(t.clone(), y.unsqueeze(1)).squeeze(1)));

    const D stepped = pattern<double>({800}, 4).slice(0, 0, 800, 2);
    CHECK(mt::equal(mt::matmul(t, stepped),
                    textbook_matmul(t.clone(), stepped.clone().unsqueeze(1)).squeeze(1)));
}

TEST(empty_matrices) {
    CHECK(mt::matmul(D({0, 3}), D({3, 2})).shape() == (Shape{0, 2}));
    CHECK(mt::matmul(D({2, 0}), D({0, 3})).shape() == (Shape{2, 3}));
    CHECK_EQ(mt::matmul(D({2, 0}), D({0, 3})).sum(), 0.0);
}

TEST(bad_shapes_throw) {
    CHECK_THROWS(mt::matmul(D({2, 3}), D({2, 3})), std::invalid_argument);
    CHECK_THROWS(mt::matmul(D({2, 3, 4}), D({4, 2})), std::invalid_argument);
    CHECK_THROWS(mt::matmul(D(Shape{}), D({2})), std::invalid_argument);
    CHECK_THROWS(mt::matmul(D({3}), D({2})), std::invalid_argument);
}

TEST(matmul_into_reuses_the_output) {
    const D a = pattern<double>({50, 60}, 1);
    const D b = pattern<double>({60, 70}, 2);
    D out({50, 70}, 123.0);
    const double* data = out.data();
    mt::matmul_into(out, a, b);
    CHECK(out.data() == data);
    CHECK(mt::equal(out, textbook_matmul(a, b)));
}

TEST(matmul_into_checks_its_arguments) {
    D a({2, 2});
    D wrong({3, 3});
    CHECK_THROWS(mt::matmul_into(wrong, a, a), std::invalid_argument);
    CHECK_THROWS(mt::matmul_into(a, a, a), std::invalid_argument);
}

TEST(matmul_into_can_write_to_a_non_contiguous_view) {
    const I a({2, 3}, {1, 2, 3, 4, 5, 6});
    const I b({3, 2}, {7, 8, 9, 10, 11, 12});
    I storage({2, 2});
    I view = storage.transpose();
    mt::matmul_into(view, a, b);
    CHECK(mt::equal(storage, I({2, 2}, {58, 139, 64, 154})));
}

TEST(safe_to_call_from_several_threads_at_once) {
    const D a = pattern<double>({150, 170}, 1);
    const D b = pattern<double>({170, 160}, 2);
    const D expected = textbook_matmul(a, b);

    std::vector<int> correct(4, 0);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < correct.size(); ++t) {
        threads.emplace_back([&, t] {
            bool all_equal = true;
            for (int repeat = 0; repeat < 5; ++repeat) {
                all_equal = all_equal && mt::equal(mt::matmul(a, b), expected);
            }
            correct[t] = all_equal ? 1 : 0;
        });
    }
    for (std::thread& thread : threads) thread.join();
    for (const int result : correct) CHECK_EQ(result, 1);
}

TEST(safe_inside_an_openmp_parallel_region) {
    const D a = pattern<double>({100, 110}, 1);
    const D b = pattern<double>({110, 120}, 2);
    const D expected = textbook_matmul(a, b);

    int wrong = 0;
#pragma omp parallel for reduction(+ : wrong)
    for (int i = 0; i < 8; ++i) {
        if (!mt::equal(mt::matmul(a, b), expected)) ++wrong;
    }
    CHECK_EQ(wrong, 0);
}

TEST_MAIN()
