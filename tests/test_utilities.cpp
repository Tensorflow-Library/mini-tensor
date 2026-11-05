// Phase 4: combining and converting tensors, printing, and runtime settings.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "minitensor/minitensor.hpp"
#include "test_harness.hpp"

using mt::Shape;
using D = mt::Tensor<double>;
using F = mt::Tensor<float>;
using I = mt::Tensor<std::int32_t>;

namespace {

std::string text_of(const I& tensor) {
    std::ostringstream out;
    out << tensor;
    return out.str();
}

}  // namespace

TEST(concat_joins_along_a_dimension) {
    const I a({2, 2}, {1, 2, 3, 4});
    const I b({1, 2}, {5, 6});
    CHECK(mt::equal(mt::concat<std::int32_t>({a, b}), I({3, 2}, {1, 2, 3, 4, 5, 6})));

    const I c({2, 1}, {7, 8});
    CHECK(mt::equal(mt::concat<std::int32_t>({a, c}, 1), I({2, 3}, {1, 2, 7, 3, 4, 8})));

    CHECK_THROWS(mt::concat<std::int32_t>({a, c}, 0), std::invalid_argument);
    CHECK_THROWS(mt::concat<std::int32_t>({}), std::invalid_argument);
}

TEST(stack_adds_a_dimension) {
    const I a({2}, {1, 2});
    const I b({2}, {3, 4});
    CHECK(mt::equal(mt::stack<std::int32_t>({a, b}), I({2, 2}, {1, 2, 3, 4})));
    CHECK(mt::equal(mt::stack<std::int32_t>({a, b}, 1), I({2, 2}, {1, 3, 2, 4})));
}

TEST(map_applies_a_function) {
    const D t({3}, {0.0, 1.0, 4.0});
    const D roots = mt::map(t, [](double x) { return std::sqrt(x); });
    CHECK(mt::equal(roots, D({3}, {0.0, 1.0, 2.0})));

    const I m({2, 2}, {1, 2, 3, 4});
    CHECK(mt::equal(mt::map(m.transpose(), [](int x) { return x * 10; }),
                    I({2, 2}, {10, 30, 20, 40})));

    const D large = D::ones({100000});
    CHECK_EQ(mt::map(large, [](double x) { return x + 1.0; }).sum(), 200000.0);
}

TEST(cast_converts_element_types) {
    const D t({3}, {1.9, -2.9, 3.0});
    const I truncated = t.cast<std::int32_t>();
    CHECK(mt::equal(truncated, I({3}, {1, -2, 3})));

    const F narrowed = I({2, 2}, {1, 2, 3, 4}).transpose().cast<float>();
    CHECK(mt::equal(narrowed, F({2, 2}, {1.0f, 3.0f, 2.0f, 4.0f})));
}

TEST(equal_and_all_close) {
    const D a({2}, {1.0, 2.0});
    CHECK(mt::equal(a, a.clone()));
    CHECK(!mt::equal(a, D({2}, {1.0, 3.0})));
    CHECK(!mt::equal(a, D({1, 2}, {1.0, 2.0})));
    CHECK(mt::all_close(a, D({2}, {1.0, 2.0000001})));
    CHECK(!mt::all_close(a, D({2}, {1.0, 2.1})));
    CHECK(!mt::all_close(a, D({2}, {1.0, std::nan("")})));
}

TEST(printing_uses_nested_brackets) {
    CHECK(text_of(I({3}, {1, 2, 3})) == "[1, 2, 3]");
    CHECK(text_of(I({2, 2}, {1, 2, 3, 4})) == "[[1, 2],\n [3, 4]]");
    CHECK(text_of(I(Shape{}, 7)) == "7");
    CHECK(text_of(I({0})) == "[]");
    CHECK(text_of(I({2, 1, 2}, {1, 2, 3, 4})) == "[[[1, 2]],\n [[3, 4]]]");
}

TEST(printing_shortens_long_dimensions) {
    CHECK(text_of(I::arange(0, 10)) == "[0, 1, 2, ..., 7, 8, 9]");
    CHECK(text_of(I::arange(0, 7)) == "[0, 1, 2, ..., 4, 5, 6]");
    CHECK(text_of(I::arange(0, 6)) == "[0, 1, 2, 3, 4, 5]");
}

TEST(thread_count_can_be_changed) {
    const int original = mt::num_threads();
    CHECK(original >= 1);
    mt::set_num_threads(3);
    CHECK_EQ(mt::num_threads(), 3);
    mt::set_num_threads(0);
    CHECK_EQ(mt::num_threads(), original);
}

TEST(simd_backend_has_a_known_name) {
    const std::string name = mt::simd_backend();
    CHECK(name == "generic" || name == "avx2" || name == "avx512");
}

TEST_MAIN()
