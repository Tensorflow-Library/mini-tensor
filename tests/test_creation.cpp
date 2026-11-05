// Phase 1: tensor representation and memory management.
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "minitensor/minitensor.hpp"
#include "test_harness.hpp"

using mt::Shape;
using D = mt::Tensor<double>;
using F = mt::Tensor<float>;
using I = mt::Tensor<std::int32_t>;
using L = mt::Tensor<std::int64_t>;

TEST(shape_reports_its_dimensions) {
    const Shape shape{2, 3, 4};
    CHECK_EQ(shape.rank(), 3u);
    CHECK_EQ(shape[1], 3u);
    CHECK_EQ(shape.num_elements(), 24u);
    CHECK(shape.to_string() == "(2, 3, 4)");
    CHECK(shape == (Shape{2, 3, 4}));
    CHECK(shape != (Shape{2, 3}));
}

TEST(shape_accepts_variables_of_any_integer_type) {
    const int rows = 2;
    const long cols = 3;
    const std::size_t depth = 4;
    CHECK_EQ((Shape{rows, rows}.num_elements()), 4u);
    CHECK_EQ((Shape{cols}.num_elements()), 3u);
    CHECK_EQ((Shape{depth, depth}.num_elements()), 16u);
    CHECK_EQ((Shape{depth, 2}.num_elements()), 8u);
    CHECK_EQ((Shape{rows, cols, depth}.num_elements()), 24u);
    CHECK_EQ(Shape(std::vector<int>{2, 5}).num_elements(), 10u);
    CHECK_EQ(D({depth, 2}).size(), 8u);
}

TEST(shape_rejects_bad_input) {
    CHECK_THROWS((Shape{1, 2, 3, 4, 5, 6, 7, 8, 9}), std::invalid_argument);
    CHECK_THROWS((Shape{2, -3}), std::invalid_argument);
    const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
    CHECK_THROWS((Shape{huge, huge}.num_elements()), std::length_error);
    CHECK_THROWS(D(Shape{huge}), std::length_error);
}

TEST(broadcast_follows_numpy_rules) {
    CHECK(mt::broadcast_shapes(Shape{3, 1}, Shape{1, 4}) == (Shape{3, 4}));
    CHECK(mt::broadcast_shapes(Shape{5, 3, 2}, Shape{2}) == (Shape{5, 3, 2}));
    CHECK(mt::broadcast_shapes(Shape{}, Shape{4, 2}) == (Shape{4, 2}));
    CHECK(mt::broadcast_shapes(Shape{0, 3}, Shape{1, 3}) == (Shape{0, 3}));
    CHECK_THROWS(mt::broadcast_shapes(Shape{2, 3}, Shape{4}), std::invalid_argument);
}

TEST(new_tensors_are_zero_filled) {
    const D t({3, 4});
    CHECK_EQ(t.rank(), 2u);
    CHECK_EQ(t.size(), 12u);
    CHECK_EQ(t.dim(1), 4u);
    for (std::size_t i = 0; i < t.size(); ++i) CHECK_EQ(t.data()[i], 0.0);
}

TEST(strides_are_row_major) {
    const D t({2, 3, 4});
    CHECK_EQ(t.strides()[0], 12);
    CHECK_EQ(t.strides()[1], 4);
    CHECK_EQ(t.strides()[2], 1);
    CHECK(t.is_contiguous());
}

TEST(buffers_are_aligned_to_cache_lines) {
    for (std::size_t n = 1; n < 2000; n += 37) {
        const F t({n});
        CHECK_EQ(reinterpret_cast<std::uintptr_t>(t.data()) % mt::kAlignment, 0u);
        const F u = F::uninitialized({n});
        CHECK_EQ(reinterpret_cast<std::uintptr_t>(u.data()) % mt::kAlignment, 0u);
    }
}

TEST(factory_functions) {
    CHECK_EQ(D::ones({2, 2}).sum(), 4.0);
    CHECK_EQ(D::full({2, 3}, 2.5).sum(), 15.0);
    CHECK_EQ(D({2, 2}, 7.0)(1, 1), 7.0);

    const D eye = D::eye(3);
    CHECK_EQ(eye(0, 0), 1.0);
    CHECK_EQ(eye(0, 1), 0.0);
    CHECK_EQ(eye.sum(), 3.0);

    const I values({2, 2}, {1, 2, 3, 4});
    CHECK_EQ(values(1, 0), 3);
    CHECK_THROWS(I({2, 2}, {1, 2, 3}), std::invalid_argument);

    const double source[] = {1.0, 2.0, 3.0, 4.0};
    CHECK_EQ(D::from_data({2, 2}, source)(1, 1), 4.0);
}

TEST(arange_and_linspace) {
    const D steps = D::arange(0.0, 1.0, 0.25);
    CHECK_EQ(steps.size(), 4u);
    CHECK_EQ(steps(3), 0.75);
    const I down = I::arange(5, 0, -2);
    CHECK_EQ(down.size(), 3u);
    CHECK_EQ(down(2), 1);
    CHECK_EQ(I::arange(3, 3).size(), 0u);
    CHECK_THROWS(D::arange(0.0, 1.0, 0.0), std::invalid_argument);

    const D line = D::linspace(0.0, 1.0, 5);
    CHECK_EQ(line.size(), 5u);
    CHECK_EQ(line(1), 0.25);
    CHECK_EQ(line(4), 1.0);
}

TEST(random_tensors_repeat_for_a_seed) {
    CHECK(mt::equal(D::rand({10, 10}, 5), D::rand({10, 10}, 5)));
    CHECK(!mt::equal(D::rand({10, 10}, 5), D::rand({10, 10}, 6)));
    const D uniform = D::rand({1000}, 1);
    CHECK(uniform.min() >= 0.0);
    CHECK(uniform.max() < 1.0);
    CHECK_NEAR(D::randn({100000}, 1).mean(), 0.0, 0.05);
}

TEST(empty_and_rank_zero_tensors) {
    const D empty;
    CHECK(empty.empty());
    CHECK_EQ(empty.rank(), 1u);
    CHECK(empty.shape() == (Shape{0}));

    const D sized({0, 3});
    CHECK_EQ(sized.size(), 0u);
    CHECK_EQ(sized.clone().size(), 0u);
    CHECK_EQ((sized + sized).size(), 0u);
    CHECK_EQ(sized.sum(), 0.0);
    CHECK_THROWS(sized.max(), std::invalid_argument);

    D scalar{Shape{}};
    scalar() = 3.0;
    CHECK_EQ(scalar.rank(), 0u);
    CHECK_EQ(scalar.size(), 1u);
    CHECK_EQ(scalar.item(), 3.0);
    CHECK_THROWS(D::ones({2}).item(), std::invalid_argument);
}

TEST(copies_are_independent) {
    D a({2, 2}, 1.0);
    D b = a;
    b(0, 0) = 9.0;
    CHECK_EQ(a(0, 0), 1.0);

    D c({5}, 0.0);
    c = a;
    CHECK(c.shape() == a.shape());
    c(1, 1) = 4.0;
    CHECK_EQ(a(1, 1), 1.0);

    D same_shape({2, 2}, 0.0);
    same_shape = a;
    CHECK_EQ(same_shape(1, 0), 1.0);
    CHECK(!same_shape.shares_data_with(a));
}

TEST(assigning_over_a_shared_buffer_leaves_the_other_view_alone) {
    D a({4}, 1.0);
    D view = a.view();
    a = D({4}, 2.0);
    CHECK_EQ(view(0), 1.0);
    CHECK_EQ(a(0), 2.0);
}

TEST(moves_transfer_the_buffer) {
    D a({100}, 1.0);
    const double* data = a.data();
    const std::size_t allocations = mt::memory_stats().allocations;

    D b = std::move(a);
    CHECK(b.data() == data);
    CHECK(a.empty());
    CHECK_EQ(a.size(), 0u);

    D c;
    c = std::move(b);
    CHECK(c.data() == data);
    CHECK(b.empty());
    CHECK_EQ(mt::memory_stats().allocations, allocations);
}

TEST(wrap_uses_existing_memory) {
    double memory[6] = {1, 2, 3, 4, 5, 6};
    {
        D t = D::wrap(memory, {2, 3});
        CHECK_EQ(t(1, 2), 6.0);
        t(0, 0) = 10.0;
        CHECK(!t.is_unique());
    }
    CHECK_EQ(memory[0], 10.0);

    // The result of an expression must not overwrite someone else's memory.
    double other[2] = {1, 2};
    const D sum = D::wrap(other, {2}) + D::ones({2});
    CHECK_EQ(sum(0), 2.0);
    CHECK_EQ(other[0], 1.0);
}

TEST(views_keep_data_alive) {
    D view;
    {
        D owner({3, 3}, 5.0);
        view = owner.slice(0, 1, 2);
    }
    CHECK_EQ(view(0, 2), 5.0);
}

TEST(memory_is_released) {
    const std::size_t before = mt::memory_stats().live_bytes;
    {
        D a({1000}, 1.0);
        D b = a.reshape({10, 100});
        D c = a.clone();
        CHECK_EQ(mt::memory_stats().live_bytes, before + 2 * 1000 * sizeof(double));
        CHECK(mt::memory_stats().peak_bytes >= mt::memory_stats().live_bytes);
    }
    CHECK_EQ(mt::memory_stats().live_bytes, before);
}

TEST(freed_large_blocks_are_reused) {
    const std::size_t n = 200000;
    const double* first = nullptr;
    {
        D a = D::uninitialized({n});
        first = a.data();
    }
    D b = D::uninitialized({n});
    CHECK(b.data() == first);

    // A nearby size can share the block, but contents must not be assumed zero.
    {
        D c = std::move(b);
        c.fill(5.0);
    }
    const D zeros({n - 1000});
    CHECK_EQ(zeros.sum(), 0.0);
    CHECK_EQ(zeros.max(), 0.0);

    const D reused = D::uninitialized({n - 1000});
    CHECK(reused.data() == first);
    CHECK_EQ(reused.size(), n - 1000);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(reused.data()) % mt::kAlignment, 0u);
}

TEST(live_bytes_stay_exact_with_cached_blocks) {
    const std::size_t before = mt::memory_stats().live_bytes;
    {
        const D a = D::uninitialized({100000});
        CHECK_EQ(mt::memory_stats().live_bytes, before + 100000 * sizeof(double));
    }
    {
        const D b = D::uninitialized({90000});
        CHECK_EQ(mt::memory_stats().live_bytes, before + 90000 * sizeof(double));
    }
    CHECK_EQ(mt::memory_stats().live_bytes, before);
}

TEST(cache_handles_many_blocks_and_can_be_cleared) {
    std::vector<D> tensors;
    for (int i = 0; i < 40; ++i) tensors.push_back(D::uninitialized({100000}));
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        tensors[i].fill(static_cast<double>(i));
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        CHECK_EQ(tensors[i](99999), static_cast<double>(i));
    }
    tensors.clear();
    mt::release_cached_memory();
    const D fresh = D::uninitialized({100000});
    CHECK_EQ(fresh.size(), 100000u);
}

TEST(all_element_types_work) {
    CHECK_EQ(F::ones({3}).sum(), 3.0f);
    CHECK_EQ(D::ones({3}).sum(), 3.0);
    CHECK_EQ(I::ones({3}).sum(), 3);
    CHECK_EQ(L::ones({3}).sum(), std::int64_t{3});
}

TEST_MAIN()
