// Phase 2: multidimensional indexing and memory layout.
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "minitensor/minitensor.hpp"
#include "test_harness.hpp"

using mt::Shape;
using D = mt::Tensor<double>;
using I = mt::Tensor<std::int32_t>;

namespace {

// A tensor holding 0, 1, 2, ...
I counting(const Shape& shape) {
    I t = I::uninitialized(shape);
    for (std::size_t i = 0; i < t.size(); ++i) t.data()[i] = static_cast<std::int32_t>(i);
    return t;
}

}  // namespace

TEST(elements_are_stored_in_row_major_order) {
    const I t = counting({2, 3, 4});
    CHECK_EQ(t(0, 0, 0), 0);
    CHECK_EQ(t(0, 0, 3), 3);
    CHECK_EQ(t(0, 1, 0), 4);
    CHECK_EQ(t(1, 0, 0), 12);
    CHECK_EQ(t(1, 2, 3), 23);
}

TEST(elements_can_be_modified) {
    D t({2, 2});
    t(0, 1) = 3.5;
    t.at(1, 0) = 2.5;
    CHECK_EQ(t.data()[1], 3.5);
    CHECK_EQ(t.data()[2], 2.5);
    const D& constant = t;
    CHECK_EQ(constant(0, 1), 3.5);
    CHECK_EQ(constant.at(1, 0), 2.5);
}

TEST(at_checks_indices) {
    D t({2, 3});
    CHECK_THROWS(t.at(2, 0), std::out_of_range);
    CHECK_THROWS(t.at(0, 3), std::out_of_range);
    CHECK_THROWS(t.at(-1, 0), std::out_of_range);
    CHECK_THROWS(t.at(0), std::out_of_range);
    CHECK_THROWS(t.at(0, 0, 0), std::out_of_range);
    CHECK_THROWS(t.dim(2), std::out_of_range);
}

TEST(bracket_indexing_selects_a_sub_tensor) {
    I t = counting({2, 3});
    I row = t[1];
    CHECK(row.shape() == (Shape{3}));
    CHECK_EQ(row(2), 5);
    CHECK_EQ(t[-1](0), 3);
    CHECK_EQ(t[0][2].item(), 2);

    row(0) = 100;
    CHECK_EQ(t(1, 0), 100);

    CHECK_THROWS(t[2], std::out_of_range);
    CHECK_THROWS(t[-3], std::out_of_range);
    CHECK_THROWS(t[0][0][0], std::out_of_range);
}

TEST(reshape_shares_data) {
    I t = counting({2, 6});
    I r = t.reshape({3, 4});
    CHECK(r.shares_data_with(t));
    CHECK(r.is_contiguous());
    CHECK_EQ(r(2, 3), 11);
    r(0, 0) = 50;
    CHECK_EQ(t(0, 0), 50);
    CHECK_THROWS(t.reshape({5, 5}), std::invalid_argument);
}

TEST(reshape_can_infer_a_size) {
    const I t = counting({2, 6});
    CHECK(t.reshape({-1}).shape() == (Shape{12}));
    CHECK(t.reshape({4, -1}).shape() == (Shape{4, 3}));
    CHECK(t.reshape({-1, 2, 2}).shape() == (Shape{3, 2, 2}));
    CHECK_THROWS(t.reshape({-1, -1}), std::invalid_argument);
    CHECK_THROWS(t.reshape({5, -1}), std::invalid_argument);
}

TEST(reshape_accepts_variables) {
    const I t = counting({2, 6});
    const int rows = 3;
    const std::size_t cols = 4;
    CHECK(t.reshape({rows, rows + 1}).shape() == (Shape{3, 4}));
    CHECK(t.reshape({cols, cols - 1}).shape() == (Shape{4, 3}));
    CHECK(t.reshape(Shape{2, 2, 3}).rank() == 3);
}

TEST(reshape_copies_when_it_has_to) {
    const I t = counting({2, 3});
    const I r = t.transpose().reshape({6});
    CHECK(!r.shares_data_with(t));
    const std::vector<std::int32_t> expected = {0, 3, 1, 4, 2, 5};
    for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(r(i), expected[i]);
}

TEST(flatten_reads_in_logical_order) {
    const I flat = counting({2, 3}).flatten();
    CHECK(flat.shape() == (Shape{6}));
    CHECK_EQ(flat(4), 4);
    CHECK_EQ(counting({2, 3}).transpose().flatten()(1), 3);
}

TEST(transpose_swaps_without_copying) {
    I t = counting({2, 3});
    I flipped = t.transpose();
    CHECK(flipped.shape() == (Shape{3, 2}));
    CHECK(flipped.shares_data_with(t));
    CHECK(!flipped.is_contiguous());
    CHECK_EQ(flipped(2, 1), 5);
    CHECK_EQ(flipped(2, 0), 2);
    flipped(0, 1) = 77;
    CHECK_EQ(t(1, 0), 77);
    CHECK(flipped.transpose().is_contiguous());
}

TEST(transpose_of_higher_ranks) {
    const I t = counting({2, 3, 4});
    CHECK(t.transpose().shape() == (Shape{4, 3, 2}));
    CHECK_EQ(t.transpose()(3, 2, 1), t(1, 2, 3));
    const I swapped = t.transpose(0, 2);
    CHECK(swapped.shape() == (Shape{4, 3, 2}));
    CHECK_EQ(swapped(1, 2, 1), t(1, 2, 1));
    CHECK_EQ(t.transpose(1, 2)(1, 3, 2), t(1, 2, 3));
    CHECK_THROWS(t.transpose(0, 3), std::out_of_range);
}

TEST(permute_reorders_dimensions) {
    const I t = counting({2, 3, 4});
    const I p = t.permute({2, 0, 1});
    CHECK(p.shape() == (Shape{4, 2, 3}));
    CHECK_EQ(p(3, 1, 2), t(1, 2, 3));
    CHECK_THROWS(t.permute({0, 1}), std::invalid_argument);
    CHECK_THROWS(t.permute({0, 1, 1}), std::invalid_argument);
    CHECK_THROWS(t.permute({0, 1, 3}), std::out_of_range);
}

TEST(slice_selects_a_range) {
    I t = counting({4, 5});
    I middle = t.slice(0, 1, 3);
    CHECK(middle.shape() == (Shape{2, 5}));
    CHECK_EQ(middle(0, 0), 5);
    CHECK(middle.is_contiguous());
    I column = t.slice(1, 2, 3);
    CHECK(column.shape() == (Shape{4, 1}));
    CHECK_EQ(column(3, 0), 17);
    CHECK(!column.is_contiguous());

    middle(1, 1) = 99;
    CHECK_EQ(t(2, 1), 99);
}

TEST(slice_handles_steps_and_python_positions) {
    const I t = counting({10});
    const I every_third = t.slice(0, 1, 10, 3);
    CHECK_EQ(every_third.size(), 3u);
    CHECK_EQ(every_third(2), 7);
    CHECK(!every_third.is_contiguous());

    CHECK_EQ(t.slice(0, -3, 10)(0), 7);
    CHECK_EQ(t.slice(0, 2, -2).size(), 6u);
    CHECK_EQ(t.slice(0, 5, 100).size(), 5u);
    CHECK_EQ(t.slice(0, 7, 3).size(), 0u);
    CHECK_THROWS(t.slice(0, 0, 5, 0), std::invalid_argument);
    CHECK_THROWS(t.slice(0, 5, 0, -1), std::invalid_argument);
    CHECK_THROWS(t.slice(1, 0, 5), std::out_of_range);
}

TEST(flip_reverses_a_dimension) {
    const I t = counting({2, 3});
    const I reversed = t.flip(1);
    CHECK_EQ(reversed(0, 0), 2);
    CHECK_EQ(reversed(1, 2), 3);
    CHECK_EQ(t.flip(0)(0, 0), 3);
    CHECK_EQ(reversed.flip(1)(0, 0), 0);
    CHECK(mt::equal(reversed.clone(), reversed));
    CHECK_EQ(I({0, 2}).flip(0).size(), 0u);
}

TEST(squeeze_and_unsqueeze) {
    const I t = counting({1, 3, 1, 2});
    CHECK(t.squeeze().shape() == (Shape{3, 2}));
    CHECK_EQ(t.squeeze()(2, 1), 5);
    CHECK(t.squeeze(0).shape() == (Shape{3, 1, 2}));
    CHECK(t.squeeze(2).shape() == (Shape{1, 3, 2}));
    CHECK_EQ(t.squeeze(2)(0, 1, 1), 3);
    CHECK_THROWS(t.squeeze(1), std::invalid_argument);

    const I v = counting({3});
    CHECK(v.unsqueeze(0).shape() == (Shape{1, 3}));
    CHECK(v.unsqueeze(1).shape() == (Shape{3, 1}));
    CHECK_EQ(v.unsqueeze(1)(2, 0), 2);
    CHECK_THROWS(v.unsqueeze(2), std::out_of_range);
}

TEST(views_of_views_compose) {
    I t = counting({4, 4});
    I piece = t.transpose().slice(0, 1, 3).slice(1, 2, 4);
    CHECK(piece.shape() == (Shape{2, 2}));
    CHECK_EQ(piece(0, 0), t(2, 1));
    CHECK_EQ(piece(1, 1), t(3, 2));
    piece(1, 1) = -1;
    CHECK_EQ(t(3, 2), -1);
}

TEST(clone_of_a_view_is_contiguous) {
    const I t = counting({3, 4});
    const I copy = t.transpose().clone();
    CHECK(copy.is_contiguous());
    CHECK(!copy.shares_data_with(t));
    CHECK(mt::equal(copy, t.transpose()));
}

TEST(contiguous_only_copies_when_needed) {
    const I t = counting({3, 4});
    CHECK(t.contiguous().shares_data_with(t));
    CHECK(!t.transpose().contiguous().shares_data_with(t));
}

TEST(assign_writes_through_a_view) {
    I t({3, 3});
    t.slice(0, 1, 2).assign(I({1, 3}, {1, 2, 3}));
    CHECK_EQ(t(1, 2), 3);
    CHECK_EQ(t(0, 2), 0);

    t.slice(1, 0, 1).assign(I({1, 1}, 7));
    CHECK_EQ(t(2, 0), 7);
    CHECK_THROWS(t.assign(I::ones({2, 2})), std::invalid_argument);
}

TEST(assign_handles_overlapping_views) {
    I t = counting({6});
    t.slice(0, 1, 6).assign(t.slice(0, 0, 5));
    const I expected({6}, {0, 0, 1, 2, 3, 4});
    CHECK(mt::equal(t, expected));

    I m = counting({3, 3});
    m.assign(m.transpose());
    CHECK_EQ(m(0, 1), 3);
    CHECK_EQ(m(1, 0), 1);
}

TEST(fill_writes_through_a_view) {
    I t({3, 3});
    t.slice(1, 1, 2).fill(4);
    CHECK_EQ(t(2, 1), 4);
    CHECK_EQ(t.sum(), 12);
}

TEST_MAIN()
