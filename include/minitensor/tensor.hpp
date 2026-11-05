/// @file tensor.hpp
/// @brief The Tensor class: storage, indexing and views.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "minitensor/binary_op.hpp"
#include "minitensor/runtime.hpp"
#include "minitensor/shape.hpp"

namespace minitensor {

namespace detail {

template <typename T>
inline constexpr bool is_supported_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::int32_t> ||
    std::is_same_v<T, std::int64_t>;

/// A pointer plus layout. This is how tensors are passed to the compiled
/// part of the library.
template <typename T>
struct View {
    T* data;
    std::size_t rank;
    const std::size_t* dims;
    const std::ptrdiff_t* strides;
};

template <typename T>
View<const T> scalar_view(const T& value) noexcept {
    return {&value, 0, nullptr, nullptr};
}

// Implemented in src/ops.cpp and src/matmul.cpp for the supported types.
template <typename T>
void binary_into(BinaryOp op, View<T> out, View<const T> a, View<const T> b);
template <typename T>
void copy_into(View<T> out, View<const T> source);
template <typename T>
void fill_into(View<T> out, T value);
template <typename T>
T reduce_sum(const T* data, std::size_t size);
template <typename T>
T reduce_min(const T* data, std::size_t size);
template <typename T>
T reduce_max(const T* data, std::size_t size);
template <typename T>
T dot_product(const T* a, const T* b, std::size_t size);
template <typename T>
void gemm(std::size_t m, std::size_t n, std::size_t k, const T* a, std::ptrdiff_t rs_a,
          std::ptrdiff_t cs_a, const T* b, std::ptrdiff_t rs_b, std::ptrdiff_t cs_b, T* c,
          std::ptrdiff_t rs_c);
template <typename T>
void gemv(std::size_t m, std::size_t n, const T* a, std::ptrdiff_t rs_a, std::ptrdiff_t cs_a,
          const T* x, std::ptrdiff_t inc_x, T* y);

}  // namespace detail

/// A multidimensional array stored in one contiguous, 64-byte aligned block,
/// in row-major order.
///
/// `T` can be float, double, std::int32_t or std::int64_t.
///
/// **Copies and views.** Copying a Tensor copies its data, like std::vector.
/// `reshape`, `transpose`, `permute`, `slice`, `flip`, `squeeze`, `unsqueeze`,
/// `flatten`, `view` and `operator[]` return *views* instead: new tensors that
/// share the same data, so writing through one changes the other. The data
/// stays alive until the last tensor using it is destroyed.
template <typename T>
class Tensor {
    static_assert(detail::is_supported_v<T>,
                  "Tensor<T> supports float, double, std::int32_t and std::int64_t");

public:
    using value_type = T;

    /// @name Creation
    /// @{

    /// Creates an empty tensor of shape `(0)`.
    Tensor() noexcept : shape_(Shape::empty()) {}

    /// Creates a tensor filled with zeros.
    explicit Tensor(const Shape& shape) : shape_(shape), size_(shape.num_elements()) {
        adopt(static_cast<T*>(detail::aligned_calloc(checked_bytes())));
    }

    /// Creates a tensor with every element set to `value`.
    Tensor(const Shape& shape, T value) : Tensor(uninitialized(shape)) { fill(value); }

    /// Creates a tensor from a list of values in row-major order.
    Tensor(const Shape& shape, std::initializer_list<T> values) : Tensor(uninitialized(shape)) {
        if (values.size() != size_) {
            throw std::invalid_argument("minitensor: " + std::to_string(values.size()) +
                                        " values given for shape " + shape.to_string());
        }
        std::copy(values.begin(), values.end(), data_);
    }

    /// Creates a tensor without initializing its elements. This is the
    /// fastest way to allocate a tensor that will be overwritten.
    static Tensor uninitialized(const Shape& shape) {
        Tensor out;
        out.shape_ = shape;
        out.size_ = shape.num_elements();
        out.adopt(static_cast<T*>(detail::aligned_malloc(out.checked_bytes())));
        return out;
    }

    static Tensor zeros(const Shape& shape) { return Tensor(shape); }
    static Tensor ones(const Shape& shape) { return Tensor(shape, T(1)); }
    static Tensor full(const Shape& shape, T value) { return Tensor(shape, value); }

    /// Creates a 1-D tensor of `start, start + step, ...` stopping before `stop`.
    static Tensor arange(T start, T stop, T step = T(1)) {
        if (step == T(0)) throw std::invalid_argument("minitensor: arange step cannot be 0");
        const double span =
            (static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(step);
        const std::size_t count = span > 0 ? static_cast<std::size_t>(std::ceil(span)) : 0;
        Tensor out = uninitialized(Shape{count});
        for (std::size_t i = 0; i < count; ++i) {
            out.data_[i] = static_cast<T>(start + static_cast<T>(i) * step);
        }
        return out;
    }

    /// Creates a 1-D tensor of `count` evenly spaced values from `start` to
    /// `stop`, both included.
    static Tensor linspace(T start, T stop, std::size_t count) {
        Tensor out = uninitialized(Shape{count});
        const double step = count > 1 ? (static_cast<double>(stop) - static_cast<double>(start)) /
                                            static_cast<double>(count - 1)
                                      : 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            out.data_[i] =
                static_cast<T>(static_cast<double>(start) + static_cast<double>(i) * step);
        }
        if (count > 1) out.data_[count - 1] = stop;
        return out;
    }

    /// Creates an `n` by `n` identity matrix.
    static Tensor eye(std::size_t n) {
        Tensor out(Shape{n, n});
        for (std::size_t i = 0; i < n; ++i) out.data_[i * n + i] = T(1);
        return out;
    }

    /// Creates a tensor of uniform random values in `[0, 1)`. For integer
    /// tensors, the values are 0 or 1.
    static Tensor rand(const Shape& shape, std::uint64_t seed = 42) {
        Tensor out = uninitialized(shape);
        std::mt19937_64 generator(seed);
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> distribution(T(0), T(1));
            for (std::size_t i = 0; i < out.size_; ++i) out.data_[i] = distribution(generator);
        } else {
            std::uniform_int_distribution<T> distribution(T(0), T(1));
            for (std::size_t i = 0; i < out.size_; ++i) out.data_[i] = distribution(generator);
        }
        return out;
    }

    /// Creates a tensor of normally distributed random values, with mean 0
    /// and standard deviation 1. Only available for float and double.
    static Tensor randn(const Shape& shape, std::uint64_t seed = 42) {
        static_assert(std::is_floating_point_v<T>, "randn needs a floating point tensor");
        Tensor out = uninitialized(shape);
        std::mt19937_64 generator(seed);
        std::normal_distribution<T> distribution(T(0), T(1));
        for (std::size_t i = 0; i < out.size_; ++i) out.data_[i] = distribution(generator);
        return out;
    }

    /// Creates a tensor by copying `shape.num_elements()` values from `values`.
    static Tensor from_data(const Shape& shape, const T* values) {
        Tensor out = uninitialized(shape);
        std::copy(values, values + out.size_, out.data_);
        return out;
    }

    /// Creates a tensor that uses existing memory without copying it. The
    /// caller keeps ownership and must keep the memory alive.
    static Tensor wrap(T* values, const Shape& shape) {
        Tensor out;
        out.shape_ = shape;
        out.size_ = shape.num_elements();
        // An empty owner gives a shared_ptr that never frees `values`.
        out.storage_ = std::shared_ptr<T>(std::shared_ptr<T>(), values);
        out.data_ = values;
        out.compute_strides();
        return out;
    }

    /// Copies the data. See the class notes on copies and views.
    Tensor(const Tensor& other) : Tensor(uninitialized(other.shape_)) {
        detail::copy_into<T>(ref(), other.cref());
    }

    /// Takes over the data of `other`, leaving it empty. Costs no allocation.
    Tensor(Tensor&& other) noexcept
        : storage_(std::move(other.storage_)),
          data_(other.data_),
          shape_(other.shape_),
          strides_(other.strides_),
          size_(other.size_) {
        other.clear();
    }

    /// Replaces this tensor with a copy of `other`.
    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;
        if (is_unique() && is_contiguous() && shape_ == other.shape_) {
            detail::copy_into<T>(ref(), other.cref());
        } else {
            Tensor copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    /// Takes over the data of `other`, leaving it empty.
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        storage_ = std::move(other.storage_);
        data_ = other.data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        size_ = other.size_;
        other.clear();
        return *this;
    }

    ~Tensor() = default;

    /// @}
    /// @name Properties
    /// @{

    /// Number of dimensions.
    std::size_t rank() const noexcept { return shape_.rank(); }

    /// Size of every dimension.
    const Shape& shape() const noexcept { return shape_; }

    /// Distance between neighbouring elements of every dimension.
    const Strides& strides() const noexcept { return strides_; }

    /// Total number of elements.
    std::size_t size() const noexcept { return size_; }

    /// True if the tensor has no elements.
    bool empty() const noexcept { return size_ == 0; }

    /// Size of dimension `d`. Throws std::out_of_range for a bad dimension.
    std::size_t dim(std::size_t d) const {
        check_dim(d);
        return shape_[d];
    }

    /// Pointer to the first element.
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    /// True if the elements are next to each other in row-major order.
    bool is_contiguous() const noexcept {
        std::ptrdiff_t expected = 1;
        for (std::size_t i = rank(); i-- > 0;) {
            if (shape_[i] == 1) continue;
            if (strides_[i] != expected) return false;
            expected *= static_cast<std::ptrdiff_t>(shape_[i]);
        }
        return true;
    }

    /// True if this is the only tensor using its data.
    bool is_unique() const noexcept { return storage_.use_count() == 1; }

    /// True if both tensors use the same block of memory.
    bool shares_data_with(const Tensor& other) const noexcept {
        return storage_ && storage_.get() == other.storage_.get();
    }

    /// @}
    /// @name Element access
    /// @{

    /// Returns the element at the given indices, e.g. `t(1, 2)`.
    ///
    /// The indices aren't checked in release builds, which keeps this as fast
    /// as raw pointer arithmetic. Use `at` for checked access.
    template <typename... Index>
    T& operator()(Index... index) noexcept {
        return data_[offset(index...)];
    }

    template <typename... Index>
    const T& operator()(Index... index) const noexcept {
        return data_[offset(index...)];
    }

    /// Like `operator()`, but throws std::out_of_range for a bad index.
    template <typename... Index>
    T& at(Index... index) {
        return data_[checked_offset(index...)];
    }

    template <typename... Index>
    const T& at(Index... index) const {
        return data_[checked_offset(index...)];
    }

    /// Returns the value of a tensor with exactly one element.
    T item() const {
        if (size_ != 1) {
            throw std::invalid_argument("minitensor: item() needs a tensor with one element");
        }
        return *data_;
    }

    /// Returns a view of index `i` of the first dimension. Negative indices
    /// count from the end.
    Tensor operator[](std::ptrdiff_t i) const {
        if (rank() == 0) throw std::out_of_range("minitensor: cannot index a rank-0 tensor");
        const std::ptrdiff_t length = static_cast<std::ptrdiff_t>(shape_[0]);
        if (i < 0) i += length;
        if (i < 0 || i >= length) {
            throw std::out_of_range("minitensor: index " + std::to_string(i) +
                                    " is out of range for shape " + shape_.to_string());
        }
        Strides strides{};
        std::copy(strides_.begin() + 1, strides_.begin() + rank(), strides.begin());
        return Tensor(storage_, data_ + i * strides_[0], shape_.without(0), strides,
                      size_ / shape_[0]);
    }

    /// @}
    /// @name Views
    /// @{

    /// Returns a view of the whole tensor.
    Tensor view() const { return Tensor(storage_, data_, shape_, strides_, size_); }

    /// Returns the same elements with a new shape. The result is a view when
    /// this tensor is contiguous, and a copy otherwise.
    Tensor reshape(const Shape& shape) const {
        if (shape.num_elements() != size_) {
            throw std::invalid_argument("minitensor: cannot reshape " + shape_.to_string() +
                                        " to " + shape.to_string());
        }
        if (!is_contiguous()) return contiguous().reshape(shape);
        return Tensor(storage_, data_, shape, row_major_strides(shape), size_);
    }

    /// Like `reshape(const Shape&)`, but one size can be -1, which means "work
    /// it out from the number of elements", e.g. `t.reshape({2, -1})`.
    Tensor reshape(std::initializer_list<Dim> dims) const {
        std::vector<std::size_t> sizes;
        std::size_t known = 1;
        std::size_t inferred = dims.size();
        for (const Dim d : dims) {
            if (d.value() == -1) {
                if (inferred != dims.size()) {
                    throw std::invalid_argument("minitensor: reshape can only infer one size");
                }
                inferred = sizes.size();
                sizes.push_back(1);
                continue;
            }
            sizes.push_back(Shape{d}[0]);
            known *= sizes.back();
        }
        if (inferred != dims.size()) {
            if (known == 0 || size_ % known != 0) {
                throw std::invalid_argument("minitensor: cannot infer a size to reshape " +
                                            shape_.to_string());
            }
            sizes[inferred] = size_ / known;
        }
        return reshape(Shape(sizes));
    }

    /// Returns a 1-D version of this tensor. It's a view when contiguous.
    Tensor flatten() const { return reshape(Shape{size_}); }

    /// Returns a view with the dimensions in reverse order. For a matrix,
    /// that's the transpose.
    Tensor transpose() const {
        std::vector<std::size_t> order(rank());
        for (std::size_t i = 0; i < rank(); ++i) order[i] = rank() - 1 - i;
        return permute(order);
    }

    /// Returns a view with dimensions `d0` and `d1` swapped.
    Tensor transpose(std::size_t d0, std::size_t d1) const {
        check_dim(d0);
        check_dim(d1);
        std::vector<std::size_t> order(rank());
        for (std::size_t i = 0; i < rank(); ++i) order[i] = i;
        std::swap(order[d0], order[d1]);
        return permute(order);
    }

    /// Returns a view whose dimension `i` is this tensor's `order[i]`.
    Tensor permute(const std::vector<std::size_t>& order) const {
        if (order.size() != rank()) {
            throw std::invalid_argument("minitensor: permute needs one entry per dimension");
        }
        std::vector<std::size_t> dims(rank());
        Strides strides{};
        std::vector<bool> seen(rank(), false);
        for (std::size_t i = 0; i < rank(); ++i) {
            const std::size_t from = order[i];
            check_dim(from);
            if (seen[from]) throw std::invalid_argument("minitensor: permute repeats a dimension");
            seen[from] = true;
            dims[i] = shape_[from];
            strides[i] = strides_[from];
        }
        return Tensor(storage_, data_, Shape(dims), strides, size_);
    }

    /// Returns a view of `start, start + step, ...` before `stop` along
    /// dimension `dim`. Negative positions count from the end, and positions
    /// past the end are clamped, as in Python.
    Tensor slice(std::size_t dim, std::ptrdiff_t start, std::ptrdiff_t stop,
                 std::ptrdiff_t step = 1) const {
        check_dim(dim);
        if (step <= 0) throw std::invalid_argument("minitensor: slice step must be positive");
        const std::ptrdiff_t length = static_cast<std::ptrdiff_t>(shape_[dim]);
        if (start < 0) start += length;
        if (stop < 0) stop += length;
        start = std::clamp<std::ptrdiff_t>(start, 0, length);
        stop = std::clamp<std::ptrdiff_t>(stop, 0, length);
        const std::size_t count =
            stop > start ? static_cast<std::size_t>((stop - start + step - 1) / step) : 0;
        Strides strides = strides_;
        strides[dim] *= step;
        const Shape shape = shape_.with(dim, count);
        return Tensor(storage_, data_ + start * strides_[dim], shape, strides,
                      shape.num_elements());
    }

    /// Returns a view with the order of dimension `dim` reversed.
    Tensor flip(std::size_t dim) const {
        check_dim(dim);
        Strides strides = strides_;
        strides[dim] = -strides[dim];
        const std::ptrdiff_t last = static_cast<std::ptrdiff_t>(shape_[dim]) - 1;
        T* first = size_ == 0 ? data_ : data_ + last * strides_[dim];
        return Tensor(storage_, first, shape_, strides, size_);
    }

    /// Returns a view with a new dimension of size 1 inserted at `dim`.
    Tensor unsqueeze(std::size_t dim) const {
        if (dim > rank()) throw std::out_of_range("minitensor: unsqueeze dimension out of range");
        std::vector<std::size_t> dims(shape_.begin(), shape_.end());
        dims.insert(dims.begin() + static_cast<std::ptrdiff_t>(dim), 1);
        const Shape shape(dims);
        Strides strides{};
        for (std::size_t i = 0; i < rank(); ++i) strides[i < dim ? i : i + 1] = strides_[i];
        strides[dim] = 1;
        return Tensor(storage_, data_, shape, strides, size_);
    }

    /// Returns a view without any dimensions of size 1.
    Tensor squeeze() const {
        std::vector<std::size_t> dims;
        Strides strides{};
        for (std::size_t i = 0; i < rank(); ++i) {
            if (shape_[i] == 1) continue;
            strides[dims.size()] = strides_[i];
            dims.push_back(shape_[i]);
        }
        return Tensor(storage_, data_, Shape(dims), strides, size_);
    }

    /// Returns a view without dimension `dim`, which must have size 1.
    Tensor squeeze(std::size_t dim) const {
        check_dim(dim);
        if (shape_[dim] != 1) {
            throw std::invalid_argument("minitensor: can only squeeze a dimension of size 1");
        }
        Strides strides{};
        for (std::size_t i = 0, kept = 0; i < rank(); ++i) {
            if (i != dim) strides[kept++] = strides_[i];
        }
        return Tensor(storage_, data_, shape_.without(dim), strides, size_);
    }

    /// Returns this tensor if it's contiguous, as a view, and a copy if not.
    Tensor contiguous() const { return is_contiguous() ? view() : Tensor(*this); }

    /// @}
    /// @name Modifying
    /// @{

    /// Returns an independent copy of the data.
    Tensor clone() const { return Tensor(*this); }

    /// Sets every element to `value`.
    Tensor& fill(T value) {
        detail::fill_into<T>(ref(), value);
        return *this;
    }

    /// Copies `source` into this tensor's existing elements, broadcasting it
    /// if needed. This writes through views, e.g. `t.slice(0, 0, 2).assign(x)`.
    Tensor& assign(const Tensor& source) {
        require_broadcast_into_this(source);
        if (overlaps_with_other_layout(source)) {
            const Tensor copy(source);
            detail::copy_into<T>(ref(), copy.cref());
        } else {
            detail::copy_into<T>(ref(), source.cref());
        }
        return *this;
    }

    Tensor& operator+=(const Tensor& rhs) { return update(detail::BinaryOp::Add, rhs); }
    Tensor& operator-=(const Tensor& rhs) { return update(detail::BinaryOp::Sub, rhs); }
    Tensor& operator*=(const Tensor& rhs) { return update(detail::BinaryOp::Mul, rhs); }
    Tensor& operator/=(const Tensor& rhs) { return update(detail::BinaryOp::Div, rhs); }
    Tensor& operator+=(T rhs) { return update(detail::BinaryOp::Add, rhs); }
    Tensor& operator-=(T rhs) { return update(detail::BinaryOp::Sub, rhs); }
    Tensor& operator*=(T rhs) { return update(detail::BinaryOp::Mul, rhs); }
    Tensor& operator/=(T rhs) { return update(detail::BinaryOp::Div, rhs); }

    /// @}
    /// @name Reductions
    /// @{

    /// Adds all elements.
    T sum() const {
        const Tensor flat = contiguous();
        return detail::reduce_sum<T>(flat.data_, flat.size_);
    }

    /// Returns the average of all elements.
    double mean() const { return static_cast<double>(sum()) / static_cast<double>(size_); }

    /// Returns the smallest element. Throws std::invalid_argument if empty.
    T min() const {
        require_not_empty("min");
        const Tensor flat = contiguous();
        return detail::reduce_min<T>(flat.data_, flat.size_);
    }

    /// Returns the largest element. Throws std::invalid_argument if empty.
    T max() const {
        require_not_empty("max");
        const Tensor flat = contiguous();
        return detail::reduce_max<T>(flat.data_, flat.size_);
    }

    /// Returns the position, in row-major order, of the largest element.
    std::size_t argmax() const {
        require_not_empty("argmax");
        const Tensor flat = contiguous();
        return static_cast<std::size_t>(std::max_element(flat.data_, flat.data_ + flat.size_) -
                                        flat.data_);
    }

    /// Adds elements along dimension `dim`, removing that dimension.
    Tensor sum(std::size_t dim) const {
        check_dim(dim);
        Tensor out(shape_.without(dim));
        for (std::size_t i = 0; i < shape_[dim]; ++i) {
            const std::ptrdiff_t index = static_cast<std::ptrdiff_t>(i);
            out += slice(dim, index, index + 1).squeeze(dim);
        }
        return out;
    }

    /// Averages elements along dimension `dim`, removing that dimension.
    Tensor mean(std::size_t dim) const {
        Tensor out = sum(dim);
        out /= static_cast<T>(shape_[dim]);
        return out;
    }

    /// @}
    /// @name Conversion
    /// @{

    /// Returns a copy with elements converted to type `U`.
    template <typename U>
    Tensor<U> cast() const {
        const Tensor flat = contiguous();
        Tensor<U> out = Tensor<U>::uninitialized(shape_);
        for (std::size_t i = 0; i < size_; ++i) out.data()[i] = static_cast<U>(flat.data_[i]);
        return out;
    }

    /// @}

    /// Layout of this tensor for the compiled part of the library.
    detail::View<T> ref() noexcept { return {data_, rank(), shape_.data(), strides_.data()}; }
    detail::View<const T> cref() const noexcept {
        return {data_, rank(), shape_.data(), strides_.data()};
    }

private:
    Tensor(std::shared_ptr<T> storage, T* data, const Shape& shape, const Strides& strides,
           std::size_t size) noexcept
        : storage_(std::move(storage)),
          data_(data),
          shape_(shape),
          strides_(strides),
          size_(size) {}

    static Strides row_major_strides(const Shape& shape) noexcept {
        Strides strides{};
        std::ptrdiff_t stride = 1;
        for (std::size_t i = shape.rank(); i-- > 0;) {
            strides[i] = stride;
            stride *= static_cast<std::ptrdiff_t>(shape[i]);
        }
        return strides;
    }

    std::size_t checked_bytes() const {
        std::size_t bytes = 0;
        if (__builtin_mul_overflow(size_, sizeof(T), &bytes)) {
            throw std::length_error("minitensor: tensor shape " + shape_.to_string() +
                                    " is too large");
        }
        return bytes;
    }

    void compute_strides() noexcept { strides_ = row_major_strides(shape_); }

    void adopt(T* buffer) {
        std::shared_ptr<T> owner;
        try {
            owner = std::shared_ptr<T>(buffer, [](T* p) { detail::aligned_free(p); });
        } catch (...) {
            detail::aligned_free(buffer);
            throw;
        }
        storage_ = std::move(owner);
        data_ = buffer;
        compute_strides();
    }

    void clear() noexcept {
        data_ = nullptr;
        shape_ = Shape::empty();
        strides_ = Strides{};
        size_ = 0;
    }

    void check_dim(std::size_t d) const {
        if (d >= rank()) {
            throw std::out_of_range("minitensor: dimension " + std::to_string(d) +
                                    " is out of range for shape " + shape_.to_string());
        }
    }

    void require_not_empty(const char* operation) const {
        if (size_ == 0) {
            throw std::invalid_argument(std::string("minitensor: ") + operation +
                                        " of an empty tensor");
        }
    }

    template <typename... Index>
    std::ptrdiff_t offset(Index... index) const noexcept {
        static_assert((std::is_integral_v<Index> && ...), "tensor indices must be integers");
        assert(sizeof...(Index) == rank());
        std::ptrdiff_t position = 0;
        std::size_t d = 0;
        ((position += static_cast<std::ptrdiff_t>(index) * strides_[d++]), ...);
        return position;
    }

    template <typename... Index>
    std::ptrdiff_t checked_offset(Index... index) const {
        static_assert((std::is_integral_v<Index> && ...), "tensor indices must be integers");
        if (sizeof...(Index) != rank()) {
            throw std::out_of_range("minitensor: " + std::to_string(sizeof...(Index)) +
                                    " indices given for shape " + shape_.to_string());
        }
        std::size_t d = 0;
        const bool valid = ((static_cast<std::ptrdiff_t>(index) >= 0 &&
                             static_cast<std::size_t>(index) < shape_[d++]) &&
                            ...);
        if (!valid) {
            throw std::out_of_range("minitensor: index is out of range for shape " +
                                    shape_.to_string());
        }
        return offset(index...);
    }

    void require_broadcast_into_this(const Tensor& source) const {
        if (broadcast_shapes(shape_, source.shape_) != shape_) {
            throw std::invalid_argument("minitensor: shape " + source.shape_.to_string() +
                                        " cannot be written into shape " + shape_.to_string());
        }
    }

    // Writing through one layout while reading through another would read
    // elements that have already been overwritten.
    bool overlaps_with_other_layout(const Tensor& other) const noexcept {
        if (!shares_data_with(other)) return false;
        return data_ != other.data_ || shape_ != other.shape_ || strides_ != other.strides_;
    }

    Tensor& update(detail::BinaryOp op, const Tensor& rhs) {
        require_broadcast_into_this(rhs);
        if (overlaps_with_other_layout(rhs)) {
            const Tensor copy(rhs);
            detail::binary_into<T>(op, ref(), cref(), copy.cref());
        } else {
            detail::binary_into<T>(op, ref(), cref(), rhs.cref());
        }
        return *this;
    }

    Tensor& update(detail::BinaryOp op, T rhs) {
        detail::binary_into<T>(op, ref(), cref(), detail::scalar_view(rhs));
        return *this;
    }

    std::shared_ptr<T> storage_;
    T* data_ = nullptr;
    Shape shape_;
    Strides strides_{};
    std::size_t size_ = 0;
};

}  // namespace minitensor
