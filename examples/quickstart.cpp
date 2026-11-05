// A tour of the library's main features.
#include <iostream>

#include "minitensor/minitensor.hpp"

int main() {
    using Tensor = mt::Tensor<double>;

    // Creating tensors.
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor zeros({2, 3});
    Tensor ones = Tensor::ones({2, 3});
    Tensor counting = Tensor::arange(0, 12);
    std::cout << "a =\n" << a << "\n\n";

    // Reading and changing elements.
    zeros(1, 2) = 42.0;
    std::cout << "a(1, 2) = " << a(1, 2) << ", zeros(1, 2) = " << zeros(1, 2) << "\n\n";

    // Element-wise arithmetic.
    std::cout << "a + ones =\n" << a + ones << "\n\n";
    std::cout << "a - 1 =\n" << a - 1 << "\n\n";
    std::cout << "a * a =\n" << a * a << "\n\n";

    // Broadcasting: the row is stretched to match each row of `a`.
    Tensor row({3}, {10, 20, 30});
    std::cout << "a + row =\n" << a + row << "\n\n";

    // Matrix multiplication.
    Tensor b({3, 2}, {1, 0, 0, 1, 1, 1});
    std::cout << "matmul(a, b) =\n" << mt::matmul(a, b) << "\n\n";

    // Views: these share memory with the original, so no data is copied.
    Tensor grid = counting.reshape({3, 4});
    Tensor flipped = grid.transpose();
    Tensor middle = grid.slice(0, 1, 2);
    middle.fill(-1);
    std::cout << "grid after filling its middle row =\n" << grid << "\n\n";
    std::cout << "the same data, transposed =\n" << flipped << "\n\n";

    // Summaries.
    std::cout << "sum = " << a.sum() << ", mean = " << a.mean() << ", max = " << a.max() << "\n";
    std::cout << "sum of each column = " << a.sum(0) << "\n\n";

    std::cout << "Running with " << mt::simd_backend() << " on " << mt::num_threads()
              << " threads\n";
    return 0;
}
