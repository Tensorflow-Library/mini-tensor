#include <iostream>
#include <minitensor/minitensor.hpp>

int main() {
    const mt::Tensor<double> a({2, 2}, {1, 2, 3, 4});
    std::cout << mt::matmul(a, a.transpose()) << "\n";
    return 0;
}
