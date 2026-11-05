// Fits a straight-line model to noisy data with gradient descent.
#include <iostream>

#include "minitensor/minitensor.hpp"

int main() {
    using Tensor = mt::Tensor<double>;

    // Make up some data where y = 2 * x0 - 3 * x1 + 5, plus a little noise.
    const std::size_t samples = 100000;
    const Tensor x = Tensor::randn({samples, 2}, 1);
    const Tensor true_weights({2, 1}, {2.0, -3.0});
    const Tensor noise = Tensor::randn({samples, 1}, 2) * 0.1;
    const Tensor y = mt::matmul(x, true_weights) + 5.0 + noise;

    Tensor weights({2, 1});
    double bias = 0.0;
    const double learning_rate = 0.1;

    for (int step = 0; step <= 100; ++step) {
        const Tensor error = mt::matmul(x, weights) + bias - y;
        const Tensor weight_gradient = mt::matmul(x.transpose(), error) * (2.0 / samples);
        const double bias_gradient = 2.0 * error.mean();

        weights -= weight_gradient * learning_rate;
        bias -= learning_rate * bias_gradient;

        if (step % 25 == 0) {
            std::cout << "step " << step << ": loss = " << (error * error).mean() << "\n";
        }
    }

    std::cout << "\nlearned weights = " << weights.flatten() << ", bias = " << bias
              << "\nexpected about [2, -3] and 5\n";
    return 0;
}
